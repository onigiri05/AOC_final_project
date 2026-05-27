`timescale 1ns/1ps
// flash_attn_wrapper.sv — FlashAttention-2 with INT8 DMA Packing + K/V Ping-Pong
//
// ── INT8 DMA Packing ──────────────────────────────────────────────────────────
// Q, K, V are stored INT8 in DRAM (4 elements packed per 32-bit AXI word).
//   DMA read burst length: ARLEN_M = d/4 − 1  (e.g., d=64 → ARLEN=15 vs 63 before → 4× less DMA)
//   Row stride in DRAM: d bytes (not d×4 bytes).
//   Global symmetric scales set via MMIO before start; hardware uses them for
//   dequantization after GEMM1/GEMM2.  p_scale is still computed on-chip
//   (softmax output) and is per j-tile.
//
// ── K/V Ping-Pong ─────────────────────────────────────────────────────────────
// Two INT8 K buffers (K_buf_A / K_buf_B) and two INT8 V buffers (V_buf_A / V_buf_B).
// pp_sel: 0 = A is the "active" (SA reads from it), B is the "prefetch" target.
//         1 = B is the "active", A is the "prefetch" target.
//
// Prefetch sub-FSM (pf_state) runs concurrently with SA computation inside the
// same always_ff block.  It uses the AXI master AR channel during states where
// the main FSM leaves it idle (FA_SA_G1_WT, FA_SA_SFX, FA_SA_G2_*, FA_NEXT_J).
// When the prefetch of K/V for j+1 is done (pf_state = PF_DONE), FA_NEXT_J
// swaps pp_sel and proceeds directly to FA_SA_G1_LD — no DMA_K/DMA_V for j>0.
//
// ── Register map (base 0x10050000) ───────────────────────────────────────────
//   0x00  FA_CONTROL  [0]=start, [1]=irq_clear
//   0x04  FA_SHAPE    [31:16]=N, [15:0]=d
//   0x08  FA_TILE     [15:0]=Br (= Bc = SA_SIZE)
//   0x0C  FA_Q_ADDR   INT8-packed Q base; row stride = d bytes
//   0x10  FA_K_ADDR   INT8-packed K base
//   0x14  FA_V_ADDR   INT8-packed V base
//   0x18  FA_O_ADDR   fp32 O base; row stride = d×4 bytes
//   0x1C  FA_STATUS   [0]=busy, [1]=done (read-only)
//   0x20  FA_Q_SCALE  Global Q scale (IEEE fp32 bits)
//   0x24  FA_K_SCALE  Global K scale (IEEE fp32 bits)
//   0x28  FA_V_SCALE  Global V scale (IEEE fp32 bits)
//
// ── Compute pipeline ─────────────────────────────────────────────────────────
//   FA_SA_G1_LD : copy Q_int8/K_int8 → sa_a/b_reg               (1 cycle)
//   FA_SA_G1_ST : pulse sa_start                                 (1 cycle)
//   FA_SA_G1_WT : wait GEMM1, dequant S=INT32×q_scale×k_scale   (~90 cycles)
//   FA_SA_SFX   : online softmax row by row, compute p_scale     (Br cycles)
//   FA_SA_G2_LD : load P_int8/V_chunk_int8 → sa_a/b_reg         (1 cycle)
//   FA_SA_G2_ST : pulse sa_start                                 (1 cycle)
//   FA_SA_G2_WT : wait GEMM2 chunk                               (~40 cycles)
//   FA_SA_G2_NX : dequant ΔO=INT32×p_scale×v_scale, accumulate  (1 cycle)
//   (repeat G2_LD→G2_NX for ceil(d/Br) chunks)
//   FA_NEXT_J   : wait pf_state==PF_DONE, swap pp_sel, loop

module flash_attn_wrapper #(
    parameter MAX_N    = 256,
    parameter MAX_D    = 64,
    parameter MAX_TILE = 64,
    parameter SA_SIZE  = 14
) (
    input  logic        ACLK,
    input  logic        ARESETn,
    output logic        FA_interrupt,

    // AXI4 Slave (MMIO)
    input  logic [3:0]  AWID_S,
    input  logic [31:0] AWADDR_S,
    input  logic [7:0]  AWLEN_S,
    input  logic [2:0]  AWSIZE_S,
    input  logic [1:0]  AWBURST_S,
    input  logic        AWVALID_S,
    output logic        AWREADY_S,

    input  logic [31:0] WDATA_S,
    input  logic [3:0]  WSTRB_S,
    input  logic        WLAST_S,
    input  logic        WVALID_S,
    output logic        WREADY_S,

    output logic [3:0]  BID_S,
    output logic [1:0]  BRESP_S,
    output logic        BVALID_S,
    input  logic        BREADY_S,

    input  logic [3:0]  ARID_S,
    input  logic [31:0] ARADDR_S,
    input  logic [7:0]  ARLEN_S,
    input  logic [2:0]  ARSIZE_S,
    input  logic [1:0]  ARBURST_S,
    input  logic        ARVALID_S,
    output logic        ARREADY_S,

    output logic [3:0]  RID_S,
    output logic [31:0] RDATA_S,
    output logic [1:0]  RRESP_S,
    output logic        RLAST_S,
    output logic        RVALID_S,
    input  logic        RREADY_S,

    // AXI4 Master (DMA)
    output logic [3:0]  ARID_M,
    output logic [31:0] ARADDR_M,
    output logic [7:0]  ARLEN_M,
    output logic [2:0]  ARSIZE_M,
    output logic [1:0]  ARBURST_M,
    output logic        ARVALID_M,
    input  logic        ARREADY_M,

    input  logic [3:0]  RID_M,
    input  logic [31:0] RDATA_M,
    input  logic [1:0]  RRESP_M,
    input  logic        RLAST_M,
    input  logic        RVALID_M,
    output logic        RREADY_M,

    output logic [3:0]  AWID_M,
    output logic [31:0] AWADDR_M,
    output logic [7:0]  AWLEN_M,
    output logic [2:0]  AWSIZE_M,
    output logic [1:0]  AWBURST_M,
    output logic        AWVALID_M,
    input  logic        AWREADY_M,

    output logic [31:0] WDATA_M,
    output logic [3:0]  WSTRB_M,
    output logic        WLAST_M,
    output logic        WVALID_M,
    input  logic        WREADY_M,

    input  logic [3:0]  BID_M,
    input  logic [1:0]  BRESP_M,
    input  logic        BVALID_M,
    output logic        BREADY_M
);

// ── DPI-C: fp32 ↔ double, softmax helpers ─────────────────────────────────
import "DPI-C" function real dpi_fp32_bits_to_real(input int unsigned bits);
import "DPI-C" function int unsigned dpi_real_to_fp32_bits(input real val);
import "DPI-C" function real dpi_expf(input real x);
import "DPI-C" function real dpi_sqrtf(input real x);

localparam AXI_RESP_OKAY   = 2'b00;
localparam AXI_RESP_SLVERR = 2'b10;

localparam [31:0] MMIO_BASE   = 32'h10050000;
localparam [31:0] REG_CONTROL = MMIO_BASE + 32'h00;
localparam [31:0] REG_SHAPE   = MMIO_BASE + 32'h04;
localparam [31:0] REG_TILE    = MMIO_BASE + 32'h08;
localparam [31:0] REG_Q_ADDR  = MMIO_BASE + 32'h0C;
localparam [31:0] REG_K_ADDR  = MMIO_BASE + 32'h10;
localparam [31:0] REG_V_ADDR  = MMIO_BASE + 32'h14;
localparam [31:0] REG_O_ADDR  = MMIO_BASE + 32'h18;
localparam [31:0] REG_STATUS  = MMIO_BASE + 32'h1C;
localparam [31:0] REG_Q_SCALE = MMIO_BASE + 32'h20;
localparam [31:0] REG_K_SCALE = MMIO_BASE + 32'h24;
localparam [31:0] REG_V_SCALE = MMIO_BASE + 32'h28;

// ── MMIO registers ────────────────────────────────────────────────────────
logic [31:0] reg_control;
logic [31:0] reg_shape;
logic [31:0] reg_tile;
logic [31:0] reg_q_addr;
logic [31:0] reg_k_addr;
logic [31:0] reg_v_addr;
logic [31:0] reg_o_addr;
logic        busy_r, done_r;

// Global quantization scales (fp64, set from MMIO fp32 bits via DPI-C)
real reg_q_scale, reg_k_scale, reg_v_scale;

wire [15:0] N_w  = reg_shape[31:16];
wire [15:0] d_w  = reg_shape[15:0];
wire [15:0] Br_w = reg_tile[15:0];

// ── INT8 tile buffers ──────────────────────────────────────────────────────
// Q: one i-tile, loaded from DMA each i-iteration
byte signed Q_buf   [0:MAX_TILE-1][0:MAX_D-1];

// K/V: ping-pong pair; pp_sel selects which is active (A=0, B=1)
byte signed K_buf_A [0:MAX_TILE-1][0:MAX_D-1];
byte signed K_buf_B [0:MAX_TILE-1][0:MAX_D-1];
byte signed V_buf_A [0:MAX_TILE-1][0:MAX_D-1];
byte signed V_buf_B [0:MAX_TILE-1][0:MAX_D-1];

// ── fp64 on-chip buffers (softmax / accumulation) ─────────────────────────
real O_buf [0:MAX_TILE-1][0:MAX_D-1];   // accumulates across j-tiles
real l_buf [0:MAX_TILE-1];
real m_buf [0:MAX_TILE-1];
real P_buf [0:MAX_TILE-1][0:MAX_TILE-1]; // softmax output (blocking = in SFX)
real S_buf [0:MAX_TILE-1][0:MAX_TILE-1]; // dequantized GEMM1 output

// ── Per-tile p_scale (computed on-chip after softmax) ─────────────────────
real p_scale;

// ── Systolic Array interface ──────────────────────────────────────────────
byte signed sa_a_reg [0:SA_SIZE-1][0:MAX_D-1];
byte signed sa_b_reg [0:SA_SIZE-1][0:MAX_D-1];
logic       sa_start;
logic [7:0] sa_depth;
logic [7:0] sa_active_rows;
logic [7:0] sa_active_cols;
logic       sa_done;
int         sa_out [0:SA_SIZE-1][0:SA_SIZE-1];

// ── INT8 quantize: fp64 → clamp(round(v/sc), -127, 127) ──────────────────
function automatic byte signed fp_to_int8(real v, real sc);
    int ri;
    if (sc == 0.0) return 8'sd0;
    ri = $rtoi((v / sc) + (v >= 0.0 ? 0.5 : -0.5));
    if (ri >  127) return  8'sd127;
    if (ri < -127) return -8'sd127;
    return byte'(ri);
endfunction

// ── Engine FSM states ─────────────────────────────────────────────────────
typedef enum logic [4:0] {
    FA_IDLE,
    FA_DMA_Q_AR,  FA_DMA_Q_R,
    FA_INIT_O_LM,
    FA_DMA_K_AR,  FA_DMA_K_R,   // first j-tile only
    FA_DMA_V_AR,  FA_DMA_V_R,   // first j-tile only
    FA_SA_G1_LD,  FA_SA_G1_ST,  FA_SA_G1_WT,
    FA_SA_SFX,
    FA_SA_G2_LD,  FA_SA_G2_ST,  FA_SA_G2_WT,  FA_SA_G2_NX,
    FA_NEXT_J,
    FA_FINALIZE,
    FA_DMA_O_AW,  FA_DMA_O_W,   FA_DMA_O_B,
    FA_NEXT_I,
    FA_DONE
} fa_state_t;

fa_state_t state;

// ── Prefetch sub-FSM states (K/V ping-pong) ───────────────────────────────
typedef enum logic [2:0] {
    PF_IDLE, PF_K_AR, PF_K_R, PF_V_AR, PF_V_R, PF_DONE
} pf_state_t;

pf_state_t pf_state;

logic [15:0] i_row;
logic [15:0] j_row;
logic [7:0]  cur_row;
logic [7:0]  word_cnt;
logic [7:0]  sfx_row;
logic [7:0]  gemm2_chunk;
logic        pp_sel;       // 0=A active, 1=B active
logic [15:0] pf_next_j;   // j_row being prefetched
logic [7:0]  pf_row;      // row counter within prefetch tile
logic [7:0]  pf_wc;       // word counter within prefetch row

// ── Combinational DMA control ─────────────────────────────────────────────
// RREADY covers both main DMA states and prefetch sub-FSM read states
assign RREADY_M = (state    == FA_DMA_Q_R) || (state    == FA_DMA_K_R) ||
                  (state    == FA_DMA_V_R) ||
                  (pf_state == PF_K_R)     || (pf_state == PF_V_R);
assign BREADY_M = (state == FA_DMA_O_B);
// O write: fp32, d words per row (AWLEN = d-1, so last word when word_cnt == d-1)
assign WLAST_M  = (state == FA_DMA_O_W) && (word_cnt == d_w[7:0] - 8'd1);

// ── Systolic Array instantiation ──────────────────────────────────────────
systolic_array #(
    .SA_SIZE (SA_SIZE),
    .MAX_D   (MAX_D)
) u_sa (
    .ACLK        (ACLK),
    .ARESETn     (ARESETn),
    .start       (sa_start),
    .depth       (sa_depth),
    .active_rows (sa_active_rows),
    .active_cols (sa_active_cols),
    .done        (sa_done),
    .a_mat       (sa_a_reg),
    .b_mat       (sa_b_reg),
    .out_mat     (sa_out)
);

// ── AXI4 Slave FSM ────────────────────────────────────────────────────────
typedef enum logic [2:0] {
    SLV_IDLE, SLV_WDATA, SLV_WRESP, SLV_RDATA
} slv_state_t;

slv_state_t slv_state;
logic [31:0] slv_addr_reg;
logic [3:0]  slv_bid_reg, slv_rid_reg;
logic        slv_werr;

always_ff @(posedge ACLK or negedge ARESETn) begin
    if (!ARESETn) begin
        slv_state    <= SLV_IDLE;
        slv_addr_reg <= '0;
        slv_bid_reg  <= '0;
        slv_rid_reg  <= '0;
        slv_werr     <= '0;
        BID_S        <= '0;
        RID_S        <= '0;
        reg_control  <= '0;
        reg_shape    <= '0;
        reg_tile     <= '0;
        reg_q_addr   <= '0;
        reg_k_addr   <= '0;
        reg_v_addr   <= '0;
        reg_o_addr   <= '0;
        reg_q_scale  <= 1.0;
        reg_k_scale  <= 1.0;
        reg_v_scale  <= 1.0;
    end else begin
        if (state != FA_IDLE) reg_control[0] <= 1'b0;

        case (slv_state)
            SLV_IDLE: begin
                if (AWVALID_S) begin
                    slv_state    <= SLV_WDATA;
                    slv_bid_reg  <= AWID_S;
                    slv_addr_reg <= AWADDR_S;
                end else if (ARVALID_S) begin
                    slv_state    <= SLV_RDATA;
                    slv_rid_reg  <= ARID_S;
                    slv_addr_reg <= ARADDR_S;
                end
            end

            SLV_WDATA: begin
                if (WVALID_S) begin
                    case (slv_addr_reg)
                        REG_CONTROL: reg_control  <= WDATA_S;
                        REG_SHAPE:   reg_shape    <= WDATA_S;
                        REG_TILE:    reg_tile      <= WDATA_S;
                        REG_Q_ADDR:  reg_q_addr   <= WDATA_S;
                        REG_K_ADDR:  reg_k_addr   <= WDATA_S;
                        REG_V_ADDR:  reg_v_addr   <= WDATA_S;
                        REG_O_ADDR:  reg_o_addr   <= WDATA_S;
                        // Scale registers: MMIO receives IEEE fp32 bits, convert to fp64
                        REG_Q_SCALE: reg_q_scale  <= dpi_fp32_bits_to_real(WDATA_S);
                        REG_K_SCALE: reg_k_scale  <= dpi_fp32_bits_to_real(WDATA_S);
                        REG_V_SCALE: reg_v_scale  <= dpi_fp32_bits_to_real(WDATA_S);
                        default:     slv_werr     <= 1'b1;
                    endcase
                    if (WLAST_S) slv_state <= SLV_WRESP;
                end
            end

            SLV_WRESP: begin
                if (BREADY_S) begin
                    slv_state <= SLV_IDLE;
                    slv_werr  <= 1'b0;
                    BID_S     <= slv_bid_reg;
                end
            end

            SLV_RDATA: begin
                if (RREADY_S) begin
                    slv_state <= SLV_IDLE;
                    RID_S     <= slv_rid_reg;
                end
            end

            default: slv_state <= SLV_IDLE;
        endcase
    end
end

always_comb begin
    AWREADY_S = (slv_state == SLV_IDLE) && AWVALID_S;
    WREADY_S  = (slv_state == SLV_WDATA);
    BVALID_S  = (slv_state == SLV_WRESP);
    BRESP_S   = slv_werr ? AXI_RESP_SLVERR : AXI_RESP_OKAY;
    ARREADY_S = (slv_state == SLV_IDLE) && !AWVALID_S && ARVALID_S;
    RVALID_S  = (slv_state == SLV_RDATA);
    RLAST_S   = (slv_state == SLV_RDATA);
    RRESP_S   = AXI_RESP_OKAY;

    case (slv_addr_reg)
        REG_CONTROL: RDATA_S = reg_control;
        REG_SHAPE:   RDATA_S = reg_shape;
        REG_TILE:    RDATA_S = reg_tile;
        REG_Q_ADDR:  RDATA_S = reg_q_addr;
        REG_K_ADDR:  RDATA_S = reg_k_addr;
        REG_V_ADDR:  RDATA_S = reg_v_addr;
        REG_O_ADDR:  RDATA_S = reg_o_addr;
        REG_STATUS:  RDATA_S = {30'b0, done_r, busy_r};
        REG_Q_SCALE: RDATA_S = dpi_real_to_fp32_bits(reg_q_scale);
        REG_K_SCALE: RDATA_S = dpi_real_to_fp32_bits(reg_k_scale);
        REG_V_SCALE: RDATA_S = dpi_real_to_fp32_bits(reg_v_scale);
        default:     RDATA_S = 32'hDEAD_BEEF;
    endcase
end

// ── Engine FSM + Prefetch Sub-FSM ─────────────────────────────────────────
//
// The always_ff block contains TWO state machines:
//   1. Main FSM (case state):     compute + DMA for Q/K(first)/V(first)/O
//   2. Prefetch FSM (case pf_state): DMA for K/V j+1 during SA compute
//
// The prefetch block appears AFTER the main case, so its NBA assignments
// to ARVALID_M / ARADDR_M / ARLEN_M win over the default (=0) set at the
// top of the block.  Main FSM never drives AR channel during SA states.
//
always_ff @(posedge ACLK or negedge ARESETn) begin
    if (!ARESETn) begin
        state          <= FA_IDLE;
        pf_state       <= PF_IDLE;
        i_row          <= '0;
        j_row          <= '0;
        cur_row        <= '0;
        word_cnt       <= '0;
        sfx_row        <= '0;
        gemm2_chunk    <= '0;
        pp_sel         <= '0;
        pf_next_j      <= '0;
        pf_row         <= '0;
        pf_wc          <= '0;
        busy_r         <= '0;
        done_r         <= '0;
        FA_interrupt   <= '0;
        sa_start       <= '0;
        sa_depth       <= '0;
        sa_active_rows <= 8'd14;
        sa_active_cols <= 8'd14;
        p_scale        <= 1.0;
        ARVALID_M  <= '0; ARADDR_M <= '0; ARLEN_M  <= '0;
        ARSIZE_M   <= 3'b010; ARBURST_M <= 2'b01; ARID_M <= '0;
        AWVALID_M  <= '0; AWADDR_M <= '0; AWLEN_M  <= '0;
        AWSIZE_M   <= 3'b010; AWBURST_M <= 2'b01; AWID_M <= '0;
        WVALID_M   <= '0; WDATA_M  <= '0; WSTRB_M  <= '0;
    end else begin
        if (reg_control[1]) begin
            FA_interrupt <= 1'b0;
            done_r       <= 1'b0;
        end

        // ── Defaults (prefetch block below may override AR signals) ─────
        ARVALID_M <= 1'b0;
        AWVALID_M <= 1'b0;
        WVALID_M  <= 1'b0;
        sa_start  <= 1'b0;

        // ════════════════════════════════════════════════════════════════
        // ── Main FSM ────────────────────────────────────────────────────
        // ════════════════════════════════════════════════════════════════
        case (state)

            FA_IDLE: begin
                if (reg_control[0]) begin
                    busy_r       <= 1'b1;
                    done_r       <= 1'b0;
                    FA_interrupt <= 1'b0;
                    i_row        <= '0;
                    cur_row      <= '0;
                    state        <= FA_DMA_Q_AR;
                end
            end

            // ── Read Q i-tile: INT8-packed, d/4 words per row ────────────
            FA_DMA_Q_AR: begin
                // Row byte address: base + (i_row + cur_row) × d
                ARADDR_M  <= reg_q_addr +
                             32'(int'(i_row) + int'(cur_row)) * 32'(int'(d_w));
                ARLEN_M   <= (d_w[7:0] >> 2) - 8'd1;   // d/4 - 1
                ARSIZE_M  <= 3'b010;
                ARBURST_M <= 2'b01;
                ARVALID_M <= 1'b1;
                word_cnt  <= '0;
                if (ARREADY_M) begin
                    ARVALID_M <= 1'b0;
                    state     <= FA_DMA_Q_R;
                end
            end

            FA_DMA_Q_R: begin
                if (RVALID_M) begin
                    // Unpack 4 INT8 from one 32-bit word (little-endian)
                    Q_buf[cur_row][word_cnt*4+0] <= RDATA_M[7:0];
                    Q_buf[cur_row][word_cnt*4+1] <= RDATA_M[15:8];
                    Q_buf[cur_row][word_cnt*4+2] <= RDATA_M[23:16];
                    Q_buf[cur_row][word_cnt*4+3] <= RDATA_M[31:24];
                    word_cnt <= word_cnt + 8'd1;
                    if (RLAST_M) begin
                        if (int'(cur_row) < int'(Br_w) - 1) begin
                            cur_row <= cur_row + 8'd1;
                            state   <= FA_DMA_Q_AR;
                        end else begin
                            cur_row  <= '0;
                            j_row    <= '0;
                            pp_sel   <= 1'b0;       // A is active for first j-tile
                            pf_state <= PF_IDLE;    // reset prefetch
                            state    <= FA_INIT_O_LM;
                        end
                    end
                end
            end

            // ── Initialize O/l/m for this i-tile ─────────────────────────
            FA_INIT_O_LM: begin
                begin : init_blk
                    int r, k;
                    for (r = 0; r < MAX_TILE; r++) begin
                        l_buf[r] <= 0.0;
                        m_buf[r] <= -1.0e38;
                        for (k = 0; k < MAX_D; k++)
                            O_buf[r][k] <= 0.0;
                    end
                end
                cur_row <= '0;
                state   <= FA_DMA_K_AR;  // first j-tile: explicit DMA
            end

            // ── Read K j-tile (first j-tile only): into active buffer ─────
            FA_DMA_K_AR: begin
                ARADDR_M  <= reg_k_addr +
                             32'(int'(j_row) + int'(cur_row)) * 32'(int'(d_w));
                ARLEN_M   <= (d_w[7:0] >> 2) - 8'd1;
                ARSIZE_M  <= 3'b010;
                ARBURST_M <= 2'b01;
                ARVALID_M <= 1'b1;
                word_cnt  <= '0;
                if (ARREADY_M) begin
                    ARVALID_M <= 1'b0;
                    state     <= FA_DMA_K_R;
                end
            end

            FA_DMA_K_R: begin
                if (RVALID_M) begin
                    // Load into active K buffer (pp_sel=0→A, pp_sel=1→B)
                    if (!pp_sel) begin
                        K_buf_A[cur_row][word_cnt*4+0] <= RDATA_M[7:0];
                        K_buf_A[cur_row][word_cnt*4+1] <= RDATA_M[15:8];
                        K_buf_A[cur_row][word_cnt*4+2] <= RDATA_M[23:16];
                        K_buf_A[cur_row][word_cnt*4+3] <= RDATA_M[31:24];
                    end else begin
                        K_buf_B[cur_row][word_cnt*4+0] <= RDATA_M[7:0];
                        K_buf_B[cur_row][word_cnt*4+1] <= RDATA_M[15:8];
                        K_buf_B[cur_row][word_cnt*4+2] <= RDATA_M[23:16];
                        K_buf_B[cur_row][word_cnt*4+3] <= RDATA_M[31:24];
                    end
                    word_cnt <= word_cnt + 8'd1;
                    if (RLAST_M) begin
                        if (int'(cur_row) < int'(Br_w) - 1) begin
                            cur_row <= cur_row + 8'd1;
                            state   <= FA_DMA_K_AR;
                        end else begin
                            cur_row <= '0;
                            state   <= FA_DMA_V_AR;
                        end
                    end
                end
            end

            // ── Read V j-tile (first j-tile only): into active buffer ─────
            FA_DMA_V_AR: begin
                ARADDR_M  <= reg_v_addr +
                             32'(int'(j_row) + int'(cur_row)) * 32'(int'(d_w));
                ARLEN_M   <= (d_w[7:0] >> 2) - 8'd1;
                ARSIZE_M  <= 3'b010;
                ARBURST_M <= 2'b01;
                ARVALID_M <= 1'b1;
                word_cnt  <= '0;
                if (ARREADY_M) begin
                    ARVALID_M <= 1'b0;
                    state     <= FA_DMA_V_R;
                end
            end

            FA_DMA_V_R: begin
                if (RVALID_M) begin
                    if (!pp_sel) begin
                        V_buf_A[cur_row][word_cnt*4+0] <= RDATA_M[7:0];
                        V_buf_A[cur_row][word_cnt*4+1] <= RDATA_M[15:8];
                        V_buf_A[cur_row][word_cnt*4+2] <= RDATA_M[23:16];
                        V_buf_A[cur_row][word_cnt*4+3] <= RDATA_M[31:24];
                    end else begin
                        V_buf_B[cur_row][word_cnt*4+0] <= RDATA_M[7:0];
                        V_buf_B[cur_row][word_cnt*4+1] <= RDATA_M[15:8];
                        V_buf_B[cur_row][word_cnt*4+2] <= RDATA_M[23:16];
                        V_buf_B[cur_row][word_cnt*4+3] <= RDATA_M[31:24];
                    end
                    word_cnt <= word_cnt + 8'd1;
                    if (RLAST_M) begin
                        if (int'(cur_row) < int'(Br_w) - 1) begin
                            cur_row <= cur_row + 8'd1;
                            state   <= FA_DMA_V_AR;
                        end else begin
                            cur_row <= '0;
                            state   <= FA_SA_G1_LD;
                        end
                    end
                end
            end

            // ════════════════════════════════════════════════════════════════
            // ── GEMM1: S_int32 = Q_int8 × K_int8^T ─────────────────────────
            // Q_buf and active K_buf are already INT8; copy directly to SA regs.
            // No per-tile scale computation — scales come from MMIO (global).
            // ════════════════════════════════════════════════════════════════
            FA_SA_G1_LD: begin
                begin : g1_load
                    int rr, kk;
                    for (rr = 0; rr < SA_SIZE; rr++)
                        for (kk = 0; kk < MAX_D; kk++) begin
                            sa_a_reg[rr][kk] <= Q_buf[rr][kk];
                            // Active K buffer selected by pp_sel
                            sa_b_reg[rr][kk] <= pp_sel ? K_buf_B[rr][kk]
                                                        : K_buf_A[rr][kk];
                        end
                end
                sa_depth       <= d_w[7:0];
                sa_active_rows <= Br_w[7:0];
                sa_active_cols <= Br_w[7:0];
                state          <= FA_SA_G1_ST;
            end

            FA_SA_G1_ST: begin
                sa_start <= 1'b1;
                state    <= FA_SA_G1_WT;
            end

            // Wait for GEMM1; dequantize S = INT32 × q_scale × k_scale / √d
            // Prefetch sub-FSM runs concurrently (see below).
            FA_SA_G1_WT: begin
                if (sa_done) begin
                    begin : g1_dequant
                        int  rr, cc;
                        real attn_scale, dq_factor;
                        attn_scale = 1.0 / dpi_sqrtf(real'(d_w));
                        dq_factor  = reg_q_scale * reg_k_scale * attn_scale;
                        for (rr = 0; rr < SA_SIZE; rr++)
                            for (cc = 0; cc < SA_SIZE; cc++)
                                if (rr < int'(Br_w) && cc < int'(Br_w))
                                    S_buf[rr][cc] = real'(sa_out[rr][cc]) * dq_factor;
                    end
                    sfx_row <= '0;
                    state   <= FA_SA_SFX;
                end
            end

            // ════════════════════════════════════════════════════════════════
            // ── Online Softmax: one row per cycle ────────────────────────────
            // P_buf written with blocking =; p_scale computed on last row.
            // Prefetch continues concurrently.
            // ════════════════════════════════════════════════════════════════
            FA_SA_SFX: begin
                begin : sfx_blk
                    int cc, kk;
                    real row_max, m_new_v, corr_v, sum_p_v;

                    row_max = S_buf[sfx_row][0];
                    for (cc = 1; cc < MAX_TILE; cc++)
                        if (cc < int'(Br_w) && S_buf[sfx_row][cc] > row_max)
                            row_max = S_buf[sfx_row][cc];

                    m_new_v = (m_buf[sfx_row] > row_max) ? m_buf[sfx_row] : row_max;
                    corr_v  = dpi_expf(m_buf[sfx_row] - m_new_v);

                    sum_p_v = 0.0;
                    for (cc = 0; cc < MAX_TILE; cc++) begin
                        if (cc < int'(Br_w)) begin
                            P_buf[sfx_row][cc] = dpi_expf(S_buf[sfx_row][cc] - m_new_v);
                            sum_p_v = sum_p_v + P_buf[sfx_row][cc];
                        end
                    end

                    for (kk = 0; kk < MAX_D; kk++)
                        if (kk < int'(d_w))
                            O_buf[sfx_row][kk] <= O_buf[sfx_row][kk] * corr_v;

                    l_buf[sfx_row] <= l_buf[sfx_row] * corr_v + sum_p_v;
                    m_buf[sfx_row] <= m_new_v;
                end

                if (sfx_row == Br_w[7:0] - 8'd1) begin
                    // Last row: P_buf fully populated (blocking =).
                    // Compute p_scale = max_abs(P_tile) / 127
                    begin : p_scale_blk
                        real mp, v;
                        int  rr, cc;
                        mp = 0.0;
                        for (rr = 0; rr < int'(Br_w); rr++)
                            for (cc = 0; cc < int'(Br_w); cc++) begin
                                v = P_buf[rr][cc];
                                if (v < 0.0) v = -v;
                                if (v > mp) mp = v;
                            end
                        p_scale <= (mp > 0.0) ? mp / 127.0 : 1.0;
                    end
                    sfx_row     <= '0;
                    gemm2_chunk <= '0;
                    state       <= FA_SA_G2_LD;
                end else begin
                    sfx_row <= sfx_row + 8'd1;
                end
            end

            // ════════════════════════════════════════════════════════════════
            // ── GEMM2: ΔO_int32 = P_int8 × V_int8^T (chunked by SA_SIZE) ───
            // V_buf is Br rows × d cols.  Split into ceil(d/SA_SIZE) chunks.
            // sa_b_reg[j][k] = V_buf[k][col_base + j]  (transpose of V chunk)
            // Prefetch continues concurrently during G2_WT.
            // ════════════════════════════════════════════════════════════════
            FA_SA_G2_LD: begin
                begin : g2_load
                    int  rr, jj, kk;
                    int  col_base, active_c;
                    col_base = int'(gemm2_chunk) * SA_SIZE;
                    active_c = int'(d_w) - col_base;
                    if (active_c > SA_SIZE) active_c = SA_SIZE;

                    // Quantize P_tile → sa_a_reg using p_scale (from SFX)
                    for (rr = 0; rr < SA_SIZE; rr++)
                        for (kk = 0; kk < SA_SIZE; kk++)
                            sa_a_reg[rr][kk] <= fp_to_int8(P_buf[rr][kk], p_scale);

                    // Transpose active V chunk → sa_b_reg (already INT8, no quantize)
                    for (jj = 0; jj < SA_SIZE; jj++)
                        for (kk = 0; kk < SA_SIZE; kk++) begin
                            if (jj < active_c)
                                sa_b_reg[jj][kk] <= pp_sel
                                    ? V_buf_B[kk][col_base + jj]
                                    : V_buf_A[kk][col_base + jj];
                            else
                                sa_b_reg[jj][kk] <= 8'sd0;
                        end

                    sa_depth       <= Br_w[7:0];
                    sa_active_rows <= Br_w[7:0];
                    sa_active_cols <= 8'(active_c);
                end
                state <= FA_SA_G2_ST;
            end

            FA_SA_G2_ST: begin
                sa_start <= 1'b1;
                state    <= FA_SA_G2_WT;
            end

            FA_SA_G2_WT: begin
                if (sa_done) state <= FA_SA_G2_NX;
            end

            // Dequantize chunk result and accumulate into O_buf
            FA_SA_G2_NX: begin
                begin : g2_accum
                    int  rr, jj;
                    int  col_base, active_c;
                    real dq;
                    col_base = int'(gemm2_chunk) * SA_SIZE;
                    active_c = int'(d_w) - col_base;
                    if (active_c > SA_SIZE) active_c = SA_SIZE;
                    dq = p_scale * reg_v_scale;

                    for (rr = 0; rr < SA_SIZE; rr++)
                        for (jj = 0; jj < SA_SIZE; jj++)
                            if (rr < int'(Br_w) && jj < active_c)
                                O_buf[rr][col_base + jj] <=
                                    O_buf[rr][col_base + jj] +
                                    real'(sa_out[rr][jj]) * dq;
                end

                if ((int'(gemm2_chunk) + 1) * SA_SIZE < int'(d_w)) begin
                    gemm2_chunk <= gemm2_chunk + 8'd1;
                    state       <= FA_SA_G2_LD;
                end else begin
                    gemm2_chunk <= '0;
                    state       <= FA_NEXT_J;
                end
            end

            // ── Advance j-counter ─────────────────────────────────────────
            // Wait until prefetch sub-FSM has loaded next K/V (pf_state==PF_DONE).
            // Then swap ping-pong and skip DMA_K/V for the next j-tile.
            FA_NEXT_J: begin
                if (int'(j_row) + int'(Br_w) < int'(N_w)) begin
                    if (pf_state == PF_DONE) begin
                        j_row    <= j_row + {8'b0, Br_w[7:0]};
                        pp_sel   <= !pp_sel;   // inactive buffer (just prefetched) → active
                        pf_state <= PF_IDLE;   // reset for next round
                        state    <= FA_SA_G1_LD;  // data ready; skip DMA_K/V
                    end
                    // else: wait here — prefetch block below continues running
                end else begin
                    state <= FA_FINALIZE;
                end
            end

            // ── Normalize O /= l ──────────────────────────────────────────
            FA_FINALIZE: begin
                begin : final_blk
                    int r, k;
                    for (r = 0; r < MAX_TILE; r++)
                        if (r < int'(Br_w))
                            for (k = 0; k < MAX_D; k++)
                                if (k < int'(d_w))
                                    O_buf[r][k] <= O_buf[r][k] / l_buf[r];
                end
                cur_row <= '0;
                state   <= FA_DMA_O_AW;
            end

            // ── Write O i-tile as fp32 ────────────────────────────────────
            FA_DMA_O_AW: begin
                // O row stride = d×4 bytes (fp32)
                AWADDR_M  <= reg_o_addr +
                             32'(int'(i_row) + int'(cur_row)) * 32'(int'(d_w)) * 4;
                AWLEN_M   <= d_w[7:0] - 8'd1;
                AWSIZE_M  <= 3'b010;
                AWBURST_M <= 2'b01;
                AWVALID_M <= 1'b1;
                word_cnt  <= '0;
                if (AWREADY_M) begin
                    AWVALID_M <= 1'b0;
                    state     <= FA_DMA_O_W;
                end
            end

            FA_DMA_O_W: begin
                WVALID_M <= 1'b1;
                WDATA_M  <= dpi_real_to_fp32_bits(O_buf[cur_row][word_cnt]);
                WSTRB_M  <= 4'hF;
                if (WREADY_M) begin
                    if (int'(word_cnt) == int'(d_w) - 1) begin
                        WVALID_M <= 1'b0;
                        state    <= FA_DMA_O_B;
                    end else begin
                        word_cnt <= word_cnt + 8'd1;
                    end
                end
            end

            FA_DMA_O_B: begin
                if (BVALID_M) begin
                    if (int'(cur_row) < int'(Br_w) - 1) begin
                        cur_row <= cur_row + 8'd1;
                        state   <= FA_DMA_O_AW;
                    end else begin
                        state <= FA_NEXT_I;
                    end
                end
            end

            // ── Advance i-counter ─────────────────────────────────────────
            FA_NEXT_I: begin
                if (int'(i_row) + int'(Br_w) < int'(N_w)) begin
                    i_row    <= i_row + {8'b0, Br_w[7:0]};
                    cur_row  <= '0;
                    pp_sel   <= 1'b0;
                    pf_state <= PF_IDLE;
                    state    <= FA_DMA_Q_AR;
                end else begin
                    state <= FA_DONE;
                end
            end

            FA_DONE: begin
                FA_interrupt <= 1'b1;
                busy_r       <= 1'b0;
                done_r       <= 1'b1;
                if (reg_control[1]) begin
                    FA_interrupt <= 1'b0;
                    done_r       <= 1'b0;
                    state        <= FA_IDLE;
                end
            end

            default: state <= FA_IDLE;
        endcase

        // ════════════════════════════════════════════════════════════════════
        // ── Prefetch Sub-FSM ─────────────────────────────────────────────────
        //
        // Active only when main FSM is in an SA compute/wait state where the
        // AXI master AR channel is idle.  Assigns come AFTER the main case, so
        // they override the default ARVALID_M=0 set at the top of this block.
        //
        // Loads K/V for the next j-tile (pf_next_j = j_row + Br) into the
        // INACTIVE ping-pong buffer (opposite of pp_sel).
        // ════════════════════════════════════════════════════════════════════
        if (state == FA_SA_G1_WT  || state == FA_SA_SFX    ||
            state == FA_SA_G2_LD  || state == FA_SA_G2_ST  ||
            state == FA_SA_G2_WT  || state == FA_SA_G2_NX  ||
            state == FA_NEXT_J) begin

            case (pf_state)

                PF_IDLE: begin
                    if (int'(j_row) + int'(Br_w) < int'(N_w)) begin
                        pf_next_j <= j_row + {8'b0, Br_w[7:0]};
                        pf_row    <= '0;
                        pf_wc     <= '0;
                        pf_state  <= PF_K_AR;
                    end else begin
                        pf_state  <= PF_DONE;  // no next tile; nothing to prefetch
                    end
                end

                PF_K_AR: begin
                    // Issue AR for next K row into INACTIVE buffer
                    ARADDR_M  <= reg_k_addr +
                                 32'(int'(pf_next_j) + int'(pf_row)) * 32'(int'(d_w));
                    ARLEN_M   <= (d_w[7:0] >> 2) - 8'd1;
                    ARSIZE_M  <= 3'b010;
                    ARBURST_M <= 2'b01;
                    ARVALID_M <= 1'b1;
                    pf_wc     <= '0;
                    if (ARREADY_M) begin
                        ARVALID_M <= 1'b0;
                        pf_state  <= PF_K_R;
                    end
                end

                PF_K_R: begin
                    if (RVALID_M) begin
                        // pp_sel=0 → A active → prefetch into B; pp_sel=1 → vice versa
                        if (!pp_sel) begin
                            K_buf_B[pf_row][pf_wc*4+0] <= RDATA_M[7:0];
                            K_buf_B[pf_row][pf_wc*4+1] <= RDATA_M[15:8];
                            K_buf_B[pf_row][pf_wc*4+2] <= RDATA_M[23:16];
                            K_buf_B[pf_row][pf_wc*4+3] <= RDATA_M[31:24];
                        end else begin
                            K_buf_A[pf_row][pf_wc*4+0] <= RDATA_M[7:0];
                            K_buf_A[pf_row][pf_wc*4+1] <= RDATA_M[15:8];
                            K_buf_A[pf_row][pf_wc*4+2] <= RDATA_M[23:16];
                            K_buf_A[pf_row][pf_wc*4+3] <= RDATA_M[31:24];
                        end
                        pf_wc <= pf_wc + 8'd1;
                        if (RLAST_M) begin
                            if (int'(pf_row) < int'(Br_w) - 1) begin
                                pf_row   <= pf_row + 8'd1;
                                pf_state <= PF_K_AR;
                            end else begin
                                pf_row   <= '0;
                                pf_state <= PF_V_AR;
                            end
                        end
                    end
                end

                PF_V_AR: begin
                    ARADDR_M  <= reg_v_addr +
                                 32'(int'(pf_next_j) + int'(pf_row)) * 32'(int'(d_w));
                    ARLEN_M   <= (d_w[7:0] >> 2) - 8'd1;
                    ARSIZE_M  <= 3'b010;
                    ARBURST_M <= 2'b01;
                    ARVALID_M <= 1'b1;
                    pf_wc     <= '0;
                    if (ARREADY_M) begin
                        ARVALID_M <= 1'b0;
                        pf_state  <= PF_V_R;
                    end
                end

                PF_V_R: begin
                    if (RVALID_M) begin
                        if (!pp_sel) begin
                            V_buf_B[pf_row][pf_wc*4+0] <= RDATA_M[7:0];
                            V_buf_B[pf_row][pf_wc*4+1] <= RDATA_M[15:8];
                            V_buf_B[pf_row][pf_wc*4+2] <= RDATA_M[23:16];
                            V_buf_B[pf_row][pf_wc*4+3] <= RDATA_M[31:24];
                        end else begin
                            V_buf_A[pf_row][pf_wc*4+0] <= RDATA_M[7:0];
                            V_buf_A[pf_row][pf_wc*4+1] <= RDATA_M[15:8];
                            V_buf_A[pf_row][pf_wc*4+2] <= RDATA_M[23:16];
                            V_buf_A[pf_row][pf_wc*4+3] <= RDATA_M[31:24];
                        end
                        pf_wc <= pf_wc + 8'd1;
                        if (RLAST_M) begin
                            if (int'(pf_row) < int'(Br_w) - 1) begin
                                pf_row   <= pf_row + 8'd1;
                                pf_state <= PF_V_AR;
                            end else begin
                                pf_state <= PF_DONE;
                            end
                        end
                    end
                end

                PF_DONE: begin end  // wait; FA_NEXT_J will reset to PF_IDLE

                default: ;
            endcase
        end

    end
end

endmodule
