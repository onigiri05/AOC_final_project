`include "ASIC.svh"

module PPU #(
    parameter int TOKEN_NUM       = 197,
    parameter int CHANNEL_NUM     = 384,
    parameter int TOKEN_TILE      = 16,
    parameter int CHANNEL_TILE    = 16,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int CHANNEL_TILE_W  = 5,
    parameter logic [7:0] ZERO_POINT = 8'd128
)(
    input  logic clk,
    input  logic rst,

    // 00: Attention output, 01: FFN FC1, 10: FFN FC2
    input  logic [1:0] ppu_mode_i,
    input  logic [5:0] scaling_factor_i,

    // -------------------------
    // Tile input handshake from Systolic Array / SRAM
    // -------------------------
    input  logic tile_valid_i,
    output logic tile_ready_o,

    // 32-bit Psum input tile (16x16x32 = 8192 bits)
    input  logic [TOKEN_TILE*CHANNEL_TILE*`DATA_BITS-1:0] psum_tile_i,

    // 8-bit Residual tile (16x16x8 = 2048 bits)
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_tile_i,

    // Indices & masks
    input  logic [TOKEN_W-1:0]        base_token_idx_i,
    input  logic [CHANNEL_TILE_W-1:0] channel_tile_idx_i,
    input  logic [TOKEN_TILE-1:0]     token_valid_mask_i,

    // -------------------------
    // Tile output handshake to Activation Buffer / GLB
    // -------------------------
    output logic data_tile_valid_o,
    input  logic data_tile_ready_i,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o,

    // -------------------------
    // Statistic output handshake to Token Stat SRAM
    // -------------------------
    output logic stat_valid_o,
    input  logic stat_ready_i,
    output logic [TOKEN_W-1:0] stat_token_idx_o,
    output logic [SUM_W-1:0]   sum_sq_o
);

    localparam int TILE_ELEMS = TOKEN_TILE * CHANNEL_TILE;

    // ==========================================
    // Stage 1 Pipeline Registers
    // ==========================================
    logic stg1_valid;
    logic stg1_ready;

    logic [1:0]                                stg1_ppu_mode;
    logic [5:0]                                stg1_scaling_factor;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] stg1_residual_tile;
    logic [TOKEN_W-1:0]                        stg1_base_token_idx;
    logic [CHANNEL_TILE_W-1:0]                 stg1_channel_tile_idx;
    logic [TOKEN_TILE-1:0]                     stg1_token_valid_mask;

    // Bypass register for non-GELU paths
    logic signed [`DATA_BITS-1:0] stg1_psum_bypass [0:TILE_ELEMS-1];

    // Handshake logic
    logic tail_ready_o;
    assign stg1_ready   = !stg1_valid || tail_ready_o;
    assign tile_ready_o = stg1_ready;

    always_ff @(posedge clk) begin
        if (rst) begin
            stg1_valid <= 1'b0;
        end else if (stg1_ready) begin
            stg1_valid <= tile_valid_i;
        end
    end

    integer i;
    always_ff @(posedge clk) begin
        if (stg1_ready && tile_valid_i) begin
            stg1_ppu_mode         <= ppu_mode_i;
            stg1_scaling_factor   <= scaling_factor_i;
            stg1_residual_tile    <= residual_tile_i;
            stg1_base_token_idx   <= base_token_idx_i;
            stg1_channel_tile_idx <= channel_tile_idx_i;
            stg1_token_valid_mask <= token_valid_mask_i;
            
            for (i = 0; i < TILE_ELEMS; i++) begin
                stg1_psum_bypass[i] <= psum_tile_i[i*`DATA_BITS +: `DATA_BITS];
            end
        end
    end

    // ==========================================
    // Datapath: GELU Array & Requantization
    // ==========================================
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] stg1_main_tile;

    genvar g;
    generate
        for (g = 0; g < TILE_ELEMS; g++) begin : gen_ppu_lanes
            logic signed [`DATA_BITS-1:0] lane_psum_in;
            logic signed [`DATA_BITS-1:0] lane_gelu_out;
            logic signed [`DATA_BITS-1:0] lane_post_gelu;
            logic [7:0]                   lane_requant_out;

            assign lane_psum_in = psum_tile_i[g*`DATA_BITS +: `DATA_BITS];

            // 1 Cycle latency matches stg1 registers
            GELU_Unit u_gelu (
                .clk     (clk),
                .rst     (rst),
                .en      (tile_valid_i && stg1_ready),
                .data_in (lane_psum_in),
                .data_out(lane_gelu_out)
            );

            // Mux: Use GELU out for FC1 mode, otherwise use bypass psum
            assign lane_post_gelu = (stg1_ppu_mode == 2'b01) ? lane_gelu_out : stg1_psum_bypass[g];

            // Pure Combinational mapping
            Requant_Unit u_requant (
                .data_in       (lane_post_gelu),
                .scaling_factor(stg1_scaling_factor),
                .data_out      (lane_requant_out)
            );

            assign stg1_main_tile[g*DATA_W +: DATA_W] = lane_requant_out;
        end
    endgenerate

    // ==========================================
    // Tail Stage: Residual Add & RMS Acc
    // ==========================================
    PPU_Residual_RMS_Tail #(
        .TOKEN_NUM       (TOKEN_NUM),
        .CHANNEL_NUM     (CHANNEL_NUM),
        .TOKEN_TILE      (TOKEN_TILE),
        .CHANNEL_TILE    (CHANNEL_TILE),
        .DATA_W          (DATA_W),
        .SUM_W           (SUM_W),
        .TOKEN_W         (TOKEN_W),
        .CHANNEL_TILE_W  (CHANNEL_TILE_W),
        .ZERO_POINT      (ZERO_POINT)
    ) u_tail (
        .clk                (clk),
        .rst                (rst),
        .ppu_mode_i         (stg1_ppu_mode),
        .tile_valid_i       (stg1_valid),
        .tile_ready_o       (tail_ready_o),
        .main_tile_i        (stg1_main_tile),
        .residual_tile_i    (stg1_residual_tile),
        .base_token_idx_i   (stg1_base_token_idx),
        .channel_tile_idx_i (stg1_channel_tile_idx),
        .token_valid_mask_i (stg1_token_valid_mask),
        .data_tile_valid_o  (data_tile_valid_o),
        .data_tile_ready_i  (data_tile_ready_i),
        .data_tile_o        (data_tile_o),
        .stat_valid_o       (stat_valid_o),
        .stat_ready_i       (stat_ready_i),
        .stat_token_idx_o   (stat_token_idx_o),
        .sum_sq_o           (sum_sq_o)
    );

endmodule