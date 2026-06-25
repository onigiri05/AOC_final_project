`timescale 1ns/1ps

// ============================================================
// Module: ViT_DDR_AxiLite_Wrapper
//
// ??wrapper ??FPGA full-design 撱箄降雿輻??憭惜嚗?
//   1. Python ?芷? AXI-Lite 銝?嗅隞方? DDR descriptor??
//   2. RTL ?芸楛??M_AXI 敺?PS DDR 霈 32-bit words??
//   3. 霈??????ViT_System_Core ?Ｘ? loader??
//   4. 閮?摰?敺?銋隞亦 RTL ??X_out 撖怠? DDR??
//
// 瘜冽?嚗??DDR master ????single-beat AXI4 read/write嚗靘?bring-up??
// 銋??交??賭?頞喉??臭誑?? LOAD/STORE FSM ?寞? burst??
//
// AXI-Lite register map:
//   0x000 CONTROL      bit0=start compute, bit1=clear done
//   0x004 PATCH_SHIFT
//   0x008 STATUS
//   0x00c DEBUG
//   0x010 PATCH_REQ
//   0x014 TRANS_REQ
//   0x018 INPUT_TGT
//   0x01c INPUT_COUNT
//   0x020 DDR_SRC_ADDR
//   0x024 DDR_DST_ADDR
//   0x028 DDR_LOCAL_BASE
//   0x02c DDR_WORD_COUNT
//   0x030 DDR_TARGET
//   0x034 DDR_STATUS
//   0x038 DDR_WORD_INDEX
//   0x03c DDR_CONTROL   bit0=start load, bit1=start store, bit8=clear DDR flags
//   0x040 PERF_CONTROL  bit0=clear counters
//   0x044..0x098        RTL performance counters
//   0x078 PAGE_REQ      bit31=valid, bit30=store, bits27:24=target, bits23:0=base
//   0x09c RUN_MODE      bit0=transformer-only start, skips patch embedding
//   0x100..0x10f        legacy software loader window
//   0x1000..            legacy X_out readback window
// ============================================================
module ViT_DDR_AxiLite_Wrapper #(
    parameter int AXI_ADDR_WIDTH  = 20,
    parameter int AXI_DATA_WIDTH  = 32,
    parameter int M_AXI_ADDR_WIDTH = 32,
    parameter int M_AXI_DATA_WIDTH = 32,
    parameter int IMG_H           = 224,
    parameter int IMG_W           = 224,
    parameter int IMG_C           = 3,
    parameter int PATCH_SIZE      = 16,
    parameter int EMBED_DIM       = 384,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int CHANNEL_TILE_W  = 6,
    parameter int ADDR_W          = 17,
    parameter int HEAD_NUM        = 6,
    parameter int HEAD_DIM        = 64,
    parameter int FFN_CHANNEL_NUM = 1536,
    parameter int SOFTMAX_COLS    = 208,
    parameter int LOAD_ADDR_W     = 20,
    parameter SOFTMAX_EXP_LUT_HEX = "exp_lut_10bit_Q1_15_range12.hex",

    parameter int PATCH_GRID_H    = IMG_H / PATCH_SIZE,
    parameter int PATCH_GRID_W    = IMG_W / PATCH_SIZE,
    parameter int PATCH_COUNT     = PATCH_GRID_H * PATCH_GRID_W,
    parameter int TOKEN_NUM       = PATCH_COUNT + 1,
    parameter int PATCH_ELEMS     = PATCH_SIZE * PATCH_SIZE * IMG_C,
    parameter int IMG_ADDR_W      = $clog2(IMG_H * IMG_W * IMG_C),
    parameter int PE_W_ADDR_W     = 17,
    parameter int PE_POS_ADDR_W   = $clog2(TOKEN_NUM * EMBED_DIM),
    parameter int PE_EMBED_ADDR_W = $clog2(EMBED_DIM)
)(
    input  logic s_axi_aclk,
    input  logic s_axi_aresetn,

    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input  logic s_axi_awvalid,
    output logic s_axi_awready,

    input  logic [AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input  logic [(AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input  logic s_axi_wvalid,
    output logic s_axi_wready,

    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input  logic s_axi_bready,

    input  logic [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input  logic s_axi_arvalid,
    output logic s_axi_arready,

    output logic [AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic s_axi_rvalid,
    input  logic s_axi_rready,

    output logic [M_AXI_ADDR_WIDTH-1:0] m_axi_awaddr,
    output logic [7:0]                  m_axi_awlen,
    output logic [2:0]                  m_axi_awsize,
    output logic [1:0]                  m_axi_awburst,
    output logic [3:0]                  m_axi_awcache,
    output logic [2:0]                  m_axi_awprot,
    output logic                        m_axi_awvalid,
    input  logic                        m_axi_awready,

    output logic [M_AXI_DATA_WIDTH-1:0] m_axi_wdata,
    output logic [(M_AXI_DATA_WIDTH/8)-1:0] m_axi_wstrb,
    output logic                        m_axi_wlast,
    output logic                        m_axi_wvalid,
    input  logic                        m_axi_wready,

    input  logic [1:0]                  m_axi_bresp,
    input  logic                        m_axi_bvalid,
    output logic                        m_axi_bready,

    output logic [M_AXI_ADDR_WIDTH-1:0] m_axi_araddr,
    output logic [7:0]                  m_axi_arlen,
    output logic [2:0]                  m_axi_arsize,
    output logic [1:0]                  m_axi_arburst,
    output logic [3:0]                  m_axi_arcache,
    output logic [2:0]                  m_axi_arprot,
    output logic                        m_axi_arvalid,
    input  logic                        m_axi_arready,

    input  logic [M_AXI_DATA_WIDTH-1:0] m_axi_rdata,
    input  logic [1:0]                  m_axi_rresp,
    input  logic                        m_axi_rlast,
    input  logic                        m_axi_rvalid,
    output logic                        m_axi_rready
);

    initial begin
        if (AXI_DATA_WIDTH != 32) begin
            $error("ViT_DDR_AxiLite_Wrapper currently expects 32-bit AXI-Lite data.");
        end
        if (M_AXI_DATA_WIDTH != 32) begin
            $error("ViT_DDR_AxiLite_Wrapper currently expects 32-bit AXI master data.");
        end
    end

    localparam logic [AXI_ADDR_WIDTH-1:0] REG_CONTROL      = 'h000;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PATCH_SHIFT  = 'h004;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STATUS       = 'h008;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DEBUG        = 'h00c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PATCH_REQ    = 'h010;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_TRANS_REQ    = 'h014;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_INPUT_TGT    = 'h018;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_INPUT_COUNT  = 'h01c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_SRC_ADDR = 'h020;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_DST_ADDR = 'h024;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_BASE     = 'h028;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_COUNT    = 'h02c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_TARGET   = 'h030;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_STATUS   = 'h034;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_INDEX    = 'h038;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_DDR_CONTROL  = 'h03c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_CONTROL = 'h040;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_TOTAL   = 'h044;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_BUSY    = 'h048;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_DDR_LD  = 'h04c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_DDR_ST  = 'h050;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_OVERLAP = 'h054;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_STALL   = 'h058;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_DDR_RDW = 'h05c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_DDR_WRW = 'h060;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_LDR_WR  = 'h064;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_OUT_WR  = 'h068;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_HOST_WR = 'h06c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_OUT_RD  = 'h070;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_ACTIVE  = 'h074;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PAGE_REQ     = 'h078;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_BRAM_RDW = 'h07c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_BRAM_WRW = 'h080;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_BRAM_CYC = 'h084;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_MAC_LO   = 'h088;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_MAC_HI   = 'h08c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_PP_WAIT  = 'h090;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_PP_LOAD  = 'h094;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_PERF_PP_OVLP  = 'h098;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_RUN_MODE      = 'h09c;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_CONTROL = 'h0a0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_STATUS  = 'h0a4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_TOTAL   = 'h0a8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_BUSY    = 'h0ac;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_DDR_LD  = 'h0b0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_DDR_ST  = 'h0b4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_DDR_RDW = 'h0b8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_DDR_WRW = 'h0bc;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_BRAM_RDW = 'h0c0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_BRAM_WRW = 'h0c4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_BRAM_CYC = 'h0c8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_MAC_LO   = 'h0cc;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_MAC_HI   = 'h0d0;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_PP_WAIT  = 'h0d4;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_PP_LOAD  = 'h0d8;
    localparam logic [AXI_ADDR_WIDTH-1:0] REG_STAGE_PP_OVLP  = 'h0dc;

    localparam logic [AXI_ADDR_WIDTH-1:0] INPUT_BASE       = 'h100;
    localparam logic [AXI_ADDR_WIDTH-1:0] INPUT_LIMIT      = 'h10f;
    localparam logic [AXI_ADDR_WIDTH-1:0] OUTPUT_BASE      = 'h1000;

    typedef enum logic [1:0] {
        WR_IDLE,
        WR_WAIT_INPUT,
        WR_RESP
    } wr_state_t;

    typedef enum logic [1:0] {
        RD_IDLE,
        RD_WAIT_OUTPUT,
        RD_CAPTURE_OUTPUT,
        RD_RESP
    } rd_state_t;

    typedef enum logic [3:0] {
        LOAD_IDLE,
        LOAD_CFG_BASE,
        LOAD_CFG_COUNT,
        LOAD_CFG_CTRL,
        LOAD_AR,
        LOAD_R,
        LOAD_PUSH,
        LOAD_DONE,
        LOAD_ERROR
    } load_state_t;

    typedef enum logic [3:0] {
        STORE_IDLE,
        STORE_REQ,
        STORE_WAIT1,
        STORE_WAIT2,
        STORE_WAIT3,
        STORE_AW_W,
        STORE_B,
        STORE_DONE,
        STORE_ERROR
    } store_state_t;

    wr_state_t wr_state_q;
    rd_state_t rd_state_q;
    load_state_t load_state_q;
    store_state_t store_state_q;

    logic aw_pending_q;
    logic w_pending_q;
    logic rd_output_wait_q;
    logic [AXI_ADDR_WIDTH-1:0] awaddr_q;
    logic [31:0] wdata_q;
    logic [3:0] wstrb_q;

    logic start_pulse;
    logic clear_done_pulse;
    logic start_ddr_load_pulse;
    logic start_ddr_store_pulse;
    logic clear_ddr_flags_pulse;
    logic ddr_load_owner_q;
    logic clear_perf_pulse;
    logic ddr_cmd_error_q;
    logic transformer_only_q;
    logic [5:0] patch_requant_shift_q;
    logic stage_checkpoint_enable_q;
    logic stage_resume_pulse;
    logic clear_stage_pulse;

    logic input_wr_valid_sw;
    logic input_wr_valid_dma;
    logic input_wr_valid_core;
    logic input_wr_ready;
    logic [3:0] input_wr_addr_sw;
    logic [3:0] input_wr_addr_dma;
    logic [3:0] input_wr_addr_core;
    logic [31:0] input_wr_data_sw;
    logic [31:0] input_wr_data_dma;
    logic [31:0] input_wr_data_core;
    logic [3:0] input_wr_strb_sw;
    logic [3:0] input_wr_strb_dma;
    logic [3:0] input_wr_strb_core;
    logic input_busy;
    logic input_done;
    logic input_error;
    logic [3:0] input_active_target;
    logic [LOAD_ADDR_W:0] input_word_count;

    logic output_host_en_sw;
    logic output_host_en_dma;
    logic output_host_en_core;
    logic [3:0] output_host_target_core;
    logic [ADDR_W-1:0] output_host_addr_sw;
    logic [ADDR_W-1:0] output_host_addr_dma;
    logic [ADDR_W-1:0] output_host_addr_core;
    logic [31:0] output_host_rd_data;

    logic busy;
    logic done_pulse;
    logic done_sticky;
    logic [31:0] status_word;
    logic [31:0] patch_request_word;
    logic [31:0] trans_request_word;
    logic [31:0] page_request_word;
    logic [31:0] debug_word;
    logic stage_checkpoint_pending;
    logic core_stage_done_pulse;
    logic [3:0] core_stage_id;
    logic [4:0] core_stage_phase;
    logic [7:0] stage_sequence_q;
    logic stage_done_sticky_q;
    logic [3:0] stage_id_q;
    logic [4:0] stage_phase_q;

    logic [M_AXI_ADDR_WIDTH-1:0] ddr_src_addr_q;
    logic [M_AXI_ADDR_WIDTH-1:0] ddr_dst_addr_q;
    logic [LOAD_ADDR_W-1:0] ddr_local_base_q;
    logic [LOAD_ADDR_W:0] ddr_word_count_q;
    logic [3:0] ddr_target_q;
    logic [LOAD_ADDR_W:0] ddr_load_index_q;
    logic [LOAD_ADDR_W:0] ddr_store_index_q;
    logic [31:0] ddr_read_data_q;
    logic [31:0] ddr_store_data_q;
    logic ddr_load_done_q;
    logic ddr_load_error_q;
    logic ddr_store_done_q;
    logic ddr_store_error_q;
    logic ddr_load_done_visible;
    logic ddr_store_done_visible;
    logic gelu_store_done_pulse;

    logic [31:0] perf_total_cycles_q;
    logic [31:0] perf_compute_busy_cycles_q;
    logic [31:0] perf_ddr_load_cycles_q;
    logic [31:0] perf_ddr_store_cycles_q;
    logic [31:0] perf_overlap_cycles_q;
    logic [31:0] perf_tile_stall_cycles_q;
    logic [31:0] perf_ddr_read_words_q;
    logic [31:0] perf_ddr_write_words_q;
    logic [31:0] perf_loader_data_words_q;
    logic [31:0] perf_output_words_q;
    logic [31:0] perf_host_loader_words_q;
    logic [31:0] perf_output_read_cycles_q;
    logic [31:0] perf_active_cycles_q;
    logic [31:0] perf_bram_read_words_q;
    logic [31:0] perf_bram_write_words_q;
    logic [31:0] perf_bram_access_cycles_q;
    logic [63:0] perf_mac_ops_q;
    logic [31:0] perf_pingpong_wait_cycles_q;
    logic [31:0] perf_pingpong_load_cycles_q;
    logic [31:0] perf_pingpong_overlap_cycles_q;
    logic [31:0] stage_base_total_q;
    logic [31:0] stage_base_busy_q;
    logic [31:0] stage_base_ddr_load_q;
    logic [31:0] stage_base_ddr_store_q;
    logic [31:0] stage_base_ddr_read_words_q;
    logic [31:0] stage_base_ddr_write_words_q;
    logic [31:0] stage_base_bram_read_words_q;
    logic [31:0] stage_base_bram_write_words_q;
    logic [31:0] stage_base_bram_access_cycles_q;
    logic [63:0] stage_base_mac_ops_q;
    logic [31:0] stage_base_pingpong_wait_q;
    logic [31:0] stage_base_pingpong_load_q;
    logic [31:0] stage_base_pingpong_overlap_q;
    logic [31:0] stage_delta_total_q;
    logic [31:0] stage_delta_busy_q;
    logic [31:0] stage_delta_ddr_load_q;
    logic [31:0] stage_delta_ddr_store_q;
    logic [31:0] stage_delta_ddr_read_words_q;
    logic [31:0] stage_delta_ddr_write_words_q;
    logic [31:0] stage_delta_bram_read_words_q;
    logic [31:0] stage_delta_bram_write_words_q;
    logic [31:0] stage_delta_bram_access_cycles_q;
    logic [63:0] stage_delta_mac_ops_q;
    logic [31:0] stage_delta_pingpong_wait_q;
    logic [31:0] stage_delta_pingpong_load_q;
    logic [31:0] stage_delta_pingpong_overlap_q;
    logic [7:0]  core_perf_bram_rd_words;
    logic [7:0]  core_perf_bram_wr_words;
    logic        core_perf_bram_active;
    logic [15:0] core_perf_mac_ops;
    logic        core_perf_pingpong_wait;
    logic        core_perf_pingpong_load;
    logic        core_perf_pingpong_overlap;

    logic aw_fire;
    logic w_fire;
    logic ar_fire;
    logic write_buffer_full;
    logic write_is_input;
    logic read_is_output;
    logic ddr_load_busy;
    logic ddr_store_busy;
    logic ddr_busy;
    logic tile_stall_now;
    logic perf_active_now;
    logic [ADDR_W-1:0] output_addr_from_axi;
    logic [31:0] ddr_status_word;

    assign aw_fire = s_axi_awvalid && s_axi_awready;
    assign w_fire = s_axi_wvalid && s_axi_wready;
    assign ar_fire = s_axi_arvalid && s_axi_arready;
    assign write_buffer_full = aw_pending_q && w_pending_q;
    assign write_is_input = (awaddr_q >= INPUT_BASE) && (awaddr_q <= INPUT_LIMIT);
    assign read_is_output = (s_axi_araddr >= OUTPUT_BASE);
    assign output_addr_from_axi = (s_axi_araddr - OUTPUT_BASE) >> 2;

    assign ddr_load_busy = ((load_state_q != LOAD_IDLE) &&
                            (load_state_q != LOAD_DONE) &&
                            (load_state_q != LOAD_ERROR)) ||
                           (ddr_load_owner_q && input_busy && !input_done);
    assign ddr_store_busy = (store_state_q != STORE_IDLE) &&
                            (store_state_q != STORE_DONE) &&
                            (store_state_q != STORE_ERROR);
    assign ddr_busy = ddr_load_busy || ddr_store_busy;
    assign ddr_load_done_visible = ddr_load_done_q &&
                                   (load_state_q == LOAD_IDLE) &&
                                   !start_ddr_load_pulse &&
                                   !clear_ddr_flags_pulse;
    assign ddr_store_done_visible = ddr_store_done_q &&
                                    (store_state_q == STORE_IDLE) &&
                                    !start_ddr_store_pulse &&
                                    !clear_ddr_flags_pulse;
    assign tile_stall_now = |status_word[9:6] || page_request_word[31];
    assign perf_active_now = !stage_checkpoint_pending &&
                             (busy || ddr_load_busy || ddr_store_busy || input_busy);

    assign input_wr_valid_core = ddr_load_busy ? input_wr_valid_dma : input_wr_valid_sw;
    assign input_wr_addr_core  = ddr_load_busy ? input_wr_addr_dma  : input_wr_addr_sw;
    assign input_wr_data_core  = ddr_load_busy ? input_wr_data_dma  : input_wr_data_sw;
    assign input_wr_strb_core  = ddr_load_busy ? input_wr_strb_dma  : input_wr_strb_sw;

    assign output_host_en_core     = ddr_store_busy ? output_host_en_dma   : output_host_en_sw;
    assign output_host_target_core = ddr_store_busy ? ddr_target_q         : 4'd0;
    assign output_host_addr_core   = ddr_store_busy ? output_host_addr_dma : output_host_addr_sw;

    assign m_axi_awlen   = 8'd0;
    assign m_axi_awsize  = 3'b010;
    assign m_axi_awburst = 2'b01;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_arlen   = 8'd0;
    assign m_axi_arsize  = 3'b010;
    assign m_axi_arburst = 2'b01;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;

    assign ddr_status_word = {
        load_state_q,
        store_state_q,
        11'd0,
        ddr_cmd_error_q,
        ddr_store_error_q,
        ddr_load_error_q,
        ddr_store_done_visible,
        ddr_load_done_visible,
        2'd0,
        ddr_store_busy,
        ddr_load_busy,
        2'd0,
        (store_state_q == STORE_ERROR),
        (load_state_q == LOAD_ERROR)
    };

    function automatic logic [31:0] read_register;
        input logic [AXI_ADDR_WIDTH-1:0] addr;
        begin
            case (addr)
                REG_CONTROL:      read_register = 32'd0;
                REG_PATCH_SHIFT:  read_register = {26'd0, patch_requant_shift_q};
                REG_STATUS:       read_register = status_word;
                REG_DEBUG:        read_register = debug_word;
                REG_PATCH_REQ:    read_register = patch_request_word;
                REG_TRANS_REQ:    read_register = trans_request_word;
                REG_INPUT_TGT:    read_register = {28'd0, input_active_target};
                REG_INPUT_COUNT:  read_register = {{(31-LOAD_ADDR_W){1'b0}}, input_word_count};
                REG_DDR_SRC_ADDR: read_register = ddr_src_addr_q[31:0];
                REG_DDR_DST_ADDR: read_register = ddr_dst_addr_q[31:0];
                REG_DDR_BASE:     read_register = {{(32-LOAD_ADDR_W){1'b0}}, ddr_local_base_q};
                REG_DDR_COUNT:    read_register = {{(31-LOAD_ADDR_W){1'b0}}, ddr_word_count_q};
                REG_DDR_TARGET:   read_register = {28'd0, ddr_target_q};
                REG_DDR_STATUS:   read_register = ddr_status_word;
                REG_DDR_INDEX:    read_register = ddr_load_busy ?
                                      {{(31-LOAD_ADDR_W){1'b0}}, ddr_load_index_q} :
                                      {{(31-LOAD_ADDR_W){1'b0}}, ddr_store_index_q};
                REG_PERF_CONTROL: read_register = 32'd0;
                REG_PERF_TOTAL:   read_register = perf_total_cycles_q;
                REG_PERF_BUSY:    read_register = perf_compute_busy_cycles_q;
                REG_PERF_DDR_LD:  read_register = perf_ddr_load_cycles_q;
                REG_PERF_DDR_ST:  read_register = perf_ddr_store_cycles_q;
                REG_PERF_OVERLAP: read_register = perf_overlap_cycles_q;
                REG_PERF_STALL:   read_register = perf_tile_stall_cycles_q;
                REG_PERF_DDR_RDW: read_register = perf_ddr_read_words_q;
                REG_PERF_DDR_WRW: read_register = perf_ddr_write_words_q;
                REG_PERF_LDR_WR:  read_register = perf_loader_data_words_q;
                REG_PERF_OUT_WR:  read_register = perf_output_words_q;
                REG_PERF_HOST_WR: read_register = perf_host_loader_words_q;
                REG_PERF_OUT_RD:  read_register = perf_output_read_cycles_q;
                REG_PERF_ACTIVE:  read_register = perf_active_cycles_q;
                REG_PAGE_REQ:     read_register = page_request_word;
                REG_PERF_BRAM_RDW: read_register = perf_bram_read_words_q;
                REG_PERF_BRAM_WRW: read_register = perf_bram_write_words_q;
                REG_PERF_BRAM_CYC: read_register = perf_bram_access_cycles_q;
                REG_PERF_MAC_LO:   read_register = perf_mac_ops_q[31:0];
                REG_PERF_MAC_HI:   read_register = perf_mac_ops_q[63:32];
                REG_PERF_PP_WAIT:  read_register = perf_pingpong_wait_cycles_q;
                REG_PERF_PP_LOAD:  read_register = perf_pingpong_load_cycles_q;
                REG_PERF_PP_OVLP:  read_register = perf_pingpong_overlap_cycles_q;
                REG_RUN_MODE:      read_register = {31'd0, transformer_only_q};
                REG_STAGE_CONTROL: read_register = {31'd0, stage_checkpoint_enable_q};
                REG_STAGE_STATUS:  read_register = {
                                      stage_checkpoint_pending,
                                      stage_done_sticky_q,
                                      12'd0,
                                      stage_sequence_q,
                                      1'd0,
                                      stage_phase_q,
                                      stage_id_q
                                  };
                REG_STAGE_TOTAL:   read_register = stage_delta_total_q;
                REG_STAGE_BUSY:    read_register = stage_delta_busy_q;
                REG_STAGE_DDR_LD:  read_register = stage_delta_ddr_load_q;
                REG_STAGE_DDR_ST:  read_register = stage_delta_ddr_store_q;
                REG_STAGE_DDR_RDW: read_register = stage_delta_ddr_read_words_q;
                REG_STAGE_DDR_WRW: read_register = stage_delta_ddr_write_words_q;
                REG_STAGE_BRAM_RDW: read_register = stage_delta_bram_read_words_q;
                REG_STAGE_BRAM_WRW: read_register = stage_delta_bram_write_words_q;
                REG_STAGE_BRAM_CYC: read_register = stage_delta_bram_access_cycles_q;
                REG_STAGE_MAC_LO:   read_register = stage_delta_mac_ops_q[31:0];
                REG_STAGE_MAC_HI:   read_register = stage_delta_mac_ops_q[63:32];
                REG_STAGE_PP_WAIT:  read_register = stage_delta_pingpong_wait_q;
                REG_STAGE_PP_LOAD:  read_register = stage_delta_pingpong_load_q;
                REG_STAGE_PP_OVLP:  read_register = stage_delta_pingpong_overlap_q;
                default:          read_register = 32'd0;
            endcase
        end
    endfunction

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            s_axi_awready <= 1'b0;
            s_axi_wready <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_bvalid <= 1'b0;
            s_axi_arready <= 1'b0;
            s_axi_rdata <= 32'd0;
            s_axi_rresp <= 2'b00;
            s_axi_rvalid <= 1'b0;

            wr_state_q <= WR_IDLE;
            rd_state_q <= RD_IDLE;
            aw_pending_q <= 1'b0;
            w_pending_q <= 1'b0;
            rd_output_wait_q <= 1'b0;
            awaddr_q <= '0;
            wdata_q <= 32'd0;
            wstrb_q <= 4'd0;

            start_pulse <= 1'b0;
            clear_done_pulse <= 1'b0;
            start_ddr_load_pulse <= 1'b0;
            start_ddr_store_pulse <= 1'b0;
            clear_ddr_flags_pulse <= 1'b0;
            clear_perf_pulse <= 1'b0;
            stage_checkpoint_enable_q <= 1'b0;
            stage_resume_pulse <= 1'b0;
            clear_stage_pulse <= 1'b0;
            ddr_cmd_error_q <= 1'b0;
            transformer_only_q <= 1'b0;
            patch_requant_shift_q <= 6'd0;

            input_wr_valid_sw <= 1'b0;
            input_wr_addr_sw <= 4'd0;
            input_wr_data_sw <= 32'd0;
            input_wr_strb_sw <= 4'd0;
            output_host_en_sw <= 1'b0;
            output_host_addr_sw <= '0;

            ddr_src_addr_q <= '0;
            ddr_dst_addr_q <= '0;
            ddr_local_base_q <= '0;
            ddr_word_count_q <= '0;
            ddr_target_q <= 4'd0;
        end
        else begin
            start_pulse <= 1'b0;
            clear_done_pulse <= 1'b0;
            start_ddr_load_pulse <= 1'b0;
            start_ddr_store_pulse <= 1'b0;
            clear_ddr_flags_pulse <= 1'b0;
            clear_perf_pulse <= 1'b0;
            stage_resume_pulse <= 1'b0;
            clear_stage_pulse <= 1'b0;
            input_wr_valid_sw <= 1'b0;
            output_host_en_sw <= 1'b0;

            s_axi_awready <= (wr_state_q == WR_IDLE) && !aw_pending_q;
            s_axi_wready <= (wr_state_q == WR_IDLE) && !w_pending_q;
            s_axi_arready <= (rd_state_q == RD_IDLE) && !s_axi_rvalid;

            if (aw_fire) begin
                awaddr_q <= s_axi_awaddr;
                aw_pending_q <= 1'b1;
            end

            if (w_fire) begin
                wdata_q <= s_axi_wdata;
                wstrb_q <= s_axi_wstrb;
                w_pending_q <= 1'b1;
            end

            if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end

            case (wr_state_q)
                WR_IDLE: begin
                    if (write_buffer_full) begin
                        if (write_is_input) begin
                            if (ddr_load_busy) begin
                                aw_pending_q <= 1'b0;
                                w_pending_q <= 1'b0;
                                s_axi_bresp <= 2'b10;
                                s_axi_bvalid <= 1'b1;
                                wr_state_q <= WR_RESP;
                            end
                            else begin
                                input_wr_addr_sw <= awaddr_q[3:0];
                                input_wr_data_sw <= wdata_q;
                                input_wr_strb_sw <= wstrb_q;
                                input_wr_valid_sw <= 1'b1;
                                wr_state_q <= WR_WAIT_INPUT;
                            end
                        end
                        else begin
                            case (awaddr_q)
                                REG_CONTROL: begin
                                    if (wdata_q[0]) begin
                                        start_pulse <= 1'b1;
                                    end
                                    if (wdata_q[1]) begin
                                        clear_done_pulse <= 1'b1;
                                    end
                                end

                                REG_PATCH_SHIFT: begin
                                    patch_requant_shift_q <= wdata_q[5:0];
                                end

                                REG_RUN_MODE: begin
                                    transformer_only_q <= wdata_q[0];
                                end

                                REG_STAGE_CONTROL: begin
                                    stage_checkpoint_enable_q <= wdata_q[0];
                                    if (wdata_q[1]) begin
                                        stage_resume_pulse <= 1'b1;
                                    end
                                    if (wdata_q[2]) begin
                                        clear_stage_pulse <= 1'b1;
                                    end
                                end

                                REG_DDR_SRC_ADDR: begin
                                    ddr_src_addr_q <= {{(M_AXI_ADDR_WIDTH-32){1'b0}}, wdata_q};
                                end

                                REG_DDR_DST_ADDR: begin
                                    ddr_dst_addr_q <= {{(M_AXI_ADDR_WIDTH-32){1'b0}}, wdata_q};
                                end

                                REG_DDR_BASE: begin
                                    ddr_local_base_q <= wdata_q[LOAD_ADDR_W-1:0];
                                end

                                REG_DDR_COUNT: begin
                                    ddr_word_count_q <= wdata_q[LOAD_ADDR_W:0];
                                end

                                REG_DDR_TARGET: begin
                                    ddr_target_q <= wdata_q[3:0];
                                end

                                REG_DDR_CONTROL: begin
                                    if (wdata_q[8]) begin
                                        clear_ddr_flags_pulse <= 1'b1;
                                        ddr_cmd_error_q <= 1'b0;
                                    end
                                    if ((wdata_q[0] && wdata_q[1]) || ((wdata_q[0] || wdata_q[1]) && ddr_busy)) begin
                                        ddr_cmd_error_q <= 1'b1;
                                    end
                                    else begin
                                        if (wdata_q[0]) begin
                                            start_ddr_load_pulse <= 1'b1;
                                        end
                                        if (wdata_q[1]) begin
                                            start_ddr_store_pulse <= 1'b1;
                                        end
                                    end
                                end

                                REG_PERF_CONTROL: begin
                                    if (wdata_q[0]) begin
                                        clear_perf_pulse <= 1'b1;
                                    end
                                end

                                default: begin
                                end
                            endcase

                            aw_pending_q <= 1'b0;
                            w_pending_q <= 1'b0;
                            s_axi_bresp <= 2'b00;
                            s_axi_bvalid <= 1'b1;
                            wr_state_q <= WR_RESP;
                        end
                    end
                end

                WR_WAIT_INPUT: begin
                    input_wr_addr_sw <= awaddr_q[3:0];
                    input_wr_data_sw <= wdata_q;
                    input_wr_strb_sw <= wstrb_q;

                    if (input_wr_ready) begin
                        input_wr_valid_sw <= 1'b0;
                        aw_pending_q <= 1'b0;
                        w_pending_q <= 1'b0;
                        s_axi_bresp <= 2'b00;
                        s_axi_bvalid <= 1'b1;
                        wr_state_q <= WR_RESP;
                    end
                    else begin
                        input_wr_valid_sw <= 1'b1;
                    end
                end

                WR_RESP: begin
                    if (!s_axi_bvalid) begin
                        wr_state_q <= WR_IDLE;
                    end
                end

                default: begin
                    wr_state_q <= WR_IDLE;
                end
            endcase

            case (rd_state_q)
                RD_IDLE: begin
                    if (ar_fire) begin
                        if (read_is_output) begin
                            if (ddr_store_busy) begin
                                s_axi_rdata <= 32'd0;
                                s_axi_rresp <= 2'b10;
                                s_axi_rvalid <= 1'b1;
                                rd_state_q <= RD_RESP;
                            end
                            else begin
                                output_host_en_sw <= 1'b1;
                                output_host_addr_sw <= output_addr_from_axi;
                                rd_output_wait_q <= 1'b0;
                                rd_state_q <= RD_WAIT_OUTPUT;
                            end
                        end
                        else begin
                            s_axi_rdata <= read_register(s_axi_araddr);
                            s_axi_rresp <= 2'b00;
                            s_axi_rvalid <= 1'b1;
                            rd_state_q <= RD_RESP;
                        end
                    end
                end

                RD_WAIT_OUTPUT: begin
                    if (rd_output_wait_q) begin
                        rd_state_q <= RD_CAPTURE_OUTPUT;
                    end
                    else begin
                        rd_output_wait_q <= 1'b1;
                    end
                end

                RD_CAPTURE_OUTPUT: begin
                    s_axi_rdata <= output_host_rd_data;
                    s_axi_rresp <= 2'b00;
                    s_axi_rvalid <= 1'b1;
                    rd_state_q <= RD_RESP;
                end

                RD_RESP: begin
                    if (!s_axi_rvalid) begin
                        rd_state_q <= RD_IDLE;
                    end
                end

                default: begin
                    rd_state_q <= RD_IDLE;
                end
            endcase
        end
    end

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            load_state_q <= LOAD_IDLE;
            input_wr_valid_dma <= 1'b0;
            input_wr_addr_dma <= 4'd0;
            input_wr_data_dma <= 32'd0;
            input_wr_strb_dma <= 4'h0;
            m_axi_araddr <= '0;
            m_axi_arvalid <= 1'b0;
            m_axi_rready <= 1'b0;
            ddr_read_data_q <= 32'd0;
            ddr_load_index_q <= '0;
            ddr_load_done_q <= 1'b0;
            ddr_load_error_q <= 1'b0;
            ddr_load_owner_q <= 1'b0;
        end
        else begin
            input_wr_valid_dma <= 1'b0;
            if (clear_ddr_flags_pulse) begin
                ddr_load_done_q <= 1'b0;
                ddr_load_error_q <= 1'b0;
                if (!input_busy) begin
                    ddr_load_owner_q <= 1'b0;
                end
            end

            // Formal DDR load completion is handled only by LOAD_PUSH -> LOAD_DONE.
            // Do not infer completion from input_done here: host_done is sticky in
            // ViT_InputLoadFSM and can still reflect the previous command while a
            // new DDR load has just armed the input loader.  That false-positive
            // path was clearing ddr_load_owner_q and returning to LOAD_IDLE before
            // any m_axi_arvalid was issued, leaving input_busy high forever.
            case (load_state_q)

                LOAD_IDLE: begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b0;
                    if (ddr_load_owner_q && input_busy && !input_done &&
                        (input_active_target == ddr_target_q) && (ddr_word_count_q != '0)) begin
                        ddr_load_done_q <= 1'b0;
                        ddr_load_error_q <= 1'b0;
                        ddr_load_index_q <= input_word_count[LOAD_ADDR_W-1:0];
                        load_state_q <= LOAD_AR;
                    end
                    else if (start_ddr_load_pulse) begin
                        ddr_load_done_q <= 1'b0;
                        ddr_load_error_q <= 1'b0;
                        ddr_load_owner_q <= 1'b1;
                        ddr_load_index_q <= '0;
                        load_state_q <= LOAD_CFG_BASE;
                    end
                end

                LOAD_CFG_BASE: begin
                    input_wr_valid_dma <= 1'b1;
                    input_wr_addr_dma <= 4'h4;
                    input_wr_data_dma <= {{(32-LOAD_ADDR_W){1'b0}}, ddr_local_base_q};
                    input_wr_strb_dma <= 4'hf;
                    if (input_wr_ready) begin
                        load_state_q <= LOAD_CFG_COUNT;
                    end
                end

                LOAD_CFG_COUNT: begin
                    input_wr_valid_dma <= 1'b1;
                    input_wr_addr_dma <= 4'h8;
                    input_wr_data_dma <= {{(31-LOAD_ADDR_W){1'b0}}, ddr_word_count_q};
                    input_wr_strb_dma <= 4'hf;
                    if (input_wr_ready) begin
                        load_state_q <= LOAD_CFG_CTRL;
                    end
                end

                LOAD_CFG_CTRL: begin
                    input_wr_valid_dma <= 1'b1;
                    input_wr_addr_dma <= 4'h0;
                    input_wr_data_dma <= {24'd0, ddr_target_q, 3'd0, 1'b1};
                    input_wr_strb_dma <= 4'hf;
                    if (input_wr_ready) begin
                        if (ddr_word_count_q == '0) begin
                            load_state_q <= LOAD_DONE;
                        end
                        else begin
                            ddr_load_index_q <= '0;
                            load_state_q <= LOAD_AR;
                        end
                    end
                end

                LOAD_AR: begin
                    m_axi_araddr <= ddr_src_addr_q + (ddr_load_index_q << 2);
                    m_axi_arvalid <= 1'b1;
                    if (m_axi_arvalid && m_axi_arready) begin
                        m_axi_arvalid <= 1'b0;
                        m_axi_rready <= 1'b1;
                        load_state_q <= LOAD_R;
                    end
                end

                LOAD_R: begin
                    if (m_axi_rvalid && m_axi_rready) begin
                        m_axi_rready <= 1'b0;
                        ddr_read_data_q <= m_axi_rdata[31:0];
                        if (m_axi_rresp != 2'b00) begin
                            ddr_load_error_q <= 1'b1;
                            load_state_q <= LOAD_ERROR;
                        end
                        else begin
                            load_state_q <= LOAD_PUSH;
                        end
                    end
                end

                LOAD_PUSH: begin
                    if (input_done &&
                        (input_active_target == ddr_target_q) &&
                        (input_word_count >= (ddr_word_count_q - 1'b1))) begin
                        input_wr_valid_dma <= 1'b0;
                        load_state_q <= LOAD_DONE;
                    end
                    else begin
                        input_wr_valid_dma <= 1'b1;
                        input_wr_addr_dma <= 4'hc;
                        input_wr_data_dma <= ddr_read_data_q;
                        input_wr_strb_dma <= 4'hf;
                        if (input_wr_ready) begin
                            if (ddr_load_index_q == (ddr_word_count_q - 1'b1)) begin
                                load_state_q <= LOAD_DONE;
                            end
                            else begin
                                ddr_load_index_q <= ddr_load_index_q + 1'b1;
                                load_state_q <= LOAD_AR;
                            end
                        end
                    end
                end

                LOAD_DONE: begin
                    ddr_load_done_q <= 1'b1;
                    ddr_load_owner_q <= 1'b0;
                    load_state_q <= LOAD_IDLE;
                end

                LOAD_ERROR: begin
                    m_axi_arvalid <= 1'b0;
                    m_axi_rready <= 1'b0;
                    ddr_load_error_q <= 1'b1;
                    ddr_load_owner_q <= 1'b0;
                    load_state_q <= LOAD_IDLE;
                end

                default: begin
                    load_state_q <= LOAD_IDLE;
                end
            endcase
        end
    end

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            store_state_q <= STORE_IDLE;
            output_host_en_dma <= 1'b0;
            output_host_addr_dma <= '0;
            m_axi_awaddr <= '0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata <= '0;
            m_axi_wstrb <= '0;
            m_axi_wlast <= 1'b0;
            m_axi_wvalid <= 1'b0;
            m_axi_bready <= 1'b0;
            ddr_store_data_q <= 32'd0;
            ddr_store_index_q <= '0;
            ddr_store_done_q <= 1'b0;
            ddr_store_error_q <= 1'b0;
            gelu_store_done_pulse <= 1'b0;
        end
        else begin
            output_host_en_dma <= 1'b0;
            gelu_store_done_pulse <= 1'b0;
            if (clear_ddr_flags_pulse) begin
                ddr_store_done_q <= 1'b0;
                ddr_store_error_q <= 1'b0;
            end

            case (store_state_q)
                STORE_IDLE: begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_wvalid <= 1'b0;
                    m_axi_bready <= 1'b0;
                    m_axi_wlast <= 1'b0;
                    if (start_ddr_store_pulse) begin
                        ddr_store_done_q <= 1'b0;
                        ddr_store_error_q <= 1'b0;
                        ddr_store_index_q <= '0;
                        store_state_q <= STORE_REQ;
                    end
                end

                STORE_REQ: begin
                    if (ddr_word_count_q == '0) begin
                        store_state_q <= STORE_DONE;
                    end
                    else begin
                        output_host_en_dma <= 1'b1;
                        output_host_addr_dma <= ddr_local_base_q[ADDR_W-1:0] + ddr_store_index_q[ADDR_W-1:0];
                        store_state_q <= STORE_WAIT1;
                    end
                end

                STORE_WAIT1: begin
                    store_state_q <= STORE_WAIT2;
                end

                STORE_WAIT2: begin
                    store_state_q <= STORE_WAIT3;
                end

                STORE_WAIT3: begin
                    ddr_store_data_q <= output_host_rd_data;
                    m_axi_awaddr <= ddr_dst_addr_q + (ddr_store_index_q << 2);
                    m_axi_wdata <= output_host_rd_data;
                    m_axi_wstrb <= 4'hf;
                    m_axi_wlast <= 1'b1;
                    m_axi_awvalid <= 1'b1;
                    m_axi_wvalid <= 1'b1;
                    store_state_q <= STORE_AW_W;
                end

                STORE_AW_W: begin
                    if (m_axi_awvalid && m_axi_awready) begin
                        m_axi_awvalid <= 1'b0;
                    end
                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        m_axi_wlast <= 1'b0;
                    end
                    if ((!m_axi_awvalid || m_axi_awready) && (!m_axi_wvalid || m_axi_wready)) begin
                        m_axi_bready <= 1'b1;
                        store_state_q <= STORE_B;
                    end
                end

                STORE_B: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 1'b0;
                        if (m_axi_bresp != 2'b00) begin
                            ddr_store_error_q <= 1'b1;
                            store_state_q <= STORE_ERROR;
                        end
                        else if (ddr_store_index_q == (ddr_word_count_q - 1'b1)) begin
                            store_state_q <= STORE_DONE;
                        end
                        else begin
                            ddr_store_index_q <= ddr_store_index_q + 1'b1;
                            store_state_q <= STORE_REQ;
                        end
                    end
                end

                STORE_DONE: begin
                    ddr_store_done_q <= 1'b1;
                    gelu_store_done_pulse <= (ddr_target_q == 4'd9);
                    store_state_q <= STORE_IDLE;
                end

                STORE_ERROR: begin
                    m_axi_awvalid <= 1'b0;
                    m_axi_wvalid <= 1'b0;
                    m_axi_wlast <= 1'b0;
                    m_axi_bready <= 1'b0;
                    ddr_store_error_q <= 1'b1;
                    store_state_q <= STORE_IDLE;
                end

                default: begin
                    store_state_q <= STORE_IDLE;
                end
            endcase
        end
    end

    // ------------------------------------------------------------
    // ?祕蝖祇?? counter嚗ython ?芾??? register ?銵具?
    // ------------------------------------------------------------
    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            perf_total_cycles_q <= 32'd0;
            perf_compute_busy_cycles_q <= 32'd0;
            perf_ddr_load_cycles_q <= 32'd0;
            perf_ddr_store_cycles_q <= 32'd0;
            perf_overlap_cycles_q <= 32'd0;
            perf_tile_stall_cycles_q <= 32'd0;
            perf_ddr_read_words_q <= 32'd0;
            perf_ddr_write_words_q <= 32'd0;
            perf_loader_data_words_q <= 32'd0;
            perf_output_words_q <= 32'd0;
            perf_host_loader_words_q <= 32'd0;
            perf_output_read_cycles_q <= 32'd0;
            perf_active_cycles_q <= 32'd0;
            perf_bram_read_words_q <= 32'd0;
            perf_bram_write_words_q <= 32'd0;
            perf_bram_access_cycles_q <= 32'd0;
            perf_mac_ops_q <= 64'd0;
            perf_pingpong_wait_cycles_q <= 32'd0;
            perf_pingpong_load_cycles_q <= 32'd0;
            perf_pingpong_overlap_cycles_q <= 32'd0;
        end
        else if (clear_perf_pulse) begin
            perf_total_cycles_q <= 32'd0;
            perf_compute_busy_cycles_q <= 32'd0;
            perf_ddr_load_cycles_q <= 32'd0;
            perf_ddr_store_cycles_q <= 32'd0;
            perf_overlap_cycles_q <= 32'd0;
            perf_tile_stall_cycles_q <= 32'd0;
            perf_ddr_read_words_q <= 32'd0;
            perf_ddr_write_words_q <= 32'd0;
            perf_loader_data_words_q <= 32'd0;
            perf_output_words_q <= 32'd0;
            perf_host_loader_words_q <= 32'd0;
            perf_output_read_cycles_q <= 32'd0;
            perf_active_cycles_q <= 32'd0;
            perf_bram_read_words_q <= 32'd0;
            perf_bram_write_words_q <= 32'd0;
            perf_bram_access_cycles_q <= 32'd0;
            perf_mac_ops_q <= 64'd0;
            perf_pingpong_wait_cycles_q <= 32'd0;
            perf_pingpong_load_cycles_q <= 32'd0;
            perf_pingpong_overlap_cycles_q <= 32'd0;
        end
        else begin
            if (perf_active_now) begin
                perf_total_cycles_q <= perf_total_cycles_q + 1'b1;
            end
            if (busy && !stage_checkpoint_pending) begin
                perf_compute_busy_cycles_q <= perf_compute_busy_cycles_q + 1'b1;
            end
            if (ddr_load_busy && !stage_checkpoint_pending) begin
                perf_ddr_load_cycles_q <= perf_ddr_load_cycles_q + 1'b1;
            end
            if (ddr_store_busy && !stage_checkpoint_pending) begin
                perf_ddr_store_cycles_q <= perf_ddr_store_cycles_q + 1'b1;
            end
            if (ddr_load_busy && busy && !stage_checkpoint_pending) begin
                perf_overlap_cycles_q <= perf_overlap_cycles_q + 1'b1;
            end
            if (tile_stall_now && !stage_checkpoint_pending) begin
                perf_tile_stall_cycles_q <= perf_tile_stall_cycles_q + 1'b1;
            end
            if (m_axi_rvalid && m_axi_rready && (m_axi_rresp == 2'b00) &&
                !stage_checkpoint_pending) begin
                perf_ddr_read_words_q <= perf_ddr_read_words_q + 1'b1;
            end
            if (m_axi_bvalid && m_axi_bready && (m_axi_bresp == 2'b00) &&
                !stage_checkpoint_pending) begin
                perf_ddr_write_words_q <= perf_ddr_write_words_q + 1'b1;
                perf_output_words_q <= perf_output_words_q + 1'b1;
            end
            if (input_wr_valid_core && input_wr_ready && (input_wr_addr_core == 4'hc) &&
                !stage_checkpoint_pending) begin
                perf_loader_data_words_q <= perf_loader_data_words_q + 1'b1;
            end
            if (input_wr_valid_sw && input_wr_ready && (input_wr_addr_sw == 4'hc) &&
                !stage_checkpoint_pending) begin
                perf_host_loader_words_q <= perf_host_loader_words_q + 1'b1;
            end
            if (output_host_en_core && !stage_checkpoint_pending) begin
                perf_output_read_cycles_q <= perf_output_read_cycles_q + 1'b1;
            end
            if (perf_active_now || start_pulse ||
                (start_ddr_load_pulse && !stage_checkpoint_pending) ||
                (start_ddr_store_pulse && !stage_checkpoint_pending)) begin
                perf_active_cycles_q <= perf_active_cycles_q + 1'b1;
            end
            if ((core_perf_bram_rd_words != 8'd0) && !stage_checkpoint_pending) begin
                perf_bram_read_words_q <= perf_bram_read_words_q + {24'd0, core_perf_bram_rd_words};
            end
            if ((core_perf_bram_wr_words != 8'd0) && !stage_checkpoint_pending) begin
                perf_bram_write_words_q <= perf_bram_write_words_q + {24'd0, core_perf_bram_wr_words};
            end
            if (core_perf_bram_active && !stage_checkpoint_pending) begin
                perf_bram_access_cycles_q <= perf_bram_access_cycles_q + 1'b1;
            end
            if ((core_perf_mac_ops != 16'd0) && !stage_checkpoint_pending) begin
                perf_mac_ops_q <= perf_mac_ops_q + {48'd0, core_perf_mac_ops};
            end
            if (core_perf_pingpong_wait && !stage_checkpoint_pending) begin
                perf_pingpong_wait_cycles_q <= perf_pingpong_wait_cycles_q + 1'b1;
            end
            if (core_perf_pingpong_load && !stage_checkpoint_pending) begin
                perf_pingpong_load_cycles_q <= perf_pingpong_load_cycles_q + 1'b1;
            end
            if (core_perf_pingpong_overlap && !stage_checkpoint_pending) begin
                perf_pingpong_overlap_cycles_q <= perf_pingpong_overlap_cycles_q + 1'b1;
            end
        end
    end

    always_ff @(posedge s_axi_aclk or negedge s_axi_aresetn) begin
        if (!s_axi_aresetn) begin
            stage_sequence_q <= 8'd0;
            stage_done_sticky_q <= 1'b0;
            stage_id_q <= 4'd0;
            stage_phase_q <= 5'd0;
            stage_base_total_q <= 32'd0;
            stage_base_busy_q <= 32'd0;
            stage_base_ddr_load_q <= 32'd0;
            stage_base_ddr_store_q <= 32'd0;
            stage_base_ddr_read_words_q <= 32'd0;
            stage_base_ddr_write_words_q <= 32'd0;
            stage_base_bram_read_words_q <= 32'd0;
            stage_base_bram_write_words_q <= 32'd0;
            stage_base_bram_access_cycles_q <= 32'd0;
            stage_base_mac_ops_q <= 64'd0;
            stage_base_pingpong_wait_q <= 32'd0;
            stage_base_pingpong_load_q <= 32'd0;
            stage_base_pingpong_overlap_q <= 32'd0;
            stage_delta_total_q <= 32'd0;
            stage_delta_busy_q <= 32'd0;
            stage_delta_ddr_load_q <= 32'd0;
            stage_delta_ddr_store_q <= 32'd0;
            stage_delta_ddr_read_words_q <= 32'd0;
            stage_delta_ddr_write_words_q <= 32'd0;
            stage_delta_bram_read_words_q <= 32'd0;
            stage_delta_bram_write_words_q <= 32'd0;
            stage_delta_bram_access_cycles_q <= 32'd0;
            stage_delta_mac_ops_q <= 64'd0;
            stage_delta_pingpong_wait_q <= 32'd0;
            stage_delta_pingpong_load_q <= 32'd0;
            stage_delta_pingpong_overlap_q <= 32'd0;
        end
        else if (clear_perf_pulse) begin
            stage_sequence_q <= 8'd0;
            stage_done_sticky_q <= 1'b0;
            stage_id_q <= 4'd0;
            stage_phase_q <= 5'd0;
            stage_base_total_q <= 32'd0;
            stage_base_busy_q <= 32'd0;
            stage_base_ddr_load_q <= 32'd0;
            stage_base_ddr_store_q <= 32'd0;
            stage_base_ddr_read_words_q <= 32'd0;
            stage_base_ddr_write_words_q <= 32'd0;
            stage_base_bram_read_words_q <= 32'd0;
            stage_base_bram_write_words_q <= 32'd0;
            stage_base_bram_access_cycles_q <= 32'd0;
            stage_base_mac_ops_q <= 64'd0;
            stage_base_pingpong_wait_q <= 32'd0;
            stage_base_pingpong_load_q <= 32'd0;
            stage_base_pingpong_overlap_q <= 32'd0;
            stage_delta_total_q <= 32'd0;
            stage_delta_busy_q <= 32'd0;
            stage_delta_ddr_load_q <= 32'd0;
            stage_delta_ddr_store_q <= 32'd0;
            stage_delta_ddr_read_words_q <= 32'd0;
            stage_delta_ddr_write_words_q <= 32'd0;
            stage_delta_bram_read_words_q <= 32'd0;
            stage_delta_bram_write_words_q <= 32'd0;
            stage_delta_bram_access_cycles_q <= 32'd0;
            stage_delta_mac_ops_q <= 64'd0;
            stage_delta_pingpong_wait_q <= 32'd0;
            stage_delta_pingpong_load_q <= 32'd0;
            stage_delta_pingpong_overlap_q <= 32'd0;
        end
        else begin
            if (clear_stage_pulse) begin
                stage_done_sticky_q <= 1'b0;
            end

            if (core_stage_done_pulse) begin
                stage_done_sticky_q <= 1'b1;
                stage_sequence_q <= stage_sequence_q + 1'b1;
                stage_id_q <= core_stage_id;
                stage_phase_q <= core_stage_phase;

                stage_delta_total_q <= perf_total_cycles_q - stage_base_total_q;
                stage_delta_busy_q <= perf_compute_busy_cycles_q - stage_base_busy_q;
                stage_delta_ddr_load_q <= perf_ddr_load_cycles_q - stage_base_ddr_load_q;
                stage_delta_ddr_store_q <= perf_ddr_store_cycles_q - stage_base_ddr_store_q;
                stage_delta_ddr_read_words_q <= perf_ddr_read_words_q - stage_base_ddr_read_words_q;
                stage_delta_ddr_write_words_q <= perf_ddr_write_words_q - stage_base_ddr_write_words_q;
                stage_delta_bram_read_words_q <= perf_bram_read_words_q - stage_base_bram_read_words_q;
                stage_delta_bram_write_words_q <= perf_bram_write_words_q - stage_base_bram_write_words_q;
                stage_delta_bram_access_cycles_q <= perf_bram_access_cycles_q - stage_base_bram_access_cycles_q;
                stage_delta_mac_ops_q <= perf_mac_ops_q - stage_base_mac_ops_q;
                stage_delta_pingpong_wait_q <= perf_pingpong_wait_cycles_q - stage_base_pingpong_wait_q;
                stage_delta_pingpong_load_q <= perf_pingpong_load_cycles_q - stage_base_pingpong_load_q;
                stage_delta_pingpong_overlap_q <= perf_pingpong_overlap_cycles_q - stage_base_pingpong_overlap_q;

                stage_base_total_q <= perf_total_cycles_q;
                stage_base_busy_q <= perf_compute_busy_cycles_q;
                stage_base_ddr_load_q <= perf_ddr_load_cycles_q;
                stage_base_ddr_store_q <= perf_ddr_store_cycles_q;
                stage_base_ddr_read_words_q <= perf_ddr_read_words_q;
                stage_base_ddr_write_words_q <= perf_ddr_write_words_q;
                stage_base_bram_read_words_q <= perf_bram_read_words_q;
                stage_base_bram_write_words_q <= perf_bram_write_words_q;
                stage_base_bram_access_cycles_q <= perf_bram_access_cycles_q;
                stage_base_mac_ops_q <= perf_mac_ops_q;
                stage_base_pingpong_wait_q <= perf_pingpong_wait_cycles_q;
                stage_base_pingpong_load_q <= perf_pingpong_load_cycles_q;
                stage_base_pingpong_overlap_q <= perf_pingpong_overlap_cycles_q;
            end
        end
    end

    ViT_System_Core #(
        .IMG_H(IMG_H),
        .IMG_W(IMG_W),
        .IMG_C(IMG_C),
        .PATCH_SIZE(PATCH_SIZE),
        .EMBED_DIM(EMBED_DIM),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ADDR_W(ADDR_W),
        .HEAD_NUM(HEAD_NUM),
        .HEAD_DIM(HEAD_DIM),
        .FFN_CHANNEL_NUM(FFN_CHANNEL_NUM),
        .SOFTMAX_COLS(SOFTMAX_COLS),
        .LOAD_ADDR_W(LOAD_ADDR_W),
        .SOFTMAX_EXP_LUT_HEX(SOFTMAX_EXP_LUT_HEX),
        .PATCH_GRID_H(PATCH_GRID_H),
        .PATCH_GRID_W(PATCH_GRID_W),
        .PATCH_COUNT(PATCH_COUNT),
        .TOKEN_NUM(TOKEN_NUM),
        .PATCH_ELEMS(PATCH_ELEMS),
        .IMG_ADDR_W(IMG_ADDR_W),
        .PE_W_ADDR_W(PE_W_ADDR_W),
        .PE_POS_ADDR_W(PE_POS_ADDR_W),
        .PE_EMBED_ADDR_W(PE_EMBED_ADDR_W)
    ) u_core (
        .clk(s_axi_aclk),
        .rst_n(s_axi_aresetn),
        .start(start_pulse),
        .clear_done(clear_done_pulse),
        .transformer_only(transformer_only_q),
        .patch_requant_shift(patch_requant_shift_q),
        .busy(busy),
        .done_pulse(done_pulse),
        .done_sticky(done_sticky),
        .input_wr_valid(input_wr_valid_core),
        .input_wr_ready(input_wr_ready),
        .input_wr_addr(input_wr_addr_core),
        .input_wr_data(input_wr_data_core),
        .input_wr_strb(input_wr_strb_core),
        .input_busy(input_busy),
        .input_done(input_done),
        .input_error(input_error),
        .input_active_target(input_active_target),
        .input_word_count(input_word_count),
        .output_host_en(output_host_en_core),
        .output_host_target(output_host_target_core),
        .output_host_addr(output_host_addr_core),
        .output_host_rd_data(output_host_rd_data),
        .stage_checkpoint_enable(stage_checkpoint_enable_q),
        .stage_checkpoint_resume(stage_resume_pulse),
        .stage_checkpoint_pending(stage_checkpoint_pending),
        .stage_done_pulse(core_stage_done_pulse),
        .stage_id(core_stage_id),
        .stage_phase(core_stage_phase),
        .gelu_store_done_i(gelu_store_done_pulse),
        .status_word(status_word),
        .patch_request_word(patch_request_word),
        .trans_request_word(trans_request_word),
        .page_request_word(page_request_word),
        .debug_word(debug_word),
        .perf_bram_rd_words_o(core_perf_bram_rd_words),
        .perf_bram_wr_words_o(core_perf_bram_wr_words),
        .perf_bram_active_o(core_perf_bram_active),
        .perf_mac_ops_o(core_perf_mac_ops),
        .perf_pingpong_wait_o(core_perf_pingpong_wait),
        .perf_pingpong_load_o(core_perf_pingpong_load),
        .perf_pingpong_overlap_o(core_perf_pingpong_overlap)
    );

endmodule
