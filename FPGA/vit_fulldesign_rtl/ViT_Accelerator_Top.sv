`timescale 1ns/1ps

// ============================================================
// Module: ViT_Accelerator_Top
// Function:
//   FPGA-oriented ViT block top after removing FlashAttention.
//
//   X -> RMSNorm1 -> QKV -> Q/K/V -> QK^T -> Score -> Softmax -> A
//     -> A*V -> O_attn -> OutProj -> Residual Add 1 -> X_mid
//     -> RMSNorm2 -> FC1 -> GELU_out -> FC2 -> Residual Add 2 -> X_out
//
// FPGA notes:
//   - All planned activation/state buffers use ActivationMem / Token_Stat_BRAM.
//   - ActivationMem is synchronous: rd_data is valid one clock after rd_addr.
//   - Some physical BRAMs are reused when lifetimes do not overlap:
//       X also stores final X_out, norm stores X_norm/O_attn/X_mid_norm,
//       and v_xmid stores V then X_mid.
//   - Q is not stored as a full activation BRAM. PH_QKV precomputes K/V only;
//     before each QK^T query tile, PH_Q_TILE recomputes one token-tile x 64 Q tile into
//     a small LUT-style cache and PH_QKT consumes it immediately.
// ============================================================
module ViT_Accelerator_Top #(
    parameter int TOKEN_NUM       = 197,
    parameter int CHANNEL_NUM     = 384,
    parameter int TOKEN_TILE      = 8,
    parameter int CHANNEL_TILE    = 8,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int CHANNEL_TILE_W  = 6,
    parameter int ADDR_W          = 17,
    parameter int LOAD_ADDR_W     = 20,
    parameter int HEAD_NUM        = 6,
    parameter int HEAD_DIM        = 64,
    parameter int FFN_CHANNEL_NUM = 1536,
    parameter int SOFTMAX_COLS    = 208,
    parameter SOFTMAX_EXP_LUT_HEX = "exp_lut_10bit_Q1_15_range12.hex"
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start_exec,
    output logic busy_exec,
    output logic done_exec,
    output logic [4:0] debug_phase,

    // Host-driven checkpoint support.  When enabled, the core stops at selected
    // architectural phase boundaries so Python can dump the selected BRAM and
    // compare it with golden data before releasing the next phase.
    input  logic                 checkpoint_enable_i,
    input  logic                 checkpoint_resume_i,
    output logic                 checkpoint_pending_o,
    output logic                 stage_done_pulse_o,
    output logic [3:0]           stage_id_o,
    output logic [4:0]           stage_phase_o,
    input  logic                 debug_read_allowed_i,
    input  logic [3:0]           debug_rd_target_i,
    input  logic [ADDR_W-1:0]    debug_rd_addr_i,
    output logic [31:0]          debug_rd_data_o,

    // Host/AXI loader writes flat X address = token * CHANNEL_NUM + channel.
    input  logic                 x_buf_we,
    input  logic [ADDR_W-1:0]    x_buf_addr,
    input  logic [DATA_W-1:0]    x_buf_wdata,
    // Full image top can write the physical X BRAM word directly. This avoids
    // synthesizing flat-address divide/mod logic during patch embedding load.
    input  logic                 x_buf_word_we,
    input  logic [16:0]          x_buf_word_addr,
    input  logic [31:0]          x_buf_word_data,
    input  logic [3:0]           x_buf_word_byte_en,

    // Debug readback for final X_out. Because this is BRAM-backed, data returns one clk later.
    input  logic [ADDR_W-1:0]    x_out_raddr,
    output logic [DATA_W-1:0]    x_out_rdata,

    // GELU_out page cache. FC1 writes one on-chip page, then the wrapper stores
    // it to DDR. FC2 reloads pages on demand and stalls systolic while missing.
    input  logic                 gelu_loader_wr_en,
    input  logic [9:0]           gelu_loader_wr_addr,
    input  logic [31:0]          gelu_loader_wr_data,
    input  logic [3:0]           gelu_loader_wr_strb,
    input  logic                 gelu_load_done_i,
    input  logic [LOAD_ADDR_W-1:0] gelu_load_base_i,
    input  logic                 gelu_store_done_i,
    input  logic                 gelu_store_rd_en,
    input  logic [ADDR_W-1:0]    gelu_store_rd_addr,
    output logic [31:0]          gelu_store_rd_data,
    output logic                 gelu_page_load_req_valid,
    output logic [LOAD_ADDR_W-1:0] gelu_page_load_req_base,
    output logic                 gelu_page_store_req_valid,
    output logic [LOAD_ADDR_W-1:0] gelu_page_store_req_base,
    output logic                 gelu_page_wait_o,

    // External projection / MLP weight and bias memories.
    output logic                 w_bram_rd_en,
    output logic                 bias_bram_rd_en,
    output logic [16:0]          w_bram_addr,
    output logic [16:0]          bias_bram_addr,
    input  logic                 w_bram_valid,
    input  logic [31:0]          w_bram_data,
    input  logic [31:0]          bias_bram_data,

    // RMSNorm parameter memories.
    output logic                 rms_norm_sel_o,
    output logic [8:0]           gamma_addr,
    input  logic signed [15:0]   gamma_data,

    // Optional FPGA debug taps for ILA.
    output logic                 ppu_data_tile_valid_o,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_data_tile_o,
    output logic                 stat_valid_o,
    output logic [TOKEN_W-1:0]   stat_token_idx_o,
    output logic [SUM_W-1:0]     sum_sq_o,
    output logic [15:0]          debug_tile_count,
    output logic [15:0]          debug_softmax_count,

    // Per-cycle BRAM and ping-pong performance events.  The wrapper turns
    // these event pulses into software-readable counters.
    output logic [7:0]           perf_bram_rd_words_o,
    output logic [7:0]           perf_bram_wr_words_o,
    output logic                 perf_bram_active_o,
    output logic                 perf_pingpong_wait_o,
    output logic                 perf_pingpong_load_o,
    output logic                 perf_pingpong_overlap_o,

    // Shared systolic interface.  This module does not instantiate Systolic;
    // ViT_Image_Accelerator_Top owns the single physical array and connects it
    // here after patch embedding is complete.
    output logic                 systolic_start_o,
    input  logic                 systolic_module_ready_i,
    output logic [16:0]          systolic_act_base_addr_o,
    output logic [16:0]          systolic_w_base_addr_o,
    output logic [16:0]          systolic_bias_base_addr_o,
    output logic [7:0]           systolic_k_tile_cnt_o,
    output logic [7:0]           systolic_act_zero_point_o,
    input  logic                 systolic_act_bram_rd_en_i,
    input  logic                 systolic_w_bram_rd_en_i,
    input  logic                 systolic_bias_bram_rd_en_i,
    input  logic [16:0]          systolic_act_bram_addr_i,
    input  logic [16:0]          systolic_w_bram_addr_i,
    input  logic [16:0]          systolic_bias_bram_addr_i,
    output logic                 systolic_w_bram_valid_o,
    output logic [31:0]          systolic_act_bram_data_o,
    output logic [31:0]          systolic_w_bram_data_o,
    output logic [31:0]          systolic_bias_bram_data_o,
    input  logic                 systolic_opsum_valid_i,
    input  logic [31:0]          systolic_opsum_i,
    output logic                 systolic_opsum_ready_o
);

    localparam logic [4:0] PH_IDLE      = 5'd0;
    localparam logic [4:0] PH_RMS1      = 5'd1;
    localparam logic [4:0] PH_QKV       = 5'd2;
    localparam logic [4:0] PH_QKT       = 5'd3;
    localparam logic [4:0] PH_SOFTMAX   = 5'd4;
    localparam logic [4:0] PH_ATTN_V    = 5'd5;
    localparam logic [4:0] PH_OUT_PROJ  = 5'd6;
    localparam logic [4:0] PH_RMS2      = 5'd7;
    localparam logic [4:0] PH_FC1       = 5'd8;
    localparam logic [4:0] PH_FC2       = 5'd9;
    localparam logic [4:0] PH_Q_TILE    = 5'd11;

    localparam logic [3:0] ST_NONE      = 4'd0;
    localparam logic [3:0] ST_RMS1      = 4'd2;
    localparam logic [3:0] ST_MHSA      = 4'd3;
    localparam logic [3:0] ST_XMID      = 4'd5;
    localparam logic [3:0] ST_RMS2      = 4'd6;
    localparam logic [3:0] ST_FC1_GELU  = 4'd7;
    localparam logic [3:0] ST_XOUT      = 4'd9;

    localparam logic [3:0] DBG_T_X_WORD = 4'd10;
    localparam logic [3:0] DBG_T_NORM   = 4'd11;
    localparam logic [3:0] DBG_T_VXMID  = 4'd12;
    localparam logic [3:0] DBG_T_SHARED = 4'd13;

    localparam int TOKEN_TILE_NUM       = (TOKEN_NUM + TOKEN_TILE - 1) / TOKEN_TILE;
    localparam int PAD_TOKEN_NUM        = TOKEN_TILE_NUM * TOKEN_TILE;
    localparam int CHANNEL_TILE_NUM     = CHANNEL_NUM / CHANNEL_TILE;
    localparam int QKV_CHANNEL_TILE_NUM = CHANNEL_TILE_NUM * 3;
    localparam int HEAD_DIM_TILE_NUM    = HEAD_DIM / CHANNEL_TILE;
    localparam int SCORE_TILE_NUM       = SOFTMAX_COLS / CHANNEL_TILE;
    localparam int FFN_CHANNEL_TILE_NUM = FFN_CHANNEL_NUM / CHANNEL_TILE;
    localparam int ACT_ROW_COUNT        = TOKEN_TILE_NUM * CHANNEL_TILE_NUM * TOKEN_TILE;
    localparam int TOTAL_ELEMS          = TOKEN_NUM * CHANNEL_NUM;
    localparam int ELEM_W               = (TOTAL_ELEMS <= 1) ? 1 : $clog2(TOTAL_ELEMS + 1);
    localparam int SCORE_COUNT          = TOKEN_NUM * TOKEN_NUM;
    localparam int SCORE_WORD_COUNT     = SCORE_COUNT;
    localparam int A_HEAD_ROW_COUNT     = TOKEN_TILE_NUM * SCORE_TILE_NUM * TOKEN_TILE;
    localparam int TOKEN_TILE_LOG2      = $clog2(TOKEN_TILE);
    localparam int CHANNEL_TILE_LOG2    = $clog2(CHANNEL_TILE);
    localparam int TILE_ROW_WORDS       = CHANNEL_TILE / 4;
    localparam int TILE_ROW_WORD_LOG2   = $clog2(TILE_ROW_WORDS);
    localparam int TOKEN_WORD_GROUPS    = TOKEN_TILE / 4;
    localparam int TOKEN_WORD_GROUP_LOG2 = $clog2(TOKEN_WORD_GROUPS);
    localparam int TILE_WORDS           = TOKEN_TILE * TILE_ROW_WORDS;
    localparam int OPS_PER_TILE         = TOKEN_TILE * CHANNEL_TILE;
    localparam int A_WORD_COUNT         = A_HEAD_ROW_COUNT * TILE_ROW_WORDS;
    localparam int FFN_ROW_COUNT        = TOKEN_TILE_NUM * FFN_CHANNEL_TILE_NUM * TOKEN_TILE;
    localparam int GELU_WORD_COUNT      = FFN_ROW_COUNT * TILE_ROW_WORDS;
    localparam int SHARED_A_BASE_WORD   = SCORE_WORD_COUNT;
    localparam int SHARED_MHSA_WORDS    = SCORE_WORD_COUNT + A_WORD_COUNT;
    localparam int SHARED_WORD_COUNT    = SHARED_MHSA_WORDS;
    localparam int SHARED_BANKS         = (SHARED_WORD_COUNT + 1023) / 1024;
    localparam int GELU_PAGE_WORDS      = 1024;
    localparam int GELU_PAGE_AW         = 10;
    localparam int ACT_WORD_COUNT       = ACT_ROW_COUNT * TILE_ROW_WORDS;
    localparam int ACT_BANKS            = (ACT_WORD_COUNT + 1023) / 1024;
    localparam int CHANNEL_WORD_NUM     = (CHANNEL_NUM + 3) / 4;
    localparam int CHANNEL_WORD_W       = (CHANNEL_WORD_NUM <= 1) ? 1 : $clog2(CHANNEL_WORD_NUM);
    localparam int K_HEAD_WORD_COUNT    = SCORE_TILE_NUM * HEAD_DIM_TILE_NUM * TILE_WORDS;
    localparam int K_HEAD_BANKS         = (K_HEAD_WORD_COUNT + 1023) / 1024;
    localparam int VXMID_BANKS          = ACT_BANKS;
    localparam logic [7:0] CHANNEL_TILE_NUM_8 = CHANNEL_TILE_NUM;
    localparam logic [15:0] TOKEN_NUM_16 = TOKEN_NUM;
    localparam int Q_TILE_CACHE_WORDS   = HEAD_DIM_TILE_NUM * TILE_WORDS;
    localparam int Q_TILE_CACHE_AW      = (Q_TILE_CACHE_WORDS <= 1) ? 1 : $clog2(Q_TILE_CACHE_WORDS);
    localparam int SOFTMAX_IDX_W        = (SOFTMAX_COLS <= 2) ? 1 : $clog2(SOFTMAX_COLS);
    localparam logic [5:0] Q_TILE_SCALE = 6'd2;

    typedef enum logic [3:0] {
        ENG_IDLE,
        ENG_RMS1_STAT_START,
        ENG_RMS1_STAT_ISSUE,
        ENG_RMS1_STAT_CAPTURE,
        ENG_RMS_START,
        ENG_RMS_WAIT,
        ENG_QCACHE_START,
        ENG_QCACHE_WAIT_BUSY,
        ENG_QCACHE_WAIT_TILE,
        ENG_SYS_START,
        ENG_SYS_WAIT_BUSY,
        ENG_SYS_WAIT_TILE,
        ENG_SOFT_LOAD,
        ENG_SOFT_START,
        ENG_SOFT_WAIT
    } engine_state_t;

    engine_state_t engine_state;
    logic [4:0] active_phase;

    // 常用 stride 都是固定常數，改成 shift/add，避免 Vivado 推出乘法器或很寬的組合乘法。
    function automatic int mul_197;
        input int value;
        begin
            mul_197 = (value << 7) + (value << 6) + (value << 2) + value;
        end
    endfunction

    function automatic int mul_208;
        input int value;
        begin
            mul_208 = (value << 7) + (value << 6) + (value << 4);
        end
    endfunction

    function automatic int mul_384;
        input int value;
        begin
            mul_384 = (value << 8) + (value << 7);
        end
    endfunction

    function automatic int mul_832;
        input int value;
        begin
            mul_832 = (value << 9) + (value << 8) + (value << 6);
        end
    endfunction

    function automatic int mul_1536;
        input int value;
        begin
            mul_1536 = (value << 10) + (value << 9);
        end
    endfunction

    function automatic int mul_6144;
        input int value;
        begin
            mul_6144 = (value << 12) + (value << 11);
        end
    endfunction

    function automatic int k_tile_words;
        input int tile_idx;
        input int k_tile_count;
        begin
            k_tile_words = tile_idx * k_tile_count * TILE_WORDS;
        end
    endfunction

    function automatic int token_tile_base;
        input int tile_i;
        begin
            token_tile_base = tile_i << TOKEN_TILE_LOG2;
        end
    endfunction

    function automatic int channel_tile_base;
        input int tile_i;
        begin
            channel_tile_base = tile_i << CHANNEL_TILE_LOG2;
        end
    endfunction

    function automatic int row_word_base;
        input int row_idx;
        begin
            row_word_base = row_idx << TILE_ROW_WORD_LOG2;
        end
    endfunction

    function automatic int head_tile_base;
        input int head_i;
        begin
            head_tile_base = head_i * HEAD_DIM_TILE_NUM;
        end
    endfunction

    function automatic int head_channel_base;
        input int head_i;
        begin
            case (HEAD_DIM)
                16:      head_channel_base = head_i << 4;
                64:      head_channel_base = head_i << 6;
                default: head_channel_base = head_i << 4;
            endcase
        end
    endfunction

    function automatic int score_tile_words;
        input int tile_idx;
        begin
            score_tile_words = tile_idx * TOKEN_TILE_NUM * TILE_WORDS;
        end
    endfunction

    function automatic int head_dim_tile_words;
        input int tile_idx;
        begin
            head_dim_tile_words = tile_idx * HEAD_DIM_TILE_NUM * TILE_WORDS;
        end
    endfunction

    function automatic int act_row_index;
        input int token_idx;
        input int channel_idx;
        input int channel_tiles;
        int token_tile_idx;
        int channel_tile_idx;
        begin
            token_tile_idx   = token_idx >> TOKEN_TILE_LOG2;
            channel_tile_idx = channel_idx >> CHANNEL_TILE_LOG2;
            if (channel_tiles == 1)
                act_row_index = (token_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
            else if (channel_tiles == 8)
                act_row_index = (token_tile_idx << 6) + (channel_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
            else if (channel_tiles == 26)
                act_row_index = mul_208(token_tile_idx) + (channel_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
            else if (channel_tiles == 48)
                act_row_index = mul_384(token_tile_idx) + (channel_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
            else if (channel_tiles == 192)
                act_row_index = mul_1536(token_tile_idx) + (channel_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
            else
                act_row_index = (channel_tile_idx << TOKEN_TILE_LOG2) + (token_idx & (TOKEN_TILE - 1));
        end
    endfunction

    function automatic int act_word_index;
        input int token_idx;
        input int channel_idx;
        input int channel_tiles;
        int row_idx;
        int lane_idx;
        begin
            row_idx = act_row_index(token_idx, channel_idx, channel_tiles);
            lane_idx = (channel_idx >> 2) & (TILE_ROW_WORDS - 1);
            act_word_index = row_word_base(row_idx) + lane_idx;
        end
    endfunction

    function automatic int score_index;
        input int head_idx;
        input int query_idx;
        input int key_idx;
        begin
            // One head is processed at a time, so score BRAM only keeps the
            // valid TOKEN_NUM x TOKEN_NUM rows for the current head.
            score_index = mul_197(query_idx) + key_idx;
        end
    endfunction

    function automatic int a_word_index;
        input int token_idx;
        input int score_col_idx;
        int row_idx;
        begin
            row_idx = act_row_index(token_idx, score_col_idx, SCORE_TILE_NUM);
            a_word_index = SHARED_A_BASE_WORD + row_word_base(row_idx) +
                           ((score_col_idx >> 2) & (TILE_ROW_WORDS - 1));
        end
    endfunction

    function automatic int gelu_word_index;
        input int token_idx;
        input int ffn_channel_idx;
        int row_idx;
        begin
            row_idx = act_row_index(token_idx, ffn_channel_idx, FFN_CHANNEL_TILE_NUM);
            gelu_word_index = row_word_base(row_idx) +
                              ((ffn_channel_idx >> 2) & (TILE_ROW_WORDS - 1));
        end
    endfunction

    function automatic int qkt_k_base_word;
        input int head_i;
        input int key_tile_i;
        begin
            // K is stored only for the current head. The scheduler reloads K
            // before each head, so the physical K buffer no longer has a head dimension.
            qkt_k_base_word = head_dim_tile_words(key_tile_i);
        end
    endfunction

    function automatic int qkt_k_word_index;
        input int head_i;
        input int key_tile_i;
        input int head_channel_i;
        input int token_word_sel_i;
        begin
            qkt_k_word_index = qkt_k_base_word(head_i, key_tile_i) +
                ((head_channel_i >> CHANNEL_TILE_LOG2) * TILE_WORDS) +
                ((head_channel_i & (CHANNEL_TILE - 1)) << TOKEN_WORD_GROUP_LOG2) +
                token_word_sel_i;
        end
    endfunction

    function automatic int attnv_v_base_word;
        input int head_i;
        input int out_tile_i;
        int head_tile_idx;
        begin
            head_tile_idx = head_tile_base(head_i) + out_tile_i;
            attnv_v_base_word = score_tile_words(head_tile_idx);
        end
    endfunction

    function automatic int attnv_v_word_index;
        input int head_i;
        input int out_tile_i;
        input int token_idx_i;
        input int channel_word_sel_i;
        begin
            attnv_v_word_index = attnv_v_base_word(head_i, out_tile_i) +
                ((token_idx_i >> TOKEN_TILE_LOG2) * TILE_WORDS) +
                ((token_idx_i & (TOKEN_TILE - 1)) << TILE_ROW_WORD_LOG2) +
                channel_word_sel_i;
        end
    endfunction

    function automatic logic [7:0] zp128_to_s8_byte;
        input logic [7:0] q;
        logic signed [8:0] centered;
        begin
            centered = $signed({1'b0, q}) - 9'sd128;
            zp128_to_s8_byte = centered[7:0];
        end
    endfunction

    function automatic logic [TOKEN_TILE-1:0] token_mask_for_mtile;
        input int mtile_i;
        int r;
        begin
            token_mask_for_mtile = '0;
            for (r = 0; r < TOKEN_TILE; r = r + 1) begin
            if ((token_tile_base(mtile_i) + r) < TOKEN_NUM)
                token_mask_for_mtile[r] = 1'b1;
            end
        end
    endfunction

    function automatic int token_count_for_mtile;
        input int mtile_i;
        int r;
        begin
            token_count_for_mtile = 0;
            for (r = 0; r < TOKEN_TILE; r = r + 1) begin
                if ((token_tile_base(mtile_i) + r) < TOKEN_NUM)
                    token_count_for_mtile = token_count_for_mtile + 1;
            end
        end
    endfunction

    function automatic logic phase_uses_systolic;
        input logic [4:0] phase_i;
        begin
            phase_uses_systolic =
                (phase_i == PH_QKV) || (phase_i == PH_Q_TILE) || (phase_i == PH_QKT) ||
                (phase_i == PH_ATTN_V) || (phase_i == PH_OUT_PROJ) ||
                (phase_i == PH_FC1) || (phase_i == PH_FC2);
        end
    endfunction

    function automatic logic phase_uses_ppu;
        input logic [4:0] phase_i;
        begin
            phase_uses_ppu =
                (phase_i == PH_QKV) || (phase_i == PH_Q_TILE) || (phase_i == PH_ATTN_V) ||
                (phase_i == PH_OUT_PROJ) || (phase_i == PH_FC1) || (phase_i == PH_FC2);
        end
    endfunction

    function automatic logic phase_uses_external_weight;
        input logic [4:0] phase_i;
        begin
            phase_uses_external_weight =
                (phase_i == PH_QKV) || (phase_i == PH_Q_TILE) ||
                (phase_i == PH_OUT_PROJ) || (phase_i == PH_FC1) || (phase_i == PH_FC2);
        end
    endfunction

    function automatic int phase_n_tiles;
        input logic [4:0] phase_i;
        begin
            case (phase_i)
                PH_QKV:      phase_n_tiles = 2 * HEAD_DIM_TILE_NUM;
                PH_QKT:      phase_n_tiles = SCORE_TILE_NUM;
                PH_ATTN_V:   phase_n_tiles = HEAD_DIM_TILE_NUM;
                PH_OUT_PROJ: phase_n_tiles = CHANNEL_TILE_NUM;
                PH_FC1:      phase_n_tiles = FFN_CHANNEL_TILE_NUM;
                PH_FC2:      phase_n_tiles = CHANNEL_TILE_NUM;
                default:     phase_n_tiles = 1;
            endcase
        end
    endfunction

    function automatic int phase_first_n_tile;
        input logic [4:0] phase_i;
        begin
            case (phase_i)
                // Q is produced on demand into LUT cache. PH_QKV only builds
                // the current head K/V tiles, using local n_tile 0..7.
                PH_QKV: phase_first_n_tile = 0;
                default: phase_first_n_tile = 0;
            endcase
        end
    endfunction

    function automatic int phase_last_n_tile;
        input logic [4:0] phase_i;
        begin
            case (phase_i)
                PH_QKV: phase_last_n_tile = 2 * HEAD_DIM_TILE_NUM - 1;
                default: phase_last_n_tile = phase_n_tiles(phase_i) - 1;
            endcase
        end
    endfunction

    function automatic logic [6:0] phase_first_n_tile7;
        input logic [4:0] phase_i;
        begin
            phase_first_n_tile7 = phase_first_n_tile(phase_i);
        end
    endfunction

    function automatic logic [6:0] q_head_first_n_tile7;
        input logic [2:0] head_i;
        begin
            q_head_first_n_tile7 = head_tile_base(head_i);
        end
    endfunction

    function automatic logic [6:0] q_head_last_n_tile7;
        input logic [2:0] head_i;
        begin
            q_head_last_n_tile7 = head_tile_base(head_i) + HEAD_DIM_TILE_NUM - 1;
        end
    endfunction

    function automatic int qkv_actual_n_tile;
        input int head_i;
        input int local_ntile_i;
        begin
            if (local_ntile_i < HEAD_DIM_TILE_NUM)
                qkv_actual_n_tile = CHANNEL_TILE_NUM + head_tile_base(head_i) + local_ntile_i;
            else
                qkv_actual_n_tile = 2 * CHANNEL_TILE_NUM + head_tile_base(head_i) +
                                    (local_ntile_i - HEAD_DIM_TILE_NUM);
        end
    endfunction

    function automatic int phase_dest_channel_base;
        input logic [4:0] phase_i;
        input int head_i;
        input int ntile_i;
        begin
            case (phase_i)
                PH_QKV: begin
                    if (ntile_i < HEAD_DIM_TILE_NUM)
                        phase_dest_channel_base = head_channel_base(head_i) + channel_tile_base(ntile_i);
                    else
                        phase_dest_channel_base = head_channel_base(head_i) + channel_tile_base(ntile_i - HEAD_DIM_TILE_NUM);
                end
                PH_ATTN_V: phase_dest_channel_base = head_channel_base(head_i) + channel_tile_base(ntile_i);
                default:   phase_dest_channel_base = channel_tile_base(ntile_i);
            endcase
        end
    endfunction

    function automatic logic is_last_systolic_tile;
        input logic [4:0] phase_i;
        input int head_i;
        input int mtile_i;
        input int ntile_i;
        begin
            is_last_systolic_tile =
                (ntile_i == phase_last_n_tile(phase_i)) &&
                (mtile_i == (TOKEN_TILE_NUM - 1));
        end
    endfunction
    function automatic logic phase_uses_weight_stationary_order;
        input logic [4:0] phase_i;
        begin
            // These phases reuse one external weight tile across all token tiles.
            // FC1 is intentionally excluded because the current GELU page cache
            // stores one token tile's hidden channels as contiguous 1024-word pages.
            phase_uses_weight_stationary_order =
                (phase_i == PH_QKV) ||
                (phase_i == PH_OUT_PROJ) ||
                (phase_i == PH_FC2);
        end
    endfunction
    // ------------------------------------------------------------
    // Controller and phase/tile indices.
    // ------------------------------------------------------------
    logic        ctrl_phase_start;
    logic [4:0]  ctrl_phase;
    logic        phase_done_to_ctrl;
    logic        phase_done_to_controller;
    logic        checkpoint_release_pulse;
    logic        checkpoint_take;
    logic [3:0]  checkpoint_stage_next;
    logic        ctrl_rms_start_unused;
    logic        ctrl_sys_start_unused;
    logic        ctrl_softmax_start_unused;
    logic [1:0]  ctrl_ppu_mode;
    logic [5:0]  ctrl_ppu_scale;
    logic [7:0]  ctrl_k_tile_cnt;
    logic [16:0] ctrl_w_base_addr;
    logic [16:0] ctrl_bias_base_addr;

    logic [2:0] head_idx;
    logic [2:0] mhsa_head_idx;
    logic       mhsa_repeat_to_ctrl;
    logic [4:0] m_tile_idx;
    logic [6:0] n_tile_idx;

    function automatic logic [3:0] phase_checkpoint_stage;
        input logic [4:0] phase_i;
        input logic       mhsa_repeat_i;
        begin
            case (phase_i)
                PH_RMS1:     phase_checkpoint_stage = ST_RMS1;
                PH_ATTN_V:   phase_checkpoint_stage = mhsa_repeat_i ? ST_NONE : ST_MHSA;
                PH_OUT_PROJ: phase_checkpoint_stage = ST_XMID;
                PH_RMS2:     phase_checkpoint_stage = ST_RMS2;
                PH_FC1:      phase_checkpoint_stage = ST_FC1_GELU;
                PH_FC2:      phase_checkpoint_stage = ST_XOUT;
                default:     phase_checkpoint_stage = ST_NONE;
            endcase
        end
    endfunction

    assign checkpoint_stage_next = phase_checkpoint_stage(ctrl_phase, mhsa_repeat_to_ctrl);
    assign checkpoint_take = phase_done_to_ctrl && checkpoint_enable_i &&
                             (checkpoint_stage_next != ST_NONE);
    assign phase_done_to_controller =
        (phase_done_to_ctrl && !checkpoint_take) || checkpoint_release_pulse;

    Global_Controller_FSM #(
        .CHANNEL_TILE_NUM(CHANNEL_TILE_NUM),
        .HEAD_DIM_TILE_NUM(HEAD_DIM_TILE_NUM),
        .SCORE_TILE_NUM(SCORE_TILE_NUM),
        .FFN_CHANNEL_TILE_NUM(FFN_CHANNEL_TILE_NUM)
    ) u_global_controller (
        .clk(clk),
        .rst_n(rst_n),
        .start_exec(start_exec),
        .phase_done_i(phase_done_to_controller),
        .mhsa_repeat_i(mhsa_repeat_to_ctrl),
        .busy_exec(busy_exec),
        .done_exec(done_exec),
        .phase_o(ctrl_phase),
        .phase_start_o(ctrl_phase_start),
        .rms_start(ctrl_rms_start_unused),
        .sys_en(ctrl_sys_start_unused),
        .softmax_start(ctrl_softmax_start_unused),
        .ppu_mode_o(ctrl_ppu_mode),
        .ppu_scaling_factor_o(ctrl_ppu_scale),
        .sys_k_tile_cnt(ctrl_k_tile_cnt),
        .sys_w_base_addr(ctrl_w_base_addr),
        .sys_bias_base_addr(ctrl_bias_base_addr)
    );

    assign debug_phase = ctrl_phase;
    assign mhsa_repeat_to_ctrl = (ctrl_phase == PH_ATTN_V) && (mhsa_head_idx != (HEAD_NUM - 1));

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            checkpoint_pending_o  <= 1'b0;
            checkpoint_release_pulse <= 1'b0;
            stage_done_pulse_o    <= 1'b0;
            stage_id_o            <= ST_NONE;
            stage_phase_o         <= PH_IDLE;
        end
        else begin
            checkpoint_release_pulse <= 1'b0;
            stage_done_pulse_o       <= 1'b0;

            if (checkpoint_pending_o && checkpoint_resume_i) begin
                checkpoint_pending_o    <= 1'b0;
                checkpoint_release_pulse <= 1'b1;
            end

            if (phase_done_to_ctrl && (checkpoint_stage_next != ST_NONE)) begin
                stage_done_pulse_o <= 1'b1;
                stage_id_o         <= checkpoint_stage_next;
                stage_phase_o      <= ctrl_phase;
                if (checkpoint_enable_i) begin
                    checkpoint_pending_o <= 1'b1;
                end
            end
        end
    end

    // ------------------------------------------------------------
    // Datapath control signals shared across BRAM muxes.
    // ------------------------------------------------------------
    logic rms_start_pulse;
    logic rms_busy;
    logic rms_done;
    logic rms_x_valid;
    logic rms_x_ready;
    logic [DATA_W-1:0] rms_x_in;
    logic rms_y_valid;
    logic rms_y_ready;
    logic rms_y_last;
    logic [DATA_W-1:0] rms_y_out;
    logic [TOKEN_W-1:0] rms_inv_addr;
    logic [8:0] rms_gamma_addr;
    logic [8:0] rms_gamma_prefetch_addr;
    logic [8:0] rms_gamma_addr_hold;
    logic [15:0] rms_inv_data_mux;
    logic signed [15:0] rms_gamma_data_reg;
    logic rms_act_wr_valid;
    logic rms_act_wr_ready;
    logic [ADDR_W-1:0] rms_act_wr_addr;
    logic [31:0] rms_act_wr_data;
    logic [3:0] rms_act_wr_byte_en;
    logic rms_act_wr_last;

    logic sys_start_pulse;
    logic sys_module_ready;
    logic sys_act_bram_rd_en;
    logic sys_w_bram_rd_en;
    logic sys_bias_bram_rd_en;
    logic [16:0] sys_act_bram_addr_raw;
    logic [16:0] sys_w_bram_addr_raw;
    logic [16:0] sys_bias_bram_addr_raw;
    logic [31:0] sys_act_bram_data;
    logic [31:0] sys_w_bram_data;
    logic [31:0] sys_bias_bram_data;
    logic sys_w_bram_valid;
    logic [7:0] sys_k_tile_cnt_eff;
    logic [1:0] ppu_mode_eff;
    logic [5:0] ppu_scale_eff;
    logic sys_opsum_valid;
    logic [31:0] sys_opsum;
    logic sys_opsum_fire;

    logic [8:0] psum_counter;
    logic ppu_tile_valid;
    logic ppu_tile_ready;
    logic ppu_psum_ready;
    logic ppu_data_valid_int;
    logic ppu_stat_valid_int;
    logic [TOKEN_W-1:0] ppu_stat_token_idx_int;
    logic [SUM_W-1:0] ppu_sum_sq_int;
    logic tile_done_pulse;

    // ------------------------------------------------------------
    // BRAM-backed activation/state storage.
    // ------------------------------------------------------------
    logic x_rd_en;
    logic [16:0] x_rd_addr;
    logic [31:0] x_rd_data;
    logic x_wr_en;
    logic [16:0] x_wr_addr;
    logic [31:0] x_wr_data;
    logic [3:0] x_wr_byte_en;

    logic norm_rd_en;
    logic [16:0] norm_rd_addr;
    logic [31:0] norm_rd_data;
    logic norm_wr_en;
    logic [16:0] norm_wr_addr;
    logic [31:0] norm_wr_data;
    logic [3:0] norm_wr_byte_en;

    (* ram_style = "block" *) logic [31:0] q_tile_cache [0:Q_TILE_CACHE_WORDS-1];
    logic q_cache_rd_en;
    logic [Q_TILE_CACHE_AW-1:0] q_cache_rd_addr;
    logic [31:0] q_cache_rd_data;
    logic q_cache_wr_en;
    logic [Q_TILE_CACHE_AW-1:0] q_cache_wr_addr;
    logic [31:0] q_cache_wr_data;

    logic k_rd_en;
    logic [16:0] k_rd_addr;
    logic [31:0] k_rd_data;
    logic k_wr_en;
    logic [16:0] k_wr_addr;
    logic [31:0] k_wr_data;
    logic [3:0] k_wr_byte_en;

    logic v_xmid_rd_en;
    logic [16:0] v_xmid_rd_addr;
    logic [31:0] v_xmid_rd_data;
    logic v_xmid_wr_en;
    logic [16:0] v_xmid_wr_addr;
    logic [31:0] v_xmid_wr_data;
    logic [3:0] v_xmid_wr_byte_en;

    logic shared_rd_en;
    logic [16:0] shared_rd_addr;
    logic [31:0] shared_rd_data;
    logic shared_wr_en;
    logic [16:0] shared_wr_addr;
    logic [31:0] shared_wr_data;
    logic [3:0] shared_wr_byte_en;

    logic [TOKEN_W-1:0] token_stat_wr_addr;
    logic [15:0] token_stat_wr_data;
    logic token_stat_wr_en;
    logic [TOKEN_W-1:0] ppu_token_stat_wr_addr;
    logic [15:0] ppu_token_stat_wr_data;
    logic ppu_token_stat_wr_en;
    logic [TOKEN_W-1:0] rms1_stat_wr_addr;
    logic [15:0] rms1_stat_wr_data;
    logic rms1_stat_wr_en;
    logic [TOKEN_W-1:0] token_stat_rd_addr;
    logic [15:0] token_stat_rd_data;

    assign token_stat_wr_en   = rms1_stat_wr_en || ppu_token_stat_wr_en;
    assign token_stat_wr_addr = rms1_stat_wr_en ? rms1_stat_wr_addr : ppu_token_stat_wr_addr;
    assign token_stat_wr_data = rms1_stat_wr_en ? rms1_stat_wr_data : ppu_token_stat_wr_data;

    logic rms_fetch_issue;
    logic rms_fetch_pending;
    logic rms_fetch_valid;
    logic [ELEM_W-1:0] rms_fetch_idx;
    logic [TOKEN_W-1:0] rms_fetch_token_idx;
    logic [8:0] rms_fetch_channel_idx;
    logic [1:0] rms_fetch_lane_d;
    logic rms_fetch_from_xmid_d;
    logic [DATA_W-1:0] rms_x_in_reg;
    logic [15:0] rms_inv_stat_reg;
    logic rms_done_seen;

    // RMSNorm1 ?? statistic pass?城? X buffer ??蛛?伍??token ??sum_sq??
    // ?秋??嚗? InvSqrt LUT ?綽蟡??Token_Stat_BRAM?? RMSNorm1 ??? BRAM ? inv_rms??
    logic [TOKEN_W-1:0] rms1_stat_token_idx;
    logic [CHANNEL_WORD_W-1:0] rms1_stat_word_idx;
    logic [SUM_W-1:0] rms1_stat_sum;
    logic [SUM_W-1:0] rms1_stat_word_sum;
    logic [SUM_W-1:0] rms1_stat_sum_next;

    function automatic logic [SUM_W-1:0] rms_square_u8_zp128;
        input logic [7:0] q;
        logic signed [9:0] centered_x;
        logic signed [19:0] square_x;
        begin
            centered_x = $signed({1'b0, q}) - $signed({2'b00, 8'd128});
            square_x = centered_x * centered_x;
            rms_square_u8_zp128 = {{(SUM_W-20){1'b0}}, square_x[19:0]};
        end
    endfunction

    function automatic logic [SUM_W-1:0] rms_word_sum_u8_zp128;
        input logic [31:0] word_data;
        input int channel_base;
        int lane;
        int channel_idx;
        begin
            rms_word_sum_u8_zp128 = '0;
            for (lane = 0; lane < 4; lane = lane + 1) begin
                channel_idx = channel_base + lane;
                if (channel_idx < CHANNEL_NUM) begin
                    rms_word_sum_u8_zp128 = rms_word_sum_u8_zp128 +
                        rms_square_u8_zp128(word_data[lane*DATA_W +: DATA_W]);
                end
            end
        end
    endfunction

    always_comb begin
        rms1_stat_word_sum = rms_word_sum_u8_zp128(
            x_rd_data,
            int'(rms1_stat_word_idx) * 4
        );
        rms1_stat_sum_next = rms1_stat_sum + rms1_stat_word_sum;
    end

    logic res_load_active;
    logic res_load_pending;
    logic [5:0] res_load_idx;
    logic [5:0] res_load_idx_d;
    logic res_load_in_range_d;
    logic res_load_from_xmid;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_residual_tile_reg;

    localparam logic [2:0] TW_NONE = 3'd0;
    localparam logic [2:0] TW_Q    = 3'd1;
    localparam logic [2:0] TW_K    = 3'd2;
    localparam logic [2:0] TW_V    = 3'd3;
    localparam logic [2:0] TW_O      = 3'd4;
    localparam logic [2:0] TW_XMID   = 3'd5;
    localparam logic [2:0] TW_XOUT   = 3'd6;
    localparam logic [2:0] TW_QCACHE = 3'd7;

    logic act_tile_write_active;
    logic [2:0] act_tile_write_kind;
    logic [5:0] act_tile_write_idx;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] act_tile_write_data;
    logic ppu_data_word_valid_int;
    logic [5:0] ppu_data_word_idx_int;
    logic [31:0] ppu_data_word_int;
    logic ppu_data_word_last_int;
    logic [4:0] act_tile_write_m_idx;
    logic [6:0] act_tile_write_n_idx;
    logic [2:0] act_tile_write_head_idx;

    logic out_proj_wait_armed;
    logic out_proj_wait_ppu_idle;
    logic out_proj_wait_stats;
    logic [4:0] out_proj_stat_count;
    logic [4:0] out_proj_stat_target;

    logic [1:0] x_out_lane_q;
    logic x_out_valid_q;
    logic debug_rd_valid_q;
    logic [3:0] debug_rd_target_q;

    integer x_rd_token_int;
    integer x_rd_channel_int;
    integer vx_rd_token_int;
    integer vx_rd_channel_int;
    integer wr_token_int;
    integer wr_channel_int;
    integer wr_word_sel_int;
    integer wr_byte_sel_int;
    integer wr_row_inner_int;
    integer wr_col_inner_int;
    integer wr_head_int;
    integer wr_head_channel_int;
    integer wr_tile_channel_int;
    integer wr_word_addr_int;
    integer wr_q_subtile_int;
    integer wr_q_cache_addr_int;
    integer wr_i;
    logic [31:0] act_tile_write_word32;
    logic [31:0] act_stream_write_word32;

    assign act_tile_write_word32 =
        act_tile_write_data[{act_tile_write_idx, 5'b00000} +: 32];
    assign act_stream_write_word32 = ppu_data_word_int;

    assign rms_act_wr_ready = 1'b1;
    assign rms_x_valid = rms_fetch_valid;
    assign rms_x_in = rms_x_in_reg;

    always_comb begin
        rms_fetch_issue = (engine_state == ENG_RMS_WAIT) &&
                          (rms_fetch_idx < TOTAL_ELEMS) &&
                          !rms_fetch_pending &&
                          (!rms_fetch_valid || (rms_x_valid && rms_x_ready));
    end

    // X BRAM read port: RMS1, OUT_PROJ residual preload, or host debug readback.
    always_comb begin
        x_rd_en   = 1'b0;
        x_rd_addr = 17'd0;

        if (engine_state == ENG_RMS1_STAT_ISSUE) begin
            x_rd_en   = 1'b1;
            x_rd_addr = act_word_index(rms1_stat_token_idx,
                                       int'(rms1_stat_word_idx) << 2,
                                       CHANNEL_TILE_NUM);
        end
        else if ((active_phase == PH_RMS1) && rms_fetch_issue) begin
            x_rd_en   = 1'b1;
            x_rd_addr = act_word_index(rms_fetch_token_idx,
                                       rms_fetch_channel_idx,
                                       CHANNEL_TILE_NUM);
        end
        else if ((active_phase == PH_OUT_PROJ) && res_load_active && !res_load_pending) begin
            x_rd_token_int   = token_tile_base(act_tile_write_m_idx) + (res_load_idx >> TILE_ROW_WORD_LOG2);
            x_rd_channel_int = channel_tile_base(act_tile_write_n_idx) +
                               ((res_load_idx & (TILE_ROW_WORDS - 1)) << 2);
            x_rd_en        = (x_rd_token_int < TOKEN_NUM) && (x_rd_channel_int < CHANNEL_NUM);
            x_rd_addr      = act_word_index(x_rd_token_int, x_rd_channel_int, CHANNEL_TILE_NUM);
        end
        else if (debug_read_allowed_i && (debug_rd_target_i == DBG_T_X_WORD)) begin
            x_rd_en   = 1'b1;
            x_rd_addr = debug_rd_addr_i[16:0];
        end
        else if (debug_read_allowed_i && (debug_rd_target_i == 4'd0) &&
                 (x_out_raddr < TOTAL_ELEMS)) begin
            x_rd_en   = 1'b1;
            x_rd_addr = act_word_index(x_out_raddr / CHANNEL_NUM,
                                       x_out_raddr % CHANNEL_NUM,
                                       CHANNEL_TILE_NUM);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_out_lane_q  <= 2'd0;
            x_out_valid_q <= 1'b0;
        end
        else begin
            x_out_valid_q <= debug_read_allowed_i && (debug_rd_target_i == 4'd0) &&
                             (x_out_raddr < TOTAL_ELEMS);
            x_out_lane_q  <= x_out_raddr[1:0];
        end
    end
    assign x_out_rdata = x_out_valid_q ? (x_rd_data >> (x_out_lane_q * DATA_W)) : 8'd0;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            debug_rd_valid_q  <= 1'b0;
            debug_rd_target_q <= 4'd0;
        end
        else begin
            debug_rd_valid_q  <= debug_read_allowed_i &&
                                 ((debug_rd_target_i == DBG_T_X_WORD) ||
                                  (debug_rd_target_i == DBG_T_NORM)   ||
                                  (debug_rd_target_i == DBG_T_VXMID)  ||
                                  (debug_rd_target_i == DBG_T_SHARED));
            debug_rd_target_q <= debug_rd_target_i;

        end
    end

    always_comb begin
        if (debug_rd_valid_q) begin
            case (debug_rd_target_q)
                DBG_T_X_WORD: debug_rd_data_o = x_rd_data;
                DBG_T_NORM:   debug_rd_data_o = norm_rd_data;
                DBG_T_VXMID:  debug_rd_data_o = v_xmid_rd_data;
                DBG_T_SHARED: debug_rd_data_o = shared_rd_data;
                default:      debug_rd_data_o = 32'd0;
            endcase
        end
        else begin
            debug_rd_data_o = 32'd0;
        end
    end

    // norm BRAM stores X_norm, then O_attn, then X_mid_norm; lifetimes do not overlap.
    always_comb begin
        norm_rd_en   = 1'b0;
        norm_rd_addr = 17'd0;
        if (((active_phase == PH_QKV) || (active_phase == PH_Q_TILE) ||
             (active_phase == PH_OUT_PROJ) || (active_phase == PH_FC1)) &&
            sys_act_bram_rd_en) begin
            norm_rd_en   = 1'b1;
            norm_rd_addr = sys_act_bram_addr_raw;
        end
        else if (debug_read_allowed_i && (debug_rd_target_i == DBG_T_NORM)) begin
            norm_rd_en   = 1'b1;
            norm_rd_addr = debug_rd_addr_i[16:0];
        end
    end

    // QK^T reads the tile-local Q cache below.
    assign q_cache_rd_en   = (active_phase == PH_QKT) && sys_act_bram_rd_en;
    assign q_cache_rd_addr = sys_act_bram_addr_raw[Q_TILE_CACHE_AW-1:0];

    // v_xmid BRAM stores V in systolic weight layout, then X_mid in activation layout.
    always_comb begin
        v_xmid_rd_en   = 1'b0;
        v_xmid_rd_addr = 17'd0;
        if ((active_phase == PH_ATTN_V) && sys_w_bram_rd_en) begin
            v_xmid_rd_en   = 1'b1;
            v_xmid_rd_addr = sys_w_bram_addr_raw;
        end
        else if ((active_phase == PH_RMS2) && rms_fetch_issue) begin
            v_xmid_rd_en   = 1'b1;
            v_xmid_rd_addr = act_word_index(rms_fetch_token_idx,
                                            rms_fetch_channel_idx,
                                            CHANNEL_TILE_NUM);
        end
        else if ((active_phase == PH_FC2) && res_load_active && !res_load_pending) begin
            vx_rd_token_int   = token_tile_base(act_tile_write_m_idx) + (res_load_idx >> TILE_ROW_WORD_LOG2);
            vx_rd_channel_int = channel_tile_base(act_tile_write_n_idx) +
                                ((res_load_idx & (TILE_ROW_WORDS - 1)) << 2);
            v_xmid_rd_en   = (vx_rd_token_int < TOKEN_NUM) && (vx_rd_channel_int < CHANNEL_NUM);
            v_xmid_rd_addr = act_word_index(vx_rd_token_int, vx_rd_channel_int, CHANNEL_TILE_NUM);
        end
        else if (debug_read_allowed_i && (debug_rd_target_i == DBG_T_VXMID)) begin
            v_xmid_rd_en   = 1'b1;
            v_xmid_rd_addr = debug_rd_addr_i[16:0];
        end
    end

    assign k_rd_en   = (active_phase == PH_QKT) && sys_w_bram_rd_en;
    assign k_rd_addr = sys_w_bram_addr_raw;

    always_ff @(posedge clk) begin
        if (q_cache_rd_en)
            q_cache_rd_data <= q_tile_cache[q_cache_rd_addr];
        if (q_cache_wr_en)
            q_tile_cache[q_cache_wr_addr] <= q_cache_wr_data;
    end
    always_comb begin
        x_wr_en      = 1'b0;
        x_wr_addr    = 17'd0;
        x_wr_data    = 32'd0;
        x_wr_byte_en = 4'b0000;

        norm_wr_en      = 1'b0;
        norm_wr_addr    = 17'd0;
        norm_wr_data    = 32'd0;
        norm_wr_byte_en = 4'b0000;

        q_cache_wr_en   = 1'b0;
        q_cache_wr_addr = '0;
        q_cache_wr_data = 32'd0;

        k_wr_en      = 1'b0;
        k_wr_addr    = 17'd0;
        k_wr_data    = 32'd0;
        k_wr_byte_en = 4'b0000;

        v_xmid_wr_en      = 1'b0;
        v_xmid_wr_addr    = 17'd0;
        v_xmid_wr_data    = 32'd0;
        v_xmid_wr_byte_en = 4'b0000;

        wr_token_int        = 0;
        wr_channel_int      = 0;
        wr_word_sel_int     = 0;
        wr_byte_sel_int     = 0;
        wr_row_inner_int    = 0;
        wr_col_inner_int    = 0;
        wr_head_int         = 0;
        wr_head_channel_int = 0;
        wr_tile_channel_int = 0;
        wr_word_addr_int    = 0;
        wr_q_subtile_int    = 0;
        wr_q_cache_addr_int = 0;

        if (x_buf_word_we && !busy_exec) begin
            x_wr_en      = 1'b1;
            x_wr_addr    = x_buf_word_addr;
            x_wr_data    = x_buf_word_data;
            x_wr_byte_en = x_buf_word_byte_en;
        end
        else if (x_buf_we && !busy_exec && (x_buf_addr < TOTAL_ELEMS)) begin
            wr_token_int    = x_buf_addr / CHANNEL_NUM;
            wr_channel_int  = x_buf_addr % CHANNEL_NUM;
            wr_byte_sel_int = wr_channel_int & 3;
            x_wr_en         = 1'b1;
            x_wr_addr       = act_word_index(wr_token_int, wr_channel_int, CHANNEL_TILE_NUM);
            x_wr_data[wr_byte_sel_int*DATA_W +: DATA_W] = x_buf_wdata;
            x_wr_byte_en    = 4'b0001 << wr_byte_sel_int;
        end

        // RMS RowPacker already outputs 32-bit words for ActivationMem.
        if (rms_act_wr_valid && rms_act_wr_ready) begin
            norm_wr_en      = 1'b1;
            norm_wr_addr    = rms_act_wr_addr[16:0];
            norm_wr_data    = rms_act_wr_data;
            norm_wr_byte_en = rms_act_wr_byte_en;
        end

        if (ppu_data_word_valid_int && (active_phase != PH_FC1)) begin
            wr_row_inner_int = ppu_data_word_idx_int >> TILE_ROW_WORD_LOG2;
            wr_word_sel_int  = ppu_data_word_idx_int & (TILE_ROW_WORDS - 1);
            wr_token_int     = token_tile_base(act_tile_write_m_idx) + wr_row_inner_int;

            case (act_tile_write_kind)
                TW_QCACHE: begin
                    wr_q_subtile_int = act_tile_write_n_idx - (act_tile_write_head_idx * HEAD_DIM_TILE_NUM);
                    wr_q_cache_addr_int = (wr_q_subtile_int * TILE_WORDS) +
                                          row_word_base(wr_row_inner_int) + wr_word_sel_int;
                    if ((wr_token_int < TOKEN_NUM) &&
                        (wr_q_subtile_int >= 0) && (wr_q_subtile_int < HEAD_DIM_TILE_NUM) &&
                        (wr_q_cache_addr_int < Q_TILE_CACHE_WORDS)) begin
                        q_cache_wr_en   = 1'b1;
                        q_cache_wr_addr = wr_q_cache_addr_int[Q_TILE_CACHE_AW-1:0];
                        q_cache_wr_data = act_stream_write_word32;
                    end
                end

                TW_Q, TW_O, TW_XMID, TW_XOUT: begin
                    wr_channel_int   = phase_dest_channel_base(active_phase,
                                                               act_tile_write_head_idx,
                                                               act_tile_write_n_idx) +
                                       (wr_word_sel_int << 2);
                    wr_word_addr_int = act_word_index(wr_token_int, wr_channel_int, CHANNEL_TILE_NUM);
                    if ((wr_token_int < TOKEN_NUM) && (wr_channel_int < CHANNEL_NUM)) begin
                        case (act_tile_write_kind)
                            TW_Q: begin
                                // Q is now written by TW_QCACHE, not by this full-buffer path.
                            end
                            TW_O: begin
                                norm_wr_en      = 1'b1;
                                norm_wr_addr    = wr_word_addr_int[16:0];
                                norm_wr_data    = act_stream_write_word32;
                                norm_wr_byte_en = 4'b1111;
                            end
                            TW_XMID: begin
                                v_xmid_wr_en      = 1'b1;
                                v_xmid_wr_addr    = wr_word_addr_int[16:0];
                                v_xmid_wr_data    = act_stream_write_word32;
                                v_xmid_wr_byte_en = 4'b1111;
                            end
                            TW_XOUT: begin
                                x_wr_en      = 1'b1;
                                x_wr_addr    = wr_word_addr_int[16:0];
                                x_wr_data    = act_stream_write_word32;
                                x_wr_byte_en = 4'b1111;
                            end
                            default: begin
                            end
                        endcase
                    end
                end

                TW_K: begin
                    // K is written by the transpose drain below after the whole tile
                    // has been collected in act_tile_write_data.
                end

                TW_V: begin
                    wr_row_inner_int    = ppu_data_word_idx_int >> TILE_ROW_WORD_LOG2;
                    wr_word_sel_int     = ppu_data_word_idx_int & (TILE_ROW_WORDS - 1);
                    wr_token_int        = token_tile_base(act_tile_write_m_idx) + wr_row_inner_int;
                    wr_tile_channel_int = (qkv_actual_n_tile(act_tile_write_head_idx, act_tile_write_n_idx) -
                                           (2 * CHANNEL_TILE_NUM)) << CHANNEL_TILE_LOG2;
                    wr_tile_channel_int = wr_tile_channel_int + (wr_word_sel_int << 2);
                    wr_head_int         = wr_tile_channel_int >> 6;
                    wr_head_channel_int = wr_tile_channel_int & 63;
                    if ((wr_token_int < TOKEN_NUM) && (wr_tile_channel_int < CHANNEL_NUM)) begin
                        v_xmid_wr_en      = 1'b1;
                        v_xmid_wr_addr    = attnv_v_word_index(wr_head_int,
                                                               wr_head_channel_int >> CHANNEL_TILE_LOG2,
                                                               wr_token_int,
                                                               wr_word_sel_int);
                        v_xmid_wr_data    = 32'd0;
                        for (wr_i = 0; wr_i < 4; wr_i = wr_i + 1) begin
                            v_xmid_wr_data[wr_i*DATA_W +: DATA_W] =
                                zp128_to_s8_byte(act_stream_write_word32[wr_i*DATA_W +: DATA_W]);
                        end
                        v_xmid_wr_byte_en = 4'b1111;
                    end
                end

                default: begin
                end
            endcase
        end

        if (act_tile_write_active) begin
            case (act_tile_write_kind)
                TW_K: begin
                    wr_col_inner_int    = act_tile_write_idx >> TOKEN_WORD_GROUP_LOG2;
                    wr_word_sel_int     = act_tile_write_idx & (TOKEN_WORD_GROUPS - 1);
                    wr_tile_channel_int = ((qkv_actual_n_tile(act_tile_write_head_idx, act_tile_write_n_idx) - CHANNEL_TILE_NUM) << CHANNEL_TILE_LOG2) +
                                          wr_col_inner_int;
                    wr_head_int         = wr_tile_channel_int >> 6;
                    wr_head_channel_int = wr_tile_channel_int & 63;
                    k_wr_data           = 32'd0;
                    for (wr_i = 0; wr_i < 4; wr_i = wr_i + 1) begin
                        k_wr_data[wr_i*DATA_W +: DATA_W] =
                            zp128_to_s8_byte(act_tile_write_data[((wr_word_sel_int*4 + wr_i)*CHANNEL_TILE + wr_col_inner_int)*DATA_W +: DATA_W]);
                    end
                    if ((token_tile_base(act_tile_write_m_idx) + (wr_word_sel_int << 2) < TOKEN_NUM) &&
                        (wr_tile_channel_int < CHANNEL_NUM)) begin
                        k_wr_en      = 1'b1;
                        k_wr_addr    = qkt_k_word_index(wr_head_int,
                                                        act_tile_write_m_idx,
                                                        wr_head_channel_int,
                                                        wr_word_sel_int);
                        k_wr_byte_en = 4'b1111;
                    end
                end

                default: begin
                end
            endcase
        end
    end

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(ACT_BANKS)) u_x_bram (
        .clk(clk), .rst_n(rst_n), .rd_en(x_rd_en), .rd_addr(x_rd_addr), .rd_data(x_rd_data),
        .wr_en(x_wr_en), .wr_addr(x_wr_addr), .wr_data(x_wr_data), .wr_byte_en(x_wr_byte_en)
    );

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(ACT_BANKS)) u_norm_bram (
        .clk(clk), .rst_n(rst_n), .rd_en(norm_rd_en), .rd_addr(norm_rd_addr), .rd_data(norm_rd_data),
        .wr_en(norm_wr_en), .wr_addr(norm_wr_addr), .wr_data(norm_wr_data), .wr_byte_en(norm_wr_byte_en)
    );

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(K_HEAD_BANKS)) u_k_weight_bram (
        .clk(clk), .rst_n(rst_n), .rd_en(k_rd_en), .rd_addr(k_rd_addr), .rd_data(k_rd_data),
        .wr_en(k_wr_en), .wr_addr(k_wr_addr), .wr_data(k_wr_data), .wr_byte_en(k_wr_byte_en)
    );

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(VXMID_BANKS)) u_v_xmid_bram (
        .clk(clk), .rst_n(rst_n), .rd_en(v_xmid_rd_en), .rd_addr(v_xmid_rd_addr), .rd_data(v_xmid_rd_data),
        .wr_en(v_xmid_wr_en), .wr_addr(v_xmid_wr_addr), .wr_data(v_xmid_wr_data), .wr_byte_en(v_xmid_wr_byte_en)
    );

    Token_Stat_BRAM #(
        .TOKEN_NUM(TOKEN_NUM),
        .TOKEN_AW(TOKEN_W),
        .DATA_W(16)
    ) u_token_stat_bram (
        .clk(clk),
        .rst_n(rst_n),
        .wr_valid_i(token_stat_wr_en),
        .wr_addr_i(token_stat_wr_addr),
        .wr_data_i(token_stat_wr_data),
        .rd_addr_i(token_stat_rd_addr),
        .rd_data_o(token_stat_rd_data)
    );
    // ------------------------------------------------------------
    // RMSNorm stream and norm BRAM writer.
    // ------------------------------------------------------------
    assign rms_norm_sel_o = (active_phase == PH_RMS2);
    assign rms_gamma_prefetch_addr = rms_fetch_channel_idx;
    assign gamma_addr = rms_fetch_issue ? rms_gamma_prefetch_addr : rms_gamma_addr_hold;
    // RMSNorm1/RMSNorm2 ?鞈? Token_Stat_BRAM ? inv_rms??
    assign rms_inv_data_mux = rms_inv_stat_reg;

    Streaming_RMSNorm_Unit #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_AW(TOKEN_W),
        .CHANNEL_AW(9),
        .X_W(DATA_W),
        .SCALE_W(16),
        .FRAC(14),
        .OUT_SHIFT(0),
        .EXT_IN_ZP128(1'b1),
        .EXT_OUT_ZP128(1'b1)
    ) u_rmsnorm (
        .clk(clk),
        .rst_n(rst_n),
        .start(rms_start_pulse),
        .busy(rms_busy),
        .done(rms_done),
        .x_valid(rms_x_valid),
        .x_ready(rms_x_ready),
        .x_in(rms_x_in),
        .inv_rms_addr(rms_inv_addr),
        .inv_rms_data(rms_inv_data_mux),
        .gamma_addr(rms_gamma_addr),
        .gamma_data(rms_gamma_data_reg),
        .y_valid(rms_y_valid),
        .y_ready(rms_y_ready),
        .y_last(rms_y_last),
        .y_out(rms_y_out)
    );

    Streaming_RMSNorm_RowPacker #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .SIGNED_TO_ZP128(1'b0)
    ) u_rms_rowpacker (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(rms_start_pulse),
        .base_addr_i({ADDR_W{1'b0}}),
        .s_data_i(rms_y_out),
        .s_valid_i(rms_y_valid),
        .s_ready_o(rms_y_ready),
        .s_last_i(rms_y_last),
        .act_wr_valid_o(rms_act_wr_valid),
        .act_wr_ready_i(rms_act_wr_ready),
        .act_wr_addr_o(rms_act_wr_addr),
        .act_wr_data_o(rms_act_wr_data),
        .act_wr_byte_en_o(rms_act_wr_byte_en),
        .act_wr_last_o(rms_act_wr_last)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rms_fetch_pending      <= 1'b0;
            rms_fetch_valid        <= 1'b0;
            rms_fetch_idx          <= '0;
            rms_fetch_token_idx    <= '0;
            rms_fetch_channel_idx  <= '0;
            rms_fetch_lane_d       <= 2'd0;
            rms_fetch_from_xmid_d  <= 1'b0;
            rms_x_in_reg           <= 8'd128;
            rms_inv_stat_reg       <= 16'd0;
            rms_gamma_addr_hold     <= 9'd0;
            rms_gamma_data_reg      <= 16'sd16384;
            token_stat_rd_addr     <= '0;
            rms_done_seen          <= 1'b0;
        end
        else begin
            if (rms_start_pulse) begin
                rms_fetch_pending      <= 1'b0;
                rms_fetch_valid        <= 1'b0;
                rms_fetch_idx          <= '0;
                rms_fetch_token_idx    <= '0;
                rms_fetch_channel_idx  <= '0;
                rms_fetch_lane_d       <= 2'd0;
                rms_fetch_from_xmid_d  <= (active_phase == PH_RMS2);
                rms_x_in_reg           <= 8'd128;
                rms_gamma_addr_hold     <= 9'd0;
                rms_gamma_data_reg      <= 16'sd16384;
                token_stat_rd_addr     <= '0;
                rms_done_seen          <= 1'b0;
            end
            else begin
                if (rms_x_valid && rms_x_ready)
                    rms_fetch_valid <= 1'b0;

                if (rms_fetch_pending) begin
                    rms_fetch_pending <= 1'b0;
                    rms_fetch_valid   <= 1'b1;
                    if (rms_fetch_from_xmid_d)
                        rms_x_in_reg <= v_xmid_rd_data >> (rms_fetch_lane_d * DATA_W);
                    else
                        rms_x_in_reg <= x_rd_data >> (rms_fetch_lane_d * DATA_W);
                    rms_inv_stat_reg <= token_stat_rd_data;
                    rms_gamma_data_reg <= gamma_data;
                end

                if (rms_fetch_issue) begin
                    rms_fetch_pending     <= 1'b1;
                    rms_fetch_lane_d      <= rms_fetch_channel_idx[1:0];
                    rms_fetch_from_xmid_d <= (active_phase == PH_RMS2);
                    rms_gamma_addr_hold   <= rms_gamma_prefetch_addr;
                    token_stat_rd_addr    <= rms_fetch_token_idx;
                    rms_fetch_idx         <= rms_fetch_idx + 1'b1;
                    if (rms_fetch_channel_idx == (CHANNEL_NUM - 1)) begin
                        rms_fetch_channel_idx <= '0;
                        rms_fetch_token_idx   <= rms_fetch_token_idx + 1'b1;
                    end
                    else begin
                        rms_fetch_channel_idx <= rms_fetch_channel_idx + 1'b1;
                    end
                end


                if (rms_done)
                    rms_done_seen <= 1'b1;
            end
        end
    end

    // ------------------------------------------------------------
    // Systolic source selection and shared intermediate BRAM.
    // ------------------------------------------------------------
    logic [16:0] current_sys_act_base;
    logic [16:0] current_sys_w_base;
    logic [16:0] current_sys_bias_base;
    logic [7:0]  current_sys_act_zero_point;
    logic use_external_weight;
    logic shared_act_phase;

    logic [8:0] softmax_load_idx;
    logic [8:0] softmax_load_idx_d;
    logic softmax_load_pending;
    logic shared_tile_store_after;
    logic [4:0] shared_tile_m_idx;
    logic [6:0] shared_tile_n_idx;

    logic gelu_page_wr_en;
    logic [9:0] gelu_page_wr_addr;
    logic [31:0] gelu_page_wr_data;
    logic [3:0] gelu_page_wr_byte_en;
    logic gelu_page_rd_en;
    logic [9:0] gelu_page_rd_addr;
    logic [31:0] gelu_page_rd_data;
    logic [16:0] gelu_page_base_q;
    logic [16:0] gelu_page_store_base_q;
    logic gelu_page_valid_q;
    logic gelu_page_store_pending;
    logic [16:0] gelu_required_addr;
    logic [16:0] gelu_required_page_base;
    logic [16:0] gelu_fc1_tile_base;
    logic [16:0] gelu_fc1_tile_page_base;
    logic gelu_fc1_tile_store_after;
    logic gelu_page_hit;
    logic gelu_fc2_phase_active;


    logic softmax_start_pulse;
    logic softmax_done;
    logic [2:0] softmax_head_idx;
    logic [TOKEN_W-1:0] softmax_row_idx;
    logic softmax_score_load_valid;
    logic [SOFTMAX_IDX_W-1:0] softmax_score_load_index;
    logic signed [31:0] softmax_score_load_data;
    logic softmax_attention_valid;
    logic [SOFTMAX_IDX_W-1:0] softmax_attention_index;
    logic signed [7:0] softmax_attention_data;

    integer shared_wr_r;
    integer shared_wr_c;
    integer shared_wr_token;
    integer shared_wr_channel;
    integer shared_wr_word;
    integer shared_wr_byte;
    integer gelu_wr_r;
    integer gelu_wr_c;
    integer gelu_wr_token;
    integer gelu_wr_channel;
    integer gelu_wr_word;

    always_comb begin
        use_external_weight = phase_uses_external_weight(active_phase);
        current_sys_act_base  = 17'd0;
        current_sys_w_base    = 17'd0;
        current_sys_bias_base = 17'd0;
        current_sys_act_zero_point = 8'd0;

        case (active_phase)
            PH_QKV: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base  = act_word_index(token_tile_base(m_tile_idx), 0, CHANNEL_TILE_NUM);
                current_sys_w_base    = ctrl_w_base_addr + k_tile_words(qkv_actual_n_tile(head_idx, n_tile_idx), ctrl_k_tile_cnt);
                current_sys_bias_base = ctrl_bias_base_addr + channel_tile_base(qkv_actual_n_tile(head_idx, n_tile_idx));
            end
            PH_Q_TILE: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base  = act_word_index(token_tile_base(m_tile_idx), 0, CHANNEL_TILE_NUM);
                current_sys_w_base    = ctrl_w_base_addr + k_tile_words(n_tile_idx, CHANNEL_TILE_NUM);
                current_sys_bias_base = ctrl_bias_base_addr + channel_tile_base(n_tile_idx);
            end
            PH_QKT: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base = 17'd0;
                current_sys_w_base   = qkt_k_base_word(head_idx, n_tile_idx);
            end
            PH_ATTN_V: begin
                current_sys_act_base = SHARED_A_BASE_WORD +
                                       act_word_index(token_tile_base(m_tile_idx), 0, SCORE_TILE_NUM);
                current_sys_w_base   = attnv_v_base_word(head_idx, n_tile_idx);
            end
            PH_OUT_PROJ: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base  = act_word_index(token_tile_base(m_tile_idx), 0, CHANNEL_TILE_NUM);
                current_sys_w_base    = ctrl_w_base_addr + k_tile_words(n_tile_idx, ctrl_k_tile_cnt);
                current_sys_bias_base = ctrl_bias_base_addr + channel_tile_base(n_tile_idx);
            end
            PH_FC1: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base  = act_word_index(token_tile_base(m_tile_idx), 0, CHANNEL_TILE_NUM);
                current_sys_w_base    = ctrl_w_base_addr + k_tile_words(n_tile_idx, ctrl_k_tile_cnt);
                current_sys_bias_base = ctrl_bias_base_addr + channel_tile_base(n_tile_idx);
            end
            PH_FC2: begin
                current_sys_act_zero_point = 8'd128;
                current_sys_act_base  = act_word_index(token_tile_base(m_tile_idx), 0, FFN_CHANNEL_TILE_NUM);
                current_sys_w_base    = ctrl_w_base_addr + k_tile_words(n_tile_idx, ctrl_k_tile_cnt);
                current_sys_bias_base = ctrl_bias_base_addr + channel_tile_base(n_tile_idx);
            end
            default: begin
            end
        endcase
    end

    always_comb begin
        gelu_fc1_tile_base = gelu_word_index(token_tile_base(m_tile_idx), channel_tile_base(n_tile_idx));
        gelu_fc1_tile_page_base = {gelu_fc1_tile_base[16:10], {GELU_PAGE_AW{1'b0}}};
        gelu_fc1_tile_store_after = (gelu_fc1_tile_base[9:0] == 10'd960) ||
                                    is_last_systolic_tile(PH_FC1, head_idx, m_tile_idx, n_tile_idx);
    end

    assign gelu_fc2_phase_active = (active_phase == PH_FC2) &&
                                   ((engine_state == ENG_SYS_START) ||
                                    (engine_state == ENG_SYS_WAIT_BUSY) ||
                                    (engine_state == ENG_SYS_WAIT_TILE));

    always_comb begin
        gelu_required_addr = current_sys_act_base;
        if ((active_phase == PH_FC2) && !sys_module_ready) begin
            gelu_required_addr = sys_act_bram_addr_raw;
        end
    end

    assign gelu_required_page_base = {gelu_required_addr[16:10], {GELU_PAGE_AW{1'b0}}};
    assign gelu_page_hit = gelu_page_valid_q && (gelu_page_base_q == gelu_required_page_base);
    assign gelu_page_load_req_valid = gelu_fc2_phase_active && !gelu_page_hit;
    assign gelu_page_load_req_base = {{(LOAD_ADDR_W-17){1'b0}}, gelu_required_page_base};
    assign gelu_page_store_req_valid = gelu_page_store_pending;
    assign gelu_page_store_req_base = {{(LOAD_ADDR_W-17){1'b0}}, gelu_page_store_base_q};
    assign gelu_page_wait_o = gelu_page_store_pending || (gelu_fc2_phase_active && !gelu_page_hit);

    assign shared_act_phase = (active_phase == PH_ATTN_V);

    always_comb begin
        case (active_phase)
            PH_QKV, PH_Q_TILE, PH_OUT_PROJ, PH_FC1: sys_act_bram_data = norm_rd_data;
            PH_QKT: sys_act_bram_data = q_cache_rd_data;
            PH_ATTN_V: sys_act_bram_data = shared_rd_data;
            PH_FC2: sys_act_bram_data = gelu_page_rd_data;
            default: sys_act_bram_data = 32'd0;
        endcase
    end

    always_comb begin
        if (use_external_weight)
            sys_w_bram_data = w_bram_data;
        else if (active_phase == PH_QKT)
            sys_w_bram_data = k_rd_data;
        else if (active_phase == PH_ATTN_V)
            sys_w_bram_data = v_xmid_rd_data;
        else
            sys_w_bram_data = 32'd0;
    end

    assign sys_bias_bram_data = use_external_weight ? bias_bram_data : 32'd0;
    assign sys_k_tile_cnt_eff = (active_phase == PH_Q_TILE) ? CHANNEL_TILE_NUM_8 : ctrl_k_tile_cnt;
    // PPU mode 2'b11 is the resource-saving pure requant path.
    // QKV/Q-tile/Attention*V do not need residual add or RMS statistics; OUT_PROJ
    // keeps ctrl mode 2'b00 so X_mid statistics are generated for RMSNorm2.
    assign ppu_mode_eff = ((active_phase == PH_QKV) ||
                           (active_phase == PH_Q_TILE) ||
                           (active_phase == PH_ATTN_V)) ? 2'b11 : ctrl_ppu_mode;
    assign ppu_scale_eff = (active_phase == PH_Q_TILE) ? Q_TILE_SCALE : ctrl_ppu_scale;
    assign sys_w_bram_valid = use_external_weight ?
                              (w_bram_valid && ((active_phase != PH_FC2) || gelu_page_hit)) :
                              1'b1;

    assign w_bram_rd_en    = use_external_weight ? sys_w_bram_rd_en    : 1'b0;
    assign bias_bram_rd_en = use_external_weight ? sys_bias_bram_rd_en : 1'b0;
    assign w_bram_addr     = use_external_weight ? sys_w_bram_addr_raw    : 17'd0;
    assign bias_bram_addr  = use_external_weight ? sys_bias_bram_addr_raw : 17'd0;

    always_comb begin
        shared_rd_en   = 1'b0;
        shared_rd_addr = 17'd0;

        if (shared_act_phase) begin
            shared_rd_en   = sys_act_bram_rd_en;
            shared_rd_addr = sys_act_bram_addr_raw;
        end
        else if ((engine_state == ENG_SOFT_LOAD) && (softmax_load_idx < TOKEN_NUM)) begin
            shared_rd_en   = 1'b1;
            shared_rd_addr = score_index(0, softmax_row_idx, softmax_load_idx);
        end
        else if (debug_read_allowed_i && (debug_rd_target_i == DBG_T_SHARED)) begin
            shared_rd_en   = 1'b1;
            shared_rd_addr = debug_rd_addr_i[16:0];
        end
    end

    always_comb begin
        shared_wr_en      = 1'b0;
        shared_wr_addr    = 17'd0;
        shared_wr_data    = 32'd0;
        shared_wr_byte_en = 4'b0000;
        shared_wr_r       = 0;
        shared_wr_c       = 0;
        shared_wr_token   = 0;
        shared_wr_channel = 0;
        shared_wr_word    = 0;
        shared_wr_byte    = 0;

        if ((active_phase == PH_QKT) && sys_opsum_fire) begin
            shared_wr_r       = psum_counter >> CHANNEL_TILE_LOG2;
            shared_wr_c       = psum_counter & (CHANNEL_TILE - 1);
            shared_wr_token   = token_tile_base(m_tile_idx) + shared_wr_r;
            shared_wr_channel = channel_tile_base(n_tile_idx) + shared_wr_c;
            if ((shared_wr_token < TOKEN_NUM) && (shared_wr_channel < TOKEN_NUM)) begin
                shared_wr_en      = 1'b1;
                shared_wr_addr    = score_index(0, shared_wr_token, shared_wr_channel);
                shared_wr_data    = sys_opsum;
                shared_wr_byte_en = 4'b1111;
            end
        end
        else if ((engine_state == ENG_SOFT_WAIT) && softmax_attention_valid) begin
            shared_wr_word = a_word_index(softmax_row_idx, softmax_attention_index);
            shared_wr_byte = softmax_attention_index & 3;
            shared_wr_en   = 1'b1;
            shared_wr_addr = shared_wr_word[16:0];
            shared_wr_data = 32'd0;
            shared_wr_data[shared_wr_byte*8 +: 8] =
                softmax_attention_data;
            shared_wr_byte_en = 4'b0001 << shared_wr_byte;
        end
    end

    always_comb begin
        gelu_page_wr_en      = gelu_loader_wr_en;
        gelu_page_wr_addr    = gelu_loader_wr_addr;
        gelu_page_wr_data    = gelu_loader_wr_data;
        gelu_page_wr_byte_en = gelu_loader_wr_strb;

        gelu_wr_r       = 0;
        gelu_wr_c       = 0;
        gelu_wr_token   = 0;
        gelu_wr_channel = 0;
        gelu_wr_word    = 0;

        if (!gelu_loader_wr_en && ppu_data_word_valid_int && (active_phase == PH_FC1)) begin
            gelu_wr_r       = ppu_data_word_idx_int >> TILE_ROW_WORD_LOG2;
            gelu_wr_c       = (ppu_data_word_idx_int & (TILE_ROW_WORDS - 1)) << 2;
            gelu_wr_token   = token_tile_base(shared_tile_m_idx) + gelu_wr_r;
            gelu_wr_channel = channel_tile_base(shared_tile_n_idx) + gelu_wr_c;
            gelu_wr_word    = gelu_word_index(gelu_wr_token, gelu_wr_channel);
            if ((gelu_wr_token < TOKEN_NUM) && (gelu_wr_channel < FFN_CHANNEL_NUM)) begin
                gelu_page_wr_en      = 1'b1;
                gelu_page_wr_addr    = gelu_wr_word[9:0];
                gelu_page_wr_data    = ppu_data_word_int;
                gelu_page_wr_byte_en = 4'b1111;
            end
        end
    end

    always_comb begin
        gelu_page_rd_en   = 1'b0;
        gelu_page_rd_addr = 10'd0;

        if (gelu_store_rd_en) begin
            gelu_page_rd_en   = 1'b1;
            gelu_page_rd_addr = gelu_store_rd_addr[9:0];
        end
        else if ((active_phase == PH_FC2) && sys_act_bram_rd_en) begin
            gelu_page_rd_en   = 1'b1;
            gelu_page_rd_addr = sys_act_bram_addr_raw[9:0];
        end
    end

    assign gelu_store_rd_data = gelu_page_rd_data;

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(1)) u_gelu_page_bram (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(gelu_page_rd_en),
        .rd_addr({7'd0, gelu_page_rd_addr}),
        .rd_data(gelu_page_rd_data),
        .wr_en(gelu_page_wr_en),
        .wr_addr({7'd0, gelu_page_wr_addr}),
        .wr_data(gelu_page_wr_data),
        .wr_byte_en(gelu_page_wr_byte_en)
    );

    ActivationMem #(.INIT_FILE("NONE"), .NUM_BANKS(SHARED_BANKS)) u_shared_intermediate_bram (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(shared_rd_en),
        .rd_addr(shared_rd_addr),
        .rd_data(shared_rd_data),
        .wr_en(shared_wr_en),
        .wr_addr(shared_wr_addr),
        .wr_data(shared_wr_data),
        .wr_byte_en(shared_wr_byte_en)
    );

    assign systolic_start_o           = sys_start_pulse;
    assign systolic_act_base_addr_o   = current_sys_act_base;
    assign systolic_w_base_addr_o     = current_sys_w_base;
    assign systolic_bias_base_addr_o  = current_sys_bias_base;
    assign systolic_k_tile_cnt_o      = sys_k_tile_cnt_eff;
    assign systolic_act_zero_point_o  = current_sys_act_zero_point;
    assign systolic_w_bram_valid_o    = sys_w_bram_valid;
    assign systolic_act_bram_data_o   = sys_act_bram_data;
    assign systolic_w_bram_data_o     = sys_w_bram_data;
    assign systolic_bias_bram_data_o  = sys_bias_bram_data;

    assign sys_module_ready       = systolic_module_ready_i;
    assign sys_act_bram_rd_en     = systolic_act_bram_rd_en_i;
    assign sys_w_bram_rd_en       = systolic_w_bram_rd_en_i;
    assign sys_bias_bram_rd_en    = systolic_bias_bram_rd_en_i;
    assign sys_act_bram_addr_raw  = systolic_act_bram_addr_i;
    assign sys_w_bram_addr_raw    = systolic_w_bram_addr_i;
    assign sys_bias_bram_addr_raw = systolic_bias_bram_addr_i;
    assign sys_opsum_valid        = systolic_opsum_valid_i;
    assign sys_opsum              = systolic_opsum_i;
    assign systolic_opsum_ready_o = (active_phase == PH_QKT) ? 1'b1 :
                                    (phase_uses_ppu(active_phase) ?
                                     (ppu_psum_ready && !res_load_active && !res_load_pending) :
                                     1'b1);
    assign sys_opsum_fire         = sys_opsum_valid && systolic_opsum_ready_o;

    logic [7:0] perf_bram_rd_words_int;
    logic [7:0] perf_bram_wr_words_int;

    always_comb begin
        perf_bram_rd_words_int = 8'd0;
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, x_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, norm_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, q_cache_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, k_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, v_xmid_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, shared_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, gelu_page_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, rms_fetch_issue};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, w_bram_rd_en};
        perf_bram_rd_words_int = perf_bram_rd_words_int + {7'd0, bias_bram_rd_en};

        perf_bram_wr_words_int = 8'd0;
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, x_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, norm_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, q_cache_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, k_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, v_xmid_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, shared_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, gelu_page_wr_en};
        perf_bram_wr_words_int = perf_bram_wr_words_int + {7'd0, token_stat_wr_en};
    end

    assign perf_bram_rd_words_o = perf_bram_rd_words_int;
    assign perf_bram_wr_words_o = perf_bram_wr_words_int;
    assign perf_bram_active_o   = |perf_bram_rd_words_int || |perf_bram_wr_words_int;
    assign perf_pingpong_wait_o = gelu_page_wait_o;
    assign perf_pingpong_load_o = gelu_page_load_req_valid;
    assign perf_pingpong_overlap_o = gelu_page_store_pending && busy_exec;
    // ------------------------------------------------------------
    // PPU, residual preload, and output tile serializers.
    // ------------------------------------------------------------
    logic [TOKEN_W-1:0] ppu_base_token_idx;
    logic [CHANNEL_TILE_W-1:0] ppu_channel_tile_idx;
    logic [TOKEN_TILE-1:0] ppu_token_valid_mask;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_residual_tile;

    assign ppu_data_tile_valid_o = ppu_data_valid_int;
    assign stat_valid_o          = ppu_stat_valid_int;
    assign stat_token_idx_o      = ppu_stat_token_idx_int;
    assign sum_sq_o              = ppu_sum_sq_int;

    assign ppu_base_token_idx   = token_tile_base(m_tile_idx);
    assign ppu_token_valid_mask = token_mask_for_mtile(m_tile_idx);
    assign ppu_channel_tile_idx =
        (active_phase == PH_QKV) ? (phase_dest_channel_base(PH_QKV, head_idx, n_tile_idx) >> CHANNEL_TILE_LOG2) :
        (active_phase == PH_Q_TILE) ? n_tile_idx[CHANNEL_TILE_W-1:0] :
        (active_phase == PH_ATTN_V) ? ((head_idx * HEAD_DIM_TILE_NUM) + n_tile_idx) :
                                      n_tile_idx[CHANNEL_TILE_W-1:0];
    assign ppu_residual_tile = ppu_residual_tile_reg;

    PPU #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ZERO_POINT(8'd128)
    ) u_ppu (
        .clk(clk),
        .rst_n(rst_n),
        .ppu_mode_i(ppu_mode_eff),
        .scaling_factor_i(ppu_scale_eff),
        .tile_valid_i(ppu_tile_valid),
        .tile_ready_o(ppu_tile_ready),
        .psum_valid_i(sys_opsum_fire && phase_uses_ppu(active_phase)),
        .psum_ready_o(ppu_psum_ready),
        .psum_i($signed(sys_opsum)),
        .residual_tile_i(ppu_residual_tile),
        .base_token_idx_i(ppu_base_token_idx),
        .channel_tile_idx_i(ppu_channel_tile_idx),
        .token_valid_mask_i(ppu_token_valid_mask),
        .data_tile_valid_o(ppu_data_valid_int),
        .data_tile_ready_i(1'b1),
        .data_tile_o(ppu_data_tile_o),
        .data_word_valid_o(ppu_data_word_valid_int),
        .data_word_ready_i(1'b1),
        .data_word_idx_o(ppu_data_word_idx_int),
        .data_word_o(ppu_data_word_int),
        .data_word_last_o(ppu_data_word_last_int),
        .stat_valid_o(ppu_stat_valid_int),
        .stat_ready_i(1'b1),
        .stat_token_idx_o(ppu_stat_token_idx_int),
        .sum_sq_o(ppu_sum_sq_int)
    );

    logic [9:0] inv_lut_addr;
    logic [15:0] inv_lut_data;
    logic [SUM_W-1:0] inv_lut_sum_sq;

    assign inv_lut_sum_sq = ((engine_state == ENG_RMS1_STAT_CAPTURE) &&
                             (rms1_stat_word_idx == (CHANNEL_WORD_NUM - 1))) ?
                            rms1_stat_sum_next : ppu_sum_sq_int;

    RMSInvSqrtAddrGen_A_Global #(
        .SUM_W(SUM_W)
    ) u_inv_addrgen (
        .sum_sq_i(inv_lut_sum_sq),
        .lut_addr_o(inv_lut_addr)
    );

    RMSInvSqrtLUT_A_Global_Case u_inv_lut (
        .addr_i(inv_lut_addr),
        .data_o(inv_lut_data)
    );

    integer res_token_int;
    integer res_channel_int;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_counter       <= 9'd0;
            ppu_tile_valid     <= 1'b0;
            shared_tile_store_after  <= 1'b0;
            shared_tile_m_idx        <= '0;
            shared_tile_n_idx        <= '0;
            gelu_page_base_q         <= '0;
            gelu_page_store_base_q   <= '0;
            gelu_page_valid_q        <= 1'b0;
            gelu_page_store_pending  <= 1'b0;
            act_tile_write_active    <= 1'b0;
            act_tile_write_kind      <= TW_NONE;
            act_tile_write_idx       <= '0;
            act_tile_write_m_idx     <= '0;
            act_tile_write_n_idx     <= '0;
            act_tile_write_head_idx  <= '0;
            res_load_active          <= 1'b0;
            res_load_pending         <= 1'b0;
            res_load_idx             <= '0;
            res_load_idx_d           <= '0;
            res_load_in_range_d      <= 1'b0;
            res_load_from_xmid       <= 1'b0;
            ppu_token_stat_wr_en <= 1'b0;
            ppu_token_stat_wr_addr <= '0;
            ppu_token_stat_wr_data <= '0;
            tile_done_pulse          <= 1'b0;
            out_proj_wait_armed      <= 1'b0;
            out_proj_wait_ppu_idle   <= 1'b0;
            out_proj_wait_stats      <= 1'b0;
            out_proj_stat_count      <= 5'd0;
            out_proj_stat_target     <= 5'd0;
            debug_tile_count         <= 16'd0;
        end
        else begin
            tile_done_pulse  <= 1'b0;
            ppu_token_stat_wr_en <= 1'b0;

            if (gelu_load_done_i) begin
                gelu_page_base_q  <= {gelu_load_base_i[16:10], {GELU_PAGE_AW{1'b0}}};
                gelu_page_valid_q <= 1'b1;
            end

            if (gelu_page_store_pending && gelu_store_done_i) begin
                gelu_page_store_pending <= 1'b0;
                tile_done_pulse         <= 1'b1;
                debug_tile_count        <= debug_tile_count + 16'd1;
            end

            if (out_proj_wait_ppu_idle && ppu_tile_ready) begin
                out_proj_wait_ppu_idle <= 1'b0;
            end

            if (out_proj_wait_armed && !out_proj_wait_ppu_idle && !out_proj_wait_stats) begin
                out_proj_wait_armed <= 1'b0;
                tile_done_pulse     <= 1'b1;
                debug_tile_count    <= debug_tile_count + 16'd1;
            end

            if (sys_start_pulse) begin
                psum_counter <= 9'd0;
                if (phase_uses_ppu(active_phase)) begin
                    ppu_residual_tile_reg <= {(TOKEN_TILE*CHANNEL_TILE){8'd128}};
                    ppu_tile_valid        <= 1'b1;
                    act_tile_write_m_idx    <= m_tile_idx;
                    act_tile_write_n_idx    <= n_tile_idx;
                    act_tile_write_head_idx <= head_idx;
                    shared_tile_m_idx       <= m_tile_idx;
                    shared_tile_n_idx       <= n_tile_idx;
                    shared_tile_store_after <= gelu_fc1_tile_store_after;

                    case (active_phase)
                        PH_QKV: begin
                            if (n_tile_idx < HEAD_DIM_TILE_NUM)
                                act_tile_write_kind <= TW_K;
                            else
                                act_tile_write_kind <= TW_V;
                        end
                        PH_Q_TILE:   act_tile_write_kind <= TW_QCACHE;
                        PH_ATTN_V:   act_tile_write_kind <= TW_O;
                        PH_OUT_PROJ: act_tile_write_kind <= TW_XMID;
                        PH_FC2:      act_tile_write_kind <= TW_XOUT;
                        default:     act_tile_write_kind <= TW_NONE;
                    endcase

                    if (active_phase == PH_FC1) begin
                        gelu_page_base_q  <= gelu_fc1_tile_page_base;
                        gelu_page_valid_q <= 1'b1;
                    end

                    if ((active_phase == PH_OUT_PROJ) || (active_phase == PH_FC2)) begin
                        res_load_active    <= 1'b1;
                        res_load_pending   <= 1'b0;
                        res_load_idx       <= 6'd0;
                        res_load_from_xmid <= (active_phase == PH_FC2);
                    end
                end
            end

            if (sys_opsum_fire) begin
                if (psum_counter == (OPS_PER_TILE - 1)) begin
                    psum_counter <= 9'd0;
                    if (active_phase == PH_QKT) begin
                        tile_done_pulse  <= 1'b1;
                        debug_tile_count <= debug_tile_count + 16'd1;
                    end
                end
                else begin
                    psum_counter <= psum_counter + 9'd1;
                end
            end

            if (res_load_pending) begin
                res_load_pending <= 1'b0;
                if (res_load_in_range_d) begin
                    if (res_load_from_xmid)
                        ppu_residual_tile_reg[res_load_idx_d*32 +: 32] <= v_xmid_rd_data;
                    else
                        ppu_residual_tile_reg[res_load_idx_d*32 +: 32] <= x_rd_data;
                end
                else begin
                    ppu_residual_tile_reg[res_load_idx_d*32 +: 32] <= 32'h80808080;
                end

                if (res_load_idx_d == (TILE_WORDS - 1)) begin
                    res_load_active <= 1'b0;
                end
            end
            else if (res_load_active) begin
                res_token_int   = token_tile_base(act_tile_write_m_idx) + (res_load_idx >> TILE_ROW_WORD_LOG2);
                res_channel_int = channel_tile_base(act_tile_write_n_idx) +
                                  ((res_load_idx & (TILE_ROW_WORDS - 1)) << 2);
                res_load_pending    <= 1'b1;
                res_load_idx_d      <= res_load_idx;
                res_load_in_range_d <= (res_token_int < TOKEN_NUM) && (res_channel_int < CHANNEL_NUM);
                res_load_idx        <= res_load_idx + 6'd1;
            end

            if (ppu_tile_valid && ppu_tile_ready)
                ppu_tile_valid <= 1'b0;

            if (ppu_data_word_valid_int &&
                (active_phase == PH_QKV) &&
                (act_tile_write_kind == TW_K)) begin
                act_tile_write_data[{ppu_data_word_idx_int, 5'b00000} +: 32] <= ppu_data_word_int;
            end

            if (act_tile_write_active) begin
                if (act_tile_write_idx == (TILE_WORDS - 1)) begin
                    act_tile_write_active <= 1'b0;
                    act_tile_write_kind   <= TW_NONE;
                    tile_done_pulse  <= 1'b1;
                    debug_tile_count <= debug_tile_count + 16'd1;
                end
                else begin
                    act_tile_write_idx <= act_tile_write_idx + 6'd1;
                end
            end

            if (ppu_data_valid_int) begin
                if (active_phase == PH_FC1) begin
                    if (shared_tile_store_after) begin
                        gelu_page_store_pending <= 1'b1;
                        gelu_page_store_base_q  <= gelu_page_base_q;
                    end
                    else begin
                        tile_done_pulse  <= 1'b1;
                        debug_tile_count <= debug_tile_count + 16'd1;
                    end
                end
                else if ((active_phase == PH_QKV) && (act_tile_write_kind == TW_K)) begin
                    act_tile_write_active <= 1'b1;
                    act_tile_write_idx    <= '0;
                end
                else begin
                    if (active_phase == PH_OUT_PROJ) begin
                        out_proj_wait_armed  <= 1'b1;
                        out_proj_wait_ppu_idle <= 1'b1;
                        // Output projection 的最後一個 channel tile 會由 PPU 產生 X_mid stat。
                        // PPU 只有在 data/stat 都送完後才會回 idle，所以 top 只需要等
                        // ppu_tile_ready；若另外用 stat pulse 計數，最後一個 tile 漏拍時
                        // 會卡在 phase 6。
                        out_proj_wait_stats  <= 1'b0;
                        out_proj_stat_count  <= 5'd0;
                        out_proj_stat_target <= token_count_for_mtile(m_tile_idx);
                    end
                    else begin
                        tile_done_pulse  <= 1'b1;
                        debug_tile_count <= debug_tile_count + 16'd1;
                    end
                end
            end

            if (ppu_stat_valid_int && (active_phase == PH_OUT_PROJ) && (ppu_stat_token_idx_int < TOKEN_NUM)) begin
                ppu_token_stat_wr_en <= 1'b1;
                ppu_token_stat_wr_addr <= ppu_stat_token_idx_int;
                ppu_token_stat_wr_data <= inv_lut_data;

                if (out_proj_wait_stats) begin
                    if ((out_proj_stat_count + 5'd1) >= out_proj_stat_target) begin
                        out_proj_wait_stats <= 1'b0;
                    end
                    out_proj_stat_count <= out_proj_stat_count + 5'd1;
                end
            end
        end
    end
    // ------------------------------------------------------------
    // Softmax row engine and A-buffer storage.
    // ------------------------------------------------------------
    Softmax_Unit #(
        .COLS(SOFTMAX_COLS),
        .EXP_LUT_HEX(SOFTMAX_EXP_LUT_HEX)
    ) u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .start(softmax_start_pulse),
        .score_load_valid(softmax_score_load_valid),
        .score_load_index(softmax_score_load_index),
        .score_load_data(softmax_score_load_data),
        .q_shift(6'd4),
        .k_shift(6'd4),
        .valid_cols(TOKEN_NUM_16),
        .attention_valid(softmax_attention_valid),
        .attention_index(softmax_attention_index),
        .attention_data(softmax_attention_data),
        .done(softmax_done)
    );

    // ------------------------------------------------------------
    // Phase engine.
    // ------------------------------------------------------------
    task automatic advance_systolic_indices;
        begin
            if (phase_uses_weight_stationary_order(active_phase)) begin
                if (m_tile_idx == (TOKEN_TILE_NUM - 1)) begin
                    m_tile_idx <= 5'd0;
                    if (n_tile_idx == phase_last_n_tile(active_phase))
                        n_tile_idx <= phase_first_n_tile7(active_phase);
                    else
                        n_tile_idx <= n_tile_idx + 7'd1;
                end
                else begin
                    m_tile_idx <= m_tile_idx + 5'd1;
                end
            end
            else begin
                if (n_tile_idx == phase_last_n_tile(active_phase)) begin
                    n_tile_idx <= phase_first_n_tile7(active_phase);
                    if (m_tile_idx == (TOKEN_TILE_NUM - 1))
                        m_tile_idx <= 5'd0;
                    else
                        m_tile_idx <= m_tile_idx + 5'd1;
                end
                else begin
                    n_tile_idx <= n_tile_idx + 7'd1;
                end
            end
        end
    endtask

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            engine_state         <= ENG_IDLE;
            active_phase         <= PH_IDLE;
            phase_done_to_ctrl   <= 1'b0;
            rms_start_pulse      <= 1'b0;
            sys_start_pulse      <= 1'b0;
            softmax_start_pulse  <= 1'b0;
            head_idx             <= 3'd0;
            mhsa_head_idx        <= 3'd0;
            m_tile_idx           <= 5'd0;
            n_tile_idx           <= 7'd0;
            softmax_head_idx     <= 3'd0;
            softmax_row_idx      <= '0;
            softmax_load_idx     <= '0;
            softmax_load_idx_d   <= '0;
            softmax_load_pending <= 1'b0;
            softmax_score_load_valid <= 1'b0;
            softmax_score_load_index <= '0;
            softmax_score_load_data  <= 32'sd0;
            debug_softmax_count  <= 16'd0;
            rms1_stat_token_idx <= '0;
            rms1_stat_word_idx  <= '0;
            rms1_stat_sum       <= '0;
            rms1_stat_wr_en     <= 1'b0;
            rms1_stat_wr_addr   <= '0;
            rms1_stat_wr_data   <= 16'd0;
        end
        else begin
            phase_done_to_ctrl  <= 1'b0;
            rms_start_pulse     <= 1'b0;
            sys_start_pulse     <= 1'b0;
            softmax_start_pulse <= 1'b0;
            softmax_score_load_valid <= 1'b0;
            rms1_stat_wr_en     <= 1'b0;

            case (engine_state)
                ENG_IDLE: begin
                    if (ctrl_phase_start) begin
                        active_phase <= ctrl_phase;
                        head_idx     <= (((ctrl_phase == PH_QKV) || (ctrl_phase == PH_QKT) ||
                                           (ctrl_phase == PH_SOFTMAX) || (ctrl_phase == PH_ATTN_V)) ?
                                          mhsa_head_idx : 3'd0);
                        m_tile_idx   <= 5'd0;
                        n_tile_idx   <= phase_first_n_tile7(ctrl_phase);

                        if (ctrl_phase == PH_RMS1) begin
                            engine_state <= ENG_RMS1_STAT_START;
                        end
                        else if (ctrl_phase == PH_RMS2) begin
                            engine_state <= ENG_RMS_START;
                        end
                        else if (ctrl_phase == PH_QKT) begin
                            // Q is tile-local: build the first 16x64 Q tile before QK^T.
                            active_phase <= PH_Q_TILE;
                            n_tile_idx   <= q_head_first_n_tile7(mhsa_head_idx);
                            engine_state <= ENG_QCACHE_START;
                        end
                        else if (phase_uses_systolic(ctrl_phase)) begin
                            engine_state <= ENG_SYS_START;
                        end
                        else if (ctrl_phase == PH_SOFTMAX) begin
                            softmax_head_idx     <= mhsa_head_idx;
                            softmax_row_idx      <= '0;
                            softmax_load_idx     <= '0;
                            softmax_load_idx_d   <= '0;
                            softmax_load_pending <= 1'b0;
                            engine_state         <= ENG_SOFT_LOAD;
                        end
                    end
                end

                ENG_RMS1_STAT_START: begin
                    rms1_stat_token_idx <= '0;
                    rms1_stat_word_idx  <= '0;
                    rms1_stat_sum       <= '0;
                    engine_state        <= ENG_RMS1_STAT_ISSUE;
                end

                ENG_RMS1_STAT_ISSUE: begin
                    // X BRAM read request.  Data returns in the next state.
                    engine_state <= ENG_RMS1_STAT_CAPTURE;
                end

                ENG_RMS1_STAT_CAPTURE: begin
                    if (rms1_stat_word_idx == (CHANNEL_WORD_NUM - 1)) begin
                        rms1_stat_wr_en   <= 1'b1;
                        rms1_stat_wr_addr <= rms1_stat_token_idx;
                        rms1_stat_wr_data <= inv_lut_data;
                        rms1_stat_sum     <= '0;
                        rms1_stat_word_idx <= '0;

                        if (rms1_stat_token_idx == (TOKEN_NUM - 1)) begin
                            rms1_stat_token_idx <= '0;
                            engine_state        <= ENG_RMS_START;
                        end
                        else begin
                            rms1_stat_token_idx <= rms1_stat_token_idx + 1'b1;
                            engine_state        <= ENG_RMS1_STAT_ISSUE;
                        end
                    end
                    else begin
                        rms1_stat_sum      <= rms1_stat_sum_next;
                        rms1_stat_word_idx <= rms1_stat_word_idx + 1'b1;
                        engine_state       <= ENG_RMS1_STAT_ISSUE;
                    end
                end

                ENG_RMS_START: begin
                    rms_start_pulse <= 1'b1;
                    engine_state    <= ENG_RMS_WAIT;
                end

                ENG_RMS_WAIT: begin
                    if (rms_done_seen && !rms_act_wr_valid &&
                        !rms_fetch_pending && !rms_fetch_valid) begin
                        phase_done_to_ctrl <= 1'b1;
                        engine_state       <= ENG_IDLE;
                    end
                end

                ENG_QCACHE_START: begin
                    if (sys_module_ready) begin
                        sys_start_pulse <= 1'b1;
                        engine_state    <= ENG_QCACHE_WAIT_BUSY;
                    end
                end

                ENG_QCACHE_WAIT_BUSY: begin
                    if (!sys_module_ready)
                        engine_state <= ENG_QCACHE_WAIT_TILE;
                end

                ENG_QCACHE_WAIT_TILE: begin
                    if (tile_done_pulse) begin
                        if (n_tile_idx == q_head_last_n_tile7(head_idx)) begin
                            active_phase <= PH_QKT;
                            n_tile_idx   <= 7'd0;
                            engine_state <= ENG_SYS_START;
                        end
                        else begin
                            n_tile_idx   <= n_tile_idx + 7'd1;
                            engine_state <= ENG_QCACHE_START;
                        end
                    end
                end

                ENG_SYS_START: begin
                    if (sys_module_ready) begin
                        sys_start_pulse <= 1'b1;
                        engine_state    <= ENG_SYS_WAIT_BUSY;
                    end
                end

                ENG_SYS_WAIT_BUSY: begin
                    if (!sys_module_ready)
                        engine_state <= ENG_SYS_WAIT_TILE;
                end

                ENG_SYS_WAIT_TILE: begin
                    if (tile_done_pulse) begin
                        if (active_phase == PH_QKT) begin
                            if (n_tile_idx == (SCORE_TILE_NUM - 1)) begin
                                if (m_tile_idx == (TOKEN_TILE_NUM - 1)) begin
                                    phase_done_to_ctrl <= 1'b1;
                                    engine_state       <= ENG_IDLE;
                                end
                                else begin
                                    m_tile_idx   <= m_tile_idx + 5'd1;
                                    n_tile_idx   <= q_head_first_n_tile7(head_idx);
                                    active_phase <= PH_Q_TILE;
                                    engine_state <= ENG_QCACHE_START;
                                end
                            end
                            else begin
                                n_tile_idx   <= n_tile_idx + 7'd1;
                                engine_state <= ENG_SYS_START;
                            end
                        end
                        else if (is_last_systolic_tile(active_phase, head_idx, m_tile_idx, n_tile_idx)) begin
                            if ((active_phase == PH_ATTN_V) && (mhsa_head_idx != (HEAD_NUM - 1)))
                                mhsa_head_idx <= mhsa_head_idx + 3'd1;
                            phase_done_to_ctrl <= 1'b1;
                            engine_state       <= ENG_IDLE;
                        end
                        else begin
                            advance_systolic_indices();
                            engine_state <= ENG_SYS_START;
                        end
                    end
                end

                ENG_SOFT_LOAD: begin
                    if (softmax_load_pending) begin
                        // 超過 TOKEN_NUM 的 padded column 由 softmax_mask 擋掉，不需要寫 0。
                        // 避免 variable-index register array 被 Vivado 推成大量 set/reset mux。
                        softmax_score_load_valid <= 1'b1;
                        softmax_score_load_index <= softmax_load_idx_d[SOFTMAX_IDX_W-1:0];
                        softmax_score_load_data  <= $signed(shared_rd_data);
                    end

                    if (softmax_load_idx < TOKEN_NUM) begin
                        softmax_load_idx_d   <= softmax_load_idx;
                        softmax_load_pending <= 1'b1;
                        softmax_load_idx     <= softmax_load_idx + 9'd1;
                    end
                    else if (softmax_load_pending) begin
                        softmax_load_pending <= 1'b0;
                        engine_state         <= ENG_SOFT_START;
                    end
                end

                ENG_SOFT_START: begin
                    softmax_start_pulse <= 1'b1;
                    engine_state        <= ENG_SOFT_WAIT;
                end

                ENG_SOFT_WAIT: begin
                    if (softmax_done) begin
                        debug_softmax_count <= debug_softmax_count + 16'd1;
                        if (softmax_row_idx == (TOKEN_NUM - 1)) begin
                            phase_done_to_ctrl <= 1'b1;
                            engine_state       <= ENG_IDLE;
                        end
                        else begin
                            softmax_row_idx      <= softmax_row_idx + 1'b1;
                            softmax_load_idx     <= '0;
                            softmax_load_idx_d   <= '0;
                            softmax_load_pending <= 1'b0;
                            engine_state         <= ENG_SOFT_LOAD;
                        end
                    end
                end

                default: begin
                    engine_state <= ENG_IDLE;
                end
            endcase
        end
    end

endmodule
