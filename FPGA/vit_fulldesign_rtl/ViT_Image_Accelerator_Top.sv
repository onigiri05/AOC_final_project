`timescale 1ns/1ps

// ============================================================
// Module: ViT_Image_Accelerator_Top
//
// 中文說明：
//   這是 raw image 版本的最外層 wrapper。
//   它先啟動 Patch_Embedding_Systolic_Top，將整張圖轉成 X Buffer token；
//   patch embedding 完成後，再啟動原本的 ViT_Accelerator_Top。
//
// FPGA bring-up 建議：
//   先接這個 wrapper 做整張圖路徑測試。
//   若只想跳過 patch embedding，仍可直接使用 ViT_Accelerator_Top。
// ============================================================
module ViT_Image_Accelerator_Top #(
    parameter int IMG_H           = 224,
    parameter int IMG_W           = 224,
    parameter int IMG_C           = 3,
    parameter int PATCH_SIZE      = 16,
    parameter int EMBED_DIM       = 384,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int TOKEN_TILE      = 8,
    parameter int CHANNEL_TILE    = 8,
    parameter int CHANNEL_TILE_W  = 6,
    parameter int ADDR_W          = 17,
    parameter int LOAD_ADDR_W     = 20,
    parameter int HEAD_NUM        = 6,
    parameter int HEAD_DIM        = 64,
    parameter int FFN_CHANNEL_NUM = 1536,
    parameter int SOFTMAX_COLS    = 208,
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
    input  logic clk,
    input  logic rst_n,

    input  logic start_exec,
    input  logic transformer_only_i,
    input  logic [5:0] patch_requant_shift_i,
    output logic busy_exec,
    output logic done_exec,
    output logic [1:0] debug_top_state,

    input  logic                 checkpoint_enable_i,
    input  logic                 checkpoint_resume_i,
    output logic                 checkpoint_pending_o,
    output logic                 stage_done_pulse_o,
    output logic [3:0]           stage_id_o,
    output logic [4:0]           stage_phase_o,
    input  logic [3:0]           debug_rd_target_i,
    input  logic [ADDR_W-1:0]    debug_rd_addr_i,
    output logic [31:0]          debug_rd_data_o,

    // ------------------------------------------------------------
    // Raw image / patch embedding memories.
    // ------------------------------------------------------------
    output logic [IMG_ADDR_W-1:0]          image_rd_addr,
    input  logic [DATA_W-1:0]              image_rd_data,
    input  logic                           image_rd_valid,

    output logic                           patch_w_rd_en,
    output logic [PE_W_ADDR_W-1:0]         patch_w_addr,
    input  logic [31:0]                    patch_w_data,
    input  logic                           patch_w_valid,

    output logic                           patch_bias_rd_en,
    output logic [16:0]                    patch_bias_addr,
    input  logic [31:0]                    patch_bias_data,

    output logic [PE_POS_ADDR_W-1:0]       patch_pos_addr,
    input  logic [DATA_W-1:0]              patch_pos_data,
    input  logic                           patch_pos_valid,

    output logic [PE_EMBED_ADDR_W-1:0]     cls_addr,
    input  logic [DATA_W-1:0]              cls_data,
    input  logic                           cls_valid,

    // ------------------------------------------------------------
    // External X loader.
    // Python uses this path to feed the previous block's X_out back into the
    // shared X buffer, then starts this top in transformer-only mode.
    // ------------------------------------------------------------
    input  logic                           x_loader_word_we,
    input  logic [16:0]                    x_loader_word_addr,
    input  logic [31:0]                    x_loader_word_data,
    input  logic [3:0]                     x_loader_word_byte_en,

    // ------------------------------------------------------------
    // Transformer block external memories.
    // ------------------------------------------------------------
    output logic                 w_bram_rd_en,
    output logic                 bias_bram_rd_en,
    output logic [16:0]          w_bram_addr,
    output logic [16:0]          bias_bram_addr,
    input  logic                 w_bram_valid,
    input  logic [31:0]          w_bram_data,
    input  logic [31:0]          bias_bram_data,

    output logic                 rms_norm_sel_o,
    output logic [8:0]           gamma_addr,
    input  logic signed [15:0]   gamma_data,

    input  logic [ADDR_W-1:0]    x_out_raddr,
    output logic [DATA_W-1:0]    x_out_rdata,

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

    output logic [4:0]           debug_phase,
    output logic [15:0]          debug_tile_count,
    output logic [15:0]          debug_softmax_count,

    output logic                 ppu_data_tile_valid_o,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_data_tile_o,
    output logic                 stat_valid_o,
    output logic [TOKEN_W-1:0]   stat_token_idx_o,
    output logic [SUM_W-1:0]     sum_sq_o,

    output logic                 patch_busy_o,
    output logic                 patch_done_o,
    output logic [2:0]           patch_debug_state_o,
    output logic [15:0]          patch_debug_patch_idx_o,
    output logic [15:0]          patch_debug_channel_idx_o,
    output logic [15:0]          patch_debug_elem_idx_o,

    output logic [7:0]           perf_bram_rd_words_o,
    output logic [7:0]           perf_bram_wr_words_o,
    output logic                 perf_bram_active_o,
    output logic [15:0]          perf_mac_ops_o,
    output logic                 perf_pingpong_wait_o,
    output logic                 perf_pingpong_load_o,
    output logic                 perf_pingpong_overlap_o
);

    typedef enum logic [1:0] {
        TOP_IDLE,
        TOP_PATCH,
        TOP_VIT_START,
        TOP_VIT_WAIT
    } top_state_t;

    top_state_t top_state_q;

    logic patch_start;
    logic patch_done;
    logic patch_x_we;
    logic [ADDR_W-1:0] patch_x_addr;
    logic [DATA_W-1:0] patch_x_wdata;
    logic patch_x_word_we;
    logic [16:0] patch_x_word_addr;
    logic [31:0] patch_x_word_data;
    logic [3:0] patch_x_word_byte_en;
    logic vit_x_word_we;
    logic [16:0] vit_x_word_addr;
    logic [31:0] vit_x_word_data;
    logic [3:0] vit_x_word_byte_en;

    logic vit_start;
    logic vit_busy;
    logic vit_done;
    logic patch_checkpoint_pending_q;
    logic vit_checkpoint_pending;
    logic vit_stage_done_pulse;
    logic [3:0] vit_stage_id;
    logic [4:0] vit_stage_phase;

    logic shared_sys_patch_owner;
    logic shared_sys_start;
    logic shared_sys_module_ready;
    logic [16:0] shared_sys_act_base_addr;
    logic [16:0] shared_sys_w_base_addr;
    logic [16:0] shared_sys_bias_base_addr;
    logic [7:0]  shared_sys_k_tile_cnt;
    logic [7:0]  shared_sys_act_zero_point;
    logic shared_sys_act_bram_rd_en;
    logic shared_sys_w_bram_rd_en;
    logic shared_sys_bias_bram_rd_en;
    logic [16:0] shared_sys_act_bram_addr;
    logic [16:0] shared_sys_w_bram_addr;
    logic [16:0] shared_sys_bias_bram_addr;
    logic shared_sys_w_bram_valid;
    logic [31:0] shared_sys_act_bram_data;
    logic [31:0] shared_sys_w_bram_data;
    logic [31:0] shared_sys_bias_bram_data;
    logic shared_sys_opsum_ready;
    logic shared_sys_opsum_valid;
    logic [31:0] shared_sys_opsum;
    logic shared_opsum_fire;
    logic shared_sys_start_q;
    logic [16:0] shared_sys_act_base_addr_q;
    logic [16:0] shared_sys_w_base_addr_q;
    logic [16:0] shared_sys_bias_base_addr_q;
    logic [7:0]  shared_sys_k_tile_cnt_q;
    logic [7:0]  shared_sys_act_zero_point_q;
    logic [7:0] vit_perf_bram_rd_words;
    logic [7:0] vit_perf_bram_wr_words;
    logic vit_perf_bram_active;
    logic vit_perf_pingpong_wait;
    logic vit_perf_pingpong_load;
    logic vit_perf_pingpong_overlap;
    logic [7:0] patch_perf_bram_rd_words;
    logic [7:0] patch_perf_bram_wr_words;

    logic patch_sys_start;
    logic patch_sys_module_ready;
    logic [16:0] patch_sys_act_base_addr;
    logic [16:0] patch_sys_w_base_addr;
    logic [16:0] patch_sys_bias_base_addr;
    logic [7:0]  patch_sys_k_tile_cnt;
    logic [7:0]  patch_sys_act_zero_point;
    logic patch_sys_act_bram_rd_en;
    logic patch_sys_w_bram_rd_en;
    logic patch_sys_bias_bram_rd_en;
    logic [16:0] patch_sys_act_bram_addr;
    logic [16:0] patch_sys_w_bram_addr;
    logic [16:0] patch_sys_bias_bram_addr;
    logic patch_sys_w_bram_valid;
    logic [31:0] patch_sys_act_bram_data;
    logic [31:0] patch_sys_w_bram_data;
    logic [31:0] patch_sys_bias_bram_data;
    logic patch_sys_opsum_ready;
    logic patch_sys_opsum_valid;
    logic [31:0] patch_sys_opsum;

    logic vit_sys_start;
    logic vit_sys_module_ready;
    logic [16:0] vit_sys_act_base_addr;
    logic [16:0] vit_sys_w_base_addr;
    logic [16:0] vit_sys_bias_base_addr;
    logic [7:0]  vit_sys_k_tile_cnt;
    logic [7:0]  vit_sys_act_zero_point;
    logic vit_sys_act_bram_rd_en;
    logic vit_sys_w_bram_rd_en;
    logic vit_sys_bias_bram_rd_en;
    logic [16:0] vit_sys_act_bram_addr;
    logic [16:0] vit_sys_w_bram_addr;
    logic [16:0] vit_sys_bias_bram_addr;
    logic vit_sys_w_bram_valid;
    logic [31:0] vit_sys_act_bram_data;
    logic [31:0] vit_sys_w_bram_data;
    logic [31:0] vit_sys_bias_bram_data;
    logic vit_sys_opsum_ready;
    logic vit_sys_opsum_valid;
    logic [31:0] vit_sys_opsum;

    assign debug_top_state = top_state_q;
    assign busy_exec = (top_state_q != TOP_IDLE);
    assign done_exec = vit_done && (top_state_q == TOP_VIT_WAIT);
    assign patch_done_o = patch_done;
    assign shared_sys_patch_owner = (top_state_q == TOP_PATCH);
    assign checkpoint_pending_o = patch_checkpoint_pending_q || vit_checkpoint_pending;
    assign stage_done_pulse_o = (patch_done &&
                                 (top_state_q == TOP_PATCH) &&
                                 !patch_checkpoint_pending_q) ?
                                1'b1 : vit_stage_done_pulse;
    assign stage_id_o = patch_checkpoint_pending_q ? 4'd1 :
                        (((patch_done) &&
                          (top_state_q == TOP_PATCH) &&
                          !patch_checkpoint_pending_q) ? 4'd1 : vit_stage_id);
    assign stage_phase_o = patch_checkpoint_pending_q ? 5'd0 :
                           (((patch_done) &&
                             (top_state_q == TOP_PATCH) &&
                             !patch_checkpoint_pending_q) ? 5'd0 : vit_stage_phase);

    Patch_Embedding_Systolic_Top #(
        .IMG_H(IMG_H),
        .IMG_W(IMG_W),
        .IMG_C(IMG_C),
        .PATCH_SIZE(PATCH_SIZE),
        .EMBED_DIM(EMBED_DIM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .ACC_W(SUM_W),
        .PATCH_GRID_H(PATCH_GRID_H),
        .PATCH_GRID_W(PATCH_GRID_W),
        .PATCH_COUNT(PATCH_COUNT),
        .TOKEN_NUM(TOKEN_NUM),
        .PATCH_ELEMS(PATCH_ELEMS),
        .IMG_ADDR_W(IMG_ADDR_W),
        .POS_ADDR_W(PE_POS_ADDR_W),
        .EMBED_ADDR_W(PE_EMBED_ADDR_W),
        .ADDR_W(ADDR_W)
    ) u_patch_embedding (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(patch_start),
        .requant_shift_i(patch_requant_shift_i),
        .busy_o(patch_busy_o),
        .done_o(patch_done),
        .image_rd_addr_o(image_rd_addr),
        .image_rd_data_i(image_rd_data),
        .image_rd_valid_i(image_rd_valid),
        .patch_w_rd_en_o(patch_w_rd_en),
        .patch_w_addr_o(patch_w_addr),
        .patch_w_data_i(patch_w_data),
        .patch_w_valid_i(patch_w_valid),
        .patch_bias_rd_en_o(patch_bias_rd_en),
        .patch_bias_addr_o(patch_bias_addr),
        .patch_bias_data_i(patch_bias_data),
        .pos_addr_o(patch_pos_addr),
        .pos_data_i(patch_pos_data),
        .pos_valid_i(patch_pos_valid),
        .cls_addr_o(cls_addr),
        .cls_data_i(cls_data),
        .cls_valid_i(cls_valid),
        .x_buf_we_o(patch_x_we),
        .x_buf_addr_o(patch_x_addr),
        .x_buf_wdata_o(patch_x_wdata),
        .x_buf_word_we_o(patch_x_word_we),
        .x_buf_word_addr_o(patch_x_word_addr),
        .x_buf_word_data_o(patch_x_word_data),
        .x_buf_word_byte_en_o(patch_x_word_byte_en),
        .debug_state_o(patch_debug_state_o),
        .debug_patch_idx_o(patch_debug_patch_idx_o),
        .debug_channel_idx_o(patch_debug_channel_idx_o),
        .debug_elem_idx_o(patch_debug_elem_idx_o),
        .systolic_start_o(patch_sys_start),
        .systolic_module_ready_i(patch_sys_module_ready),
        .systolic_act_base_addr_o(patch_sys_act_base_addr),
        .systolic_w_base_addr_o(patch_sys_w_base_addr),
        .systolic_bias_base_addr_o(patch_sys_bias_base_addr),
        .systolic_k_tile_cnt_o(patch_sys_k_tile_cnt),
        .systolic_act_zero_point_o(patch_sys_act_zero_point),
        .systolic_act_bram_rd_en_i(patch_sys_act_bram_rd_en),
        .systolic_w_bram_rd_en_i(patch_sys_w_bram_rd_en),
        .systolic_bias_bram_rd_en_i(patch_sys_bias_bram_rd_en),
        .systolic_act_bram_addr_i(patch_sys_act_bram_addr),
        .systolic_w_bram_addr_i(patch_sys_w_bram_addr),
        .systolic_bias_bram_addr_i(patch_sys_bias_bram_addr),
        .systolic_w_bram_valid_o(patch_sys_w_bram_valid),
        .systolic_act_bram_data_o(patch_sys_act_bram_data),
        .systolic_w_bram_data_o(patch_sys_w_bram_data),
        .systolic_bias_bram_data_o(patch_sys_bias_bram_data),
        .systolic_opsum_ready_o(patch_sys_opsum_ready),
        .systolic_opsum_valid_i(patch_sys_opsum_valid),
        .systolic_opsum_i(patch_sys_opsum)
    );

    ViT_Accelerator_Top #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(EMBED_DIM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ADDR_W(ADDR_W),
        .LOAD_ADDR_W(LOAD_ADDR_W),
        .HEAD_NUM(HEAD_NUM),
        .HEAD_DIM(HEAD_DIM),
        .FFN_CHANNEL_NUM(FFN_CHANNEL_NUM),
        .SOFTMAX_COLS(SOFTMAX_COLS),
        .SOFTMAX_EXP_LUT_HEX(SOFTMAX_EXP_LUT_HEX)
    ) u_vit (
        .clk(clk),
        .rst_n(rst_n),
        .start_exec(vit_start),
        .busy_exec(vit_busy),
        .done_exec(vit_done),
        .debug_phase(debug_phase),
        .checkpoint_enable_i(checkpoint_enable_i),
        .checkpoint_resume_i(checkpoint_resume_i && !patch_checkpoint_pending_q),
        .checkpoint_pending_o(vit_checkpoint_pending),
        .stage_done_pulse_o(vit_stage_done_pulse),
        .stage_id_o(vit_stage_id),
        .stage_phase_o(vit_stage_phase),
        .debug_read_allowed_i((top_state_q == TOP_IDLE) ||
                              patch_checkpoint_pending_q ||
                              vit_checkpoint_pending),
        .debug_rd_target_i(debug_rd_target_i),
        .debug_rd_addr_i(debug_rd_addr_i),
        .debug_rd_data_o(debug_rd_data_o),
        .x_buf_we(patch_x_we),
        .x_buf_addr(patch_x_addr),
        .x_buf_wdata(patch_x_wdata),
        .x_buf_word_we(vit_x_word_we),
        .x_buf_word_addr(vit_x_word_addr),
        .x_buf_word_data(vit_x_word_data),
        .x_buf_word_byte_en(vit_x_word_byte_en),
        .x_out_raddr(x_out_raddr),
        .x_out_rdata(x_out_rdata),
        .gelu_loader_wr_en(gelu_loader_wr_en),
        .gelu_loader_wr_addr(gelu_loader_wr_addr),
        .gelu_loader_wr_data(gelu_loader_wr_data),
        .gelu_loader_wr_strb(gelu_loader_wr_strb),
        .gelu_load_done_i(gelu_load_done_i),
        .gelu_load_base_i(gelu_load_base_i),
        .gelu_store_done_i(gelu_store_done_i),
        .gelu_store_rd_en(gelu_store_rd_en),
        .gelu_store_rd_addr(gelu_store_rd_addr),
        .gelu_store_rd_data(gelu_store_rd_data),
        .gelu_page_load_req_valid(gelu_page_load_req_valid),
        .gelu_page_load_req_base(gelu_page_load_req_base),
        .gelu_page_store_req_valid(gelu_page_store_req_valid),
        .gelu_page_store_req_base(gelu_page_store_req_base),
        .gelu_page_wait_o(gelu_page_wait_o),
        .w_bram_rd_en(w_bram_rd_en),
        .bias_bram_rd_en(bias_bram_rd_en),
        .w_bram_addr(w_bram_addr),
        .bias_bram_addr(bias_bram_addr),
        .w_bram_valid(w_bram_valid),
        .w_bram_data(w_bram_data),
        .bias_bram_data(bias_bram_data),
        .rms_norm_sel_o(rms_norm_sel_o),
        .gamma_addr(gamma_addr),
        .gamma_data(gamma_data),
        .ppu_data_tile_valid_o(ppu_data_tile_valid_o),
        .ppu_data_tile_o(ppu_data_tile_o),
        .stat_valid_o(stat_valid_o),
        .stat_token_idx_o(stat_token_idx_o),
        .sum_sq_o(sum_sq_o),
        .debug_tile_count(debug_tile_count),
        .debug_softmax_count(debug_softmax_count),
        .perf_bram_rd_words_o(vit_perf_bram_rd_words),
        .perf_bram_wr_words_o(vit_perf_bram_wr_words),
        .perf_bram_active_o(vit_perf_bram_active),
        .perf_pingpong_wait_o(vit_perf_pingpong_wait),
        .perf_pingpong_load_o(vit_perf_pingpong_load),
        .perf_pingpong_overlap_o(vit_perf_pingpong_overlap),
        .systolic_start_o(vit_sys_start),
        .systolic_module_ready_i(vit_sys_module_ready),
        .systolic_act_base_addr_o(vit_sys_act_base_addr),
        .systolic_w_base_addr_o(vit_sys_w_base_addr),
        .systolic_bias_base_addr_o(vit_sys_bias_base_addr),
        .systolic_k_tile_cnt_o(vit_sys_k_tile_cnt),
        .systolic_act_zero_point_o(vit_sys_act_zero_point),
        .systolic_act_bram_rd_en_i(vit_sys_act_bram_rd_en),
        .systolic_w_bram_rd_en_i(vit_sys_w_bram_rd_en),
        .systolic_bias_bram_rd_en_i(vit_sys_bias_bram_rd_en),
        .systolic_act_bram_addr_i(vit_sys_act_bram_addr),
        .systolic_w_bram_addr_i(vit_sys_w_bram_addr),
        .systolic_bias_bram_addr_i(vit_sys_bias_bram_addr),
        .systolic_w_bram_valid_o(vit_sys_w_bram_valid),
        .systolic_act_bram_data_o(vit_sys_act_bram_data),
        .systolic_w_bram_data_o(vit_sys_w_bram_data),
        .systolic_bias_bram_data_o(vit_sys_bias_bram_data),
        .systolic_opsum_ready_o(vit_sys_opsum_ready),
        .systolic_opsum_valid_i(vit_sys_opsum_valid),
        .systolic_opsum_i(vit_sys_opsum)
    );

    assign shared_opsum_fire = shared_sys_opsum_valid && shared_sys_opsum_ready;

    always_comb begin
        patch_perf_bram_rd_words = 8'd0;
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, image_rd_valid};
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, patch_w_rd_en};
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, patch_bias_rd_en};
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, patch_pos_valid};
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, cls_valid};
        patch_perf_bram_rd_words = patch_perf_bram_rd_words + {7'd0, patch_sys_act_bram_rd_en};

        patch_perf_bram_wr_words = 8'd0;
        patch_perf_bram_wr_words = patch_perf_bram_wr_words + {7'd0, patch_x_word_we};
        patch_perf_bram_wr_words = patch_perf_bram_wr_words + {7'd0, x_loader_word_we};
    end

    always_comb begin
        if (top_state_q == TOP_PATCH) begin
            vit_x_word_we      = patch_x_word_we;
            vit_x_word_addr    = patch_x_word_addr;
            vit_x_word_data    = patch_x_word_data;
            vit_x_word_byte_en = patch_x_word_byte_en;
        end
        else begin
            vit_x_word_we      = x_loader_word_we;
            vit_x_word_addr    = x_loader_word_addr;
            vit_x_word_data    = x_loader_word_data;
            vit_x_word_byte_en = x_loader_word_byte_en;
        end
    end

    assign perf_bram_rd_words_o = vit_perf_bram_rd_words + patch_perf_bram_rd_words;
    assign perf_bram_wr_words_o = vit_perf_bram_wr_words + patch_perf_bram_wr_words;
    assign perf_bram_active_o   = vit_perf_bram_active ||
                                  |patch_perf_bram_rd_words ||
                                  |patch_perf_bram_wr_words;
    assign perf_mac_ops_o = shared_opsum_fire ?
                            ({8'd0, shared_sys_k_tile_cnt_q} << 3) :
                            16'd0;
    assign perf_pingpong_wait_o    = vit_perf_pingpong_wait;
    assign perf_pingpong_load_o    = vit_perf_pingpong_load;
    assign perf_pingpong_overlap_o = vit_perf_pingpong_overlap;

    assign shared_sys_start          = shared_sys_patch_owner ? patch_sys_start          : vit_sys_start;
    assign shared_sys_act_base_addr  = shared_sys_patch_owner ? patch_sys_act_base_addr  : vit_sys_act_base_addr;
    assign shared_sys_w_base_addr    = shared_sys_patch_owner ? patch_sys_w_base_addr    : vit_sys_w_base_addr;
    assign shared_sys_bias_base_addr = shared_sys_patch_owner ? patch_sys_bias_base_addr : vit_sys_bias_base_addr;
    assign shared_sys_k_tile_cnt     = shared_sys_patch_owner ? patch_sys_k_tile_cnt     : vit_sys_k_tile_cnt;
    assign shared_sys_act_zero_point = shared_sys_patch_owner ? patch_sys_act_zero_point : vit_sys_act_zero_point;
    assign shared_sys_w_bram_valid   = shared_sys_patch_owner ? patch_sys_w_bram_valid   : vit_sys_w_bram_valid;
    assign shared_sys_act_bram_data  = shared_sys_patch_owner ? patch_sys_act_bram_data  : vit_sys_act_bram_data;
    assign shared_sys_w_bram_data    = shared_sys_patch_owner ? patch_sys_w_bram_data    : vit_sys_w_bram_data;
    assign shared_sys_bias_bram_data = shared_sys_patch_owner ? patch_sys_bias_bram_data : vit_sys_bias_bram_data;
    assign shared_sys_opsum_ready    = shared_sys_patch_owner ? patch_sys_opsum_ready : vit_sys_opsum_ready;

    // Pipeline the launch/config path into the single shared systolic array.
    // The owners already wait for module_ready to fall after issuing start, so
    // this one-cycle delayed launch does not change the tile ordering.  It does
    // cut the counter/function/mux path before Systolic captures its base
    // addresses and k-tile count.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            shared_sys_start_q          <= 1'b0;
            shared_sys_act_base_addr_q  <= 17'd0;
            shared_sys_w_base_addr_q    <= 17'd0;
            shared_sys_bias_base_addr_q <= 17'd0;
            shared_sys_k_tile_cnt_q     <= 8'd0;
            shared_sys_act_zero_point_q <= 8'd0;
        end
        else begin
            shared_sys_start_q <= shared_sys_start;
            if (shared_sys_start) begin
                shared_sys_act_base_addr_q  <= shared_sys_act_base_addr;
                shared_sys_w_base_addr_q    <= shared_sys_w_base_addr;
                shared_sys_bias_base_addr_q <= shared_sys_bias_base_addr;
                shared_sys_k_tile_cnt_q     <= shared_sys_k_tile_cnt;
                shared_sys_act_zero_point_q <= shared_sys_act_zero_point;
            end
        end
    end

    assign patch_sys_module_ready    = shared_sys_patch_owner ? shared_sys_module_ready : 1'b0;
    assign patch_sys_act_bram_rd_en  = shared_sys_patch_owner ? shared_sys_act_bram_rd_en : 1'b0;
    assign patch_sys_w_bram_rd_en    = shared_sys_patch_owner ? shared_sys_w_bram_rd_en : 1'b0;
    assign patch_sys_bias_bram_rd_en = shared_sys_patch_owner ? shared_sys_bias_bram_rd_en : 1'b0;
    assign patch_sys_act_bram_addr   = shared_sys_patch_owner ? shared_sys_act_bram_addr : 17'd0;
    assign patch_sys_w_bram_addr     = shared_sys_patch_owner ? shared_sys_w_bram_addr : 17'd0;
    assign patch_sys_bias_bram_addr  = shared_sys_patch_owner ? shared_sys_bias_bram_addr : 17'd0;
    assign patch_sys_opsum_valid     = shared_sys_patch_owner ? shared_sys_opsum_valid : 1'b0;
    assign patch_sys_opsum           = shared_sys_patch_owner ? shared_sys_opsum : 32'd0;

    assign vit_sys_module_ready      = shared_sys_patch_owner ? 1'b0 : shared_sys_module_ready;
    assign vit_sys_act_bram_rd_en    = shared_sys_patch_owner ? 1'b0 : shared_sys_act_bram_rd_en;
    assign vit_sys_w_bram_rd_en      = shared_sys_patch_owner ? 1'b0 : shared_sys_w_bram_rd_en;
    assign vit_sys_bias_bram_rd_en   = shared_sys_patch_owner ? 1'b0 : shared_sys_bias_bram_rd_en;
    assign vit_sys_act_bram_addr     = shared_sys_patch_owner ? 17'd0 : shared_sys_act_bram_addr;
    assign vit_sys_w_bram_addr       = shared_sys_patch_owner ? 17'd0 : shared_sys_w_bram_addr;
    assign vit_sys_bias_bram_addr    = shared_sys_patch_owner ? 17'd0 : shared_sys_bias_bram_addr;
    assign vit_sys_opsum_valid       = shared_sys_patch_owner ? 1'b0 : shared_sys_opsum_valid;
    assign vit_sys_opsum             = shared_sys_patch_owner ? 32'd0 : shared_sys_opsum;

    Systolic u_shared_systolic (
        .clk(clk),
        .rst_n(rst_n),
        .en(shared_sys_start_q),
        .module_ready(shared_sys_module_ready),
        .act_base_addr(shared_sys_act_base_addr_q),
        .w_base_addr(shared_sys_w_base_addr_q),
        .bias_base_addr(shared_sys_bias_base_addr_q),
        .k_tile_cnt(shared_sys_k_tile_cnt_q),
        .act_zero_point(shared_sys_act_zero_point_q),
        .act_bram_rd_en(shared_sys_act_bram_rd_en),
        .w_bram_rd_en(shared_sys_w_bram_rd_en),
        .bias_bram_rd_en(shared_sys_bias_bram_rd_en),
        .act_bram_addr(shared_sys_act_bram_addr),
        .w_bram_addr(shared_sys_w_bram_addr),
        .bias_bram_addr(shared_sys_bias_bram_addr),
        .w_bram_valid(shared_sys_w_bram_valid),
        .act_bram_data(shared_sys_act_bram_data),
        .w_bram_data(shared_sys_w_bram_data),
        .bias_bram_data(shared_sys_bias_bram_data),
        .opsum_ready(shared_sys_opsum_ready),
        .opsum_valid(shared_sys_opsum_valid),
        .opsum(shared_sys_opsum)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            top_state_q <= TOP_IDLE;
            patch_start <= 1'b0;
            vit_start <= 1'b0;
            patch_checkpoint_pending_q <= 1'b0;
        end
        else begin
            patch_start <= 1'b0;
            vit_start <= 1'b0;

            case (top_state_q)
                TOP_IDLE: begin
                    if (start_exec) begin
                        if (transformer_only_i) begin
                            top_state_q <= TOP_VIT_START;
                        end
                        else begin
                            patch_start <= 1'b1;
                            top_state_q <= TOP_PATCH;
                        end
                    end
                end

                TOP_PATCH: begin
                    if (patch_checkpoint_pending_q) begin
                        if (checkpoint_resume_i) begin
                            patch_checkpoint_pending_q <= 1'b0;
                            top_state_q <= TOP_VIT_START;
                        end
                    end
                    else if (patch_done) begin
                        if (checkpoint_enable_i) begin
                            patch_checkpoint_pending_q <= 1'b1;
                        end
                        else begin
                            top_state_q <= TOP_VIT_START;
                        end
                    end
                end

                TOP_VIT_START: begin
                    if (!vit_checkpoint_pending) begin
                        vit_start <= 1'b1;
                        top_state_q <= TOP_VIT_WAIT;
                    end
                end

                TOP_VIT_WAIT: begin
                    if (vit_done) begin
                        top_state_q <= TOP_IDLE;
                    end
                end

                default: begin
                    top_state_q <= TOP_IDLE;
                end
            endcase
        end
    end

endmodule
