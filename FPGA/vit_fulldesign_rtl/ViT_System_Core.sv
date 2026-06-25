`timescale 1ns/1ps

// ============================================================
// Module: ViT_System_Core
//
// Big core around ViT_Image_Accelerator_Top.
// The host no longer drives BRAM pins directly. Instead it streams commands
// into ViT_InputLoadFSM; this core generates internal memory write enables,
// supplies read data to patch embedding / transformer, and stalls systolic
// weight reads until the requested tile has been loaded.
//
// Memory target IDs for the loader:
//   0 IMAGE_BYTES       : DATA packs 4 image bytes
//   1 POS_BYTES         : DATA packs 4 position bytes
//   2 CLS_BYTES         : DATA packs 4 CLS bytes
//   3 LEGACY_INV_RMS    : accepted no-op; RMS1 inv_rms is generated on-chip
//   4 GAMMA_HALVES      : DATA[15:0] is one signed 16-bit gamma value
//                          addr 0..383 = RMSNorm1 gamma, 384..767 = RMSNorm2 gamma
//   5 PATCH_WEIGHT_TILE : BASE is tile id, DATA is one 8x8 weight tile
//   6 PATCH_BIAS_TILE   : BASE is tile id, DATA is one 8-output bias tile
//   7 TRANS_WEIGHT_TILE : BASE is tile id, DATA is one 8x8 weight tile
//   8 TRANS_BIAS_TILE   : BASE is tile id, DATA is one 8-output bias tile
//   9 GELU_PAGE         : BASE is 1024-word DDR page base
//  10 X_BUFFER_WORDS    : DATA packs 4 X activation bytes, BASE is X word addr
// ============================================================
module ViT_System_Core #(
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
    input  logic clk,
    input  logic rst_n,

    input  logic       start,
    input  logic       clear_done,
    input  logic       transformer_only,
    input  logic [5:0] patch_requant_shift,

    output logic busy,
    output logic done_pulse,
    output logic done_sticky,

    input  logic                 input_wr_valid,
    output logic                 input_wr_ready,
    input  logic [3:0]           input_wr_addr,
    input  logic [31:0]          input_wr_data,
    input  logic [3:0]           input_wr_strb,
    output logic                 input_busy,
    output logic                 input_done,
    output logic                 input_error,
    output logic [3:0]           input_active_target,
    output logic [LOAD_ADDR_W:0] input_word_count,

    input  logic                 output_host_en,
    input  logic [3:0]           output_host_target,
    input  logic [ADDR_W-1:0]    output_host_addr,
    output logic [31:0]          output_host_rd_data,

    input  logic                 stage_checkpoint_enable,
    input  logic                 stage_checkpoint_resume,
    output logic                 stage_checkpoint_pending,
    output logic                 stage_done_pulse,
    output logic [3:0]           stage_id,
    output logic [4:0]           stage_phase,

    input  logic                 gelu_store_done_i,

    output logic [31:0] status_word,
    output logic [31:0] patch_request_word,
    output logic [31:0] trans_request_word,
    output logic [31:0] page_request_word,
    output logic [31:0] debug_word,

    // Per-cycle performance events.  The AXI-Lite wrapper accumulates these
    // into readable counters for notebook-side bandwidth/energy estimates.
    output logic [7:0]  perf_bram_rd_words_o,
    output logic [7:0]  perf_bram_wr_words_o,
    output logic        perf_bram_active_o,
    output logic [15:0] perf_mac_ops_o,
    output logic        perf_pingpong_wait_o,
    output logic        perf_pingpong_load_o,
    output logic        perf_pingpong_overlap_o
);

    localparam int IMAGE_BYTES = IMG_H * IMG_W * IMG_C;
    localparam int POS_BYTES   = TOKEN_NUM * EMBED_DIM;
    localparam int CLS_BYTES   = EMBED_DIM;
    localparam int IMAGE_WORDS = (IMAGE_BYTES + 3) / 4;
    localparam int POS_WORDS   = (POS_BYTES + 3) / 4;
    localparam int CLS_WORDS   = (CLS_BYTES + 3) / 4;
    localparam int X_WORDS     = (TOKEN_NUM * EMBED_DIM + 3) / 4;
    localparam int PAGE_WORDS  = 1024;
    localparam int PAGE_AW     = 10;
    localparam int IMAGE_CACHE_SLOTS = 4;
    localparam int IMAGE_CACHE_SLOT_W = 2;
    localparam int IMAGE_BANKS = IMAGE_CACHE_SLOTS;
    localparam int POS_BANKS   = 1;
    localparam int TILE_ROW_WORDS = CHANNEL_TILE / 4;
    localparam int WEIGHT_TILE_WORDS = CHANNEL_TILE * TILE_ROW_WORDS;
    localparam int BIAS_TILE_WORDS = CHANNEL_TILE;
    localparam int WEIGHT_TILE_WORD_AW = $clog2(WEIGHT_TILE_WORDS);
    localparam int BIAS_TILE_WORD_AW = $clog2(BIAS_TILE_WORDS);
    localparam int PATCH_WEIGHT_TILE_COUNT = ((PATCH_ELEMS + CHANNEL_TILE - 1) / CHANNEL_TILE) *
                                             ((EMBED_DIM + CHANNEL_TILE - 1) / CHANNEL_TILE);
    localparam int PATCH_BIAS_TILE_COUNT = ((EMBED_DIM + CHANNEL_TILE - 1) / CHANNEL_TILE);
    localparam int CLS_WORD_AW = (CLS_WORDS <= 1) ? 1 : $clog2(CLS_WORDS);
    localparam logic [LOAD_ADDR_W-1:0] EMBED_DIM_LOAD = EMBED_DIM;
    localparam logic [LOAD_ADDR_W-1:0] GAMMA_COUNT_LOAD = 2 * EMBED_DIM;

    localparam logic [3:0] T_IMAGE   = 4'd0;
    localparam logic [3:0] T_POS     = 4'd1;
    localparam logic [3:0] T_CLS     = 4'd2;
    localparam logic [3:0] T_INV_RMS = 4'd3;
    localparam logic [3:0] T_GAMMA   = 4'd4;
    localparam logic [3:0] T_PW_TILE = 4'd5;
    localparam logic [3:0] T_PB_TILE = 4'd6;
    localparam logic [3:0] T_TW_TILE = 4'd7;
    localparam logic [3:0] T_TB_TILE = 4'd8;
    localparam logic [3:0] T_GELU_PAGE = 4'd9;
    localparam logic [3:0] T_X_BUF  = 4'd10;
    localparam logic [3:0] T_NORM_BUF = 4'd11;
    localparam logic [3:0] T_VXMID_BUF = 4'd12;
    localparam logic [3:0] T_SHARED_BUF = 4'd13;

    logic load_start_pulse;
    logic [3:0] load_start_target;
    logic [LOAD_ADDR_W-1:0] load_start_base;
    logic load_wr_en;
    logic [3:0] load_wr_target;
    logic [LOAD_ADDR_W-1:0] load_wr_addr;
    logic [31:0] load_wr_data;
    logic [3:0] load_wr_strb;
    logic load_done_pulse;
    logic [3:0] load_done_target;
    logic [LOAD_ADDR_W-1:0] load_done_base;
    logic [LOAD_ADDR_W:0] load_done_count;

    ViT_InputLoadFSM #(
        .ADDR_WIDTH(LOAD_ADDR_W),
        .TARGET_WIDTH(4),
        .TARGET_MAX(10)
    ) u_input_loader (
        .clk(clk),
        .rst_n(rst_n),
        .host_wr_valid(input_wr_valid),
        .host_wr_ready(input_wr_ready),
        .host_wr_addr(input_wr_addr),
        .host_wr_data(input_wr_data),
        .host_wr_strb(input_wr_strb),
        .host_busy(input_busy),
        .host_done(input_done),
        .host_error(input_error),
        .host_active_target(input_active_target),
        .host_word_count(input_word_count),
        .load_start_pulse(load_start_pulse),
        .load_start_target(load_start_target),
        .load_start_base(load_start_base),
        .load_wr_en(load_wr_en),
        .load_wr_target(load_wr_target),
        .load_wr_addr(load_wr_addr),
        .load_wr_data(load_wr_data),
        .load_wr_strb(load_wr_strb),
        .load_done_pulse(load_done_pulse),
        .load_done_target(load_done_target),
        .load_done_base(load_done_base),
        .load_done_count(load_done_count)
    );

    // Host-loaded staging memories.
    // Image and position are 1024-word BRAM page caches because patch embedding
    // streams them from DDR by page. CLS is only 384 bytes, so it uses
    // distributed RAM to avoid wasting one RAMB36E1.
    // Scalar byte reads are synchronous: address is sampled, then the byte is
    // selected from the returned word one cycle later.
    logic image_mem_wr_en;
    logic pos_mem_wr_en;
    logic cls_mem_wr_en;
    logic [16:0] image_mem_wr_addr;
    logic [16:0] pos_mem_wr_addr;
    logic [16:0] cls_mem_wr_addr;
    logic [16:0] image_mem_rd_addr;
    logic [16:0] pos_mem_rd_addr;
    logic [16:0] cls_mem_rd_addr;
    logic [LOAD_ADDR_W-1:0] image_page_base_q [0:IMAGE_CACHE_SLOTS-1];
    logic [LOAD_ADDR_W-1:0] pos_page_base_q;
    logic [IMAGE_CACHE_SLOTS-1:0] image_page_valid_q;
    logic pos_page_valid_q;
    logic image_page_hit;
    logic pos_page_hit;
    logic image_page_hit_q;
    logic pos_page_hit_q;
    logic [IMAGE_CACHE_SLOT_W-1:0] image_rd_slot;
    logic [IMAGE_CACHE_SLOT_W-1:0] image_load_slot_q;
    logic [IMAGE_CACHE_SLOT_W-1:0] image_replace_slot_q;
    logic [IMAGE_CACHE_SLOT_W-1:0] image_next_load_slot;
    logic [LOAD_ADDR_W-1:0] image_req_base;
    logic [LOAD_ADDR_W-1:0] pos_req_base;
    logic [31:0] image_mem_rd_word;
    logic [31:0] pos_mem_rd_word;
    logic [31:0] cls_mem_rd_word;
    logic [1:0] image_rd_lane_q;
    logic [1:0] pos_rd_lane_q;
    logic [1:0] cls_rd_lane_q;
    logic image_rd_in_range_q;
    logic pos_rd_in_range_q;
    logic cls_rd_in_range_q;

    logic patch_weight_wr_en;
    logic patch_bias_wr_en;
    logic trans_weight_wr_en;
    logic trans_bias_wr_en;
    logic [16:0] patch_weight_wr_addr;
    logic [16:0] patch_bias_wr_addr;
    logic [16:0] trans_weight_wr_addr;
    logic [16:0] trans_bias_wr_addr;
    logic [16:0] patch_weight_rd_addr;
    logic [16:0] patch_bias_rd_addr;
    logic [16:0] trans_weight_rd_addr;
    logic [16:0] trans_bias_rd_addr;
    logic gamma_wr_valid;
    logic [9:0] gamma_wr_addr;
    logic [9:0] gamma_rd_addr;
    logic signed [15:0] gamma_wr_data;
    logic gelu_loader_wr_en;
    logic [9:0] gelu_loader_wr_addr;
    logic [31:0] gelu_loader_wr_data;
    logic [3:0] gelu_loader_wr_strb;
    logic x_loader_word_we;
    logic [16:0] x_loader_word_addr;
    logic [31:0] x_loader_word_data;
    logic [3:0] x_loader_word_byte_en;
    logic [LOAD_ADDR_W-1:0] gelu_load_base_q;
    logic gelu_load_done_pulse;
    logic gelu_page_load_req_valid;
    logic [LOAD_ADDR_W-1:0] gelu_page_load_req_base;
    logic gelu_page_store_req_valid;
    logic [LOAD_ADDR_W-1:0] gelu_page_store_req_base;
    logic [31:0] gelu_page_store_rd_data;
    logic gelu_store_rd_en;
    logic [ADDR_W-1:0] gelu_store_rd_addr;
    logic gelu_page_wait_o;
    logic output_host_pending;
    logic [3:0] output_host_target_q;
    logic image_page_req_valid;
    logic pos_page_req_valid;

    logic [LOAD_ADDR_W-1:0] patch_weight_tile_id_q;
    logic [LOAD_ADDR_W-1:0] patch_bias_tile_id_q;
    logic [LOAD_ADDR_W-1:0] trans_weight_tile_id_q;
    logic [LOAD_ADDR_W-1:0] trans_bias_tile_id_q;
    logic patch_weight_valid_q;
    logic patch_bias_valid_q;
    logic trans_weight_valid_q;
    logic trans_bias_valid_q;

    integer lane;
    logic [LOAD_ADDR_W-1:0] byte_base_addr;
    logic [LOAD_ADDR_W-1:0] tile_word_offset;

    always_comb begin
        byte_base_addr   = load_wr_addr << 2;
        // Tile targets use BASE as tile id.  The local word index is the
        // current DATA write offset inside that tile, not the absolute BASE.
        tile_word_offset = load_wr_addr - load_start_base;
    end

    assign gamma_wr_valid = load_wr_en &&
                            (load_wr_target == T_GAMMA) &&
                            ((load_wr_strb[0] || load_wr_strb[1]) &&
                             (load_wr_addr < GAMMA_COUNT_LOAD));
    assign gamma_wr_addr = load_wr_addr[9:0];
    assign gamma_wr_data = load_wr_data[15:0];

    assign image_mem_wr_en   = load_wr_en && (load_wr_target == T_IMAGE) && (tile_word_offset < PAGE_WORDS);
    assign pos_mem_wr_en     = load_wr_en && (load_wr_target == T_POS) && (tile_word_offset < PAGE_WORDS);
    assign cls_mem_wr_en     = load_wr_en && (load_wr_target == T_CLS) && (load_wr_addr < CLS_WORDS);
    assign image_mem_wr_addr = {{(17-IMAGE_CACHE_SLOT_W-PAGE_AW){1'b0}},
                                image_load_slot_q,
                                tile_word_offset[PAGE_AW-1:0]};
    assign pos_mem_wr_addr   = {7'd0, tile_word_offset[PAGE_AW-1:0]};
    assign cls_mem_wr_addr   = load_wr_addr[16:0];
    assign gelu_loader_wr_en   = load_wr_en && (load_wr_target == T_GELU_PAGE) && (tile_word_offset < PAGE_WORDS);
    assign gelu_loader_wr_addr = tile_word_offset[9:0];
    assign gelu_loader_wr_data = load_wr_data;
    assign gelu_loader_wr_strb = load_wr_strb;
    assign gelu_load_done_pulse = load_done_pulse && (load_done_target == T_GELU_PAGE);
    assign x_loader_word_we      = load_wr_en && (load_wr_target == T_X_BUF) && (load_wr_addr < X_WORDS);
    assign x_loader_word_addr    = load_wr_addr[16:0];
    assign x_loader_word_data    = load_wr_data;
    assign x_loader_word_byte_en = load_wr_strb;

    assign patch_weight_wr_en   = load_wr_en && (load_wr_target == T_PW_TILE) && (tile_word_offset < WEIGHT_TILE_WORDS);
    assign patch_bias_wr_en     = load_wr_en && (load_wr_target == T_PB_TILE) && (tile_word_offset < BIAS_TILE_WORDS);
    assign trans_weight_wr_en   = load_wr_en && (load_wr_target == T_TW_TILE) && (tile_word_offset < WEIGHT_TILE_WORDS);
    assign trans_bias_wr_en     = load_wr_en && (load_wr_target == T_TB_TILE) && (tile_word_offset < BIAS_TILE_WORDS);
    assign patch_weight_wr_addr = {{(17-WEIGHT_TILE_WORD_AW){1'b0}}, tile_word_offset[WEIGHT_TILE_WORD_AW-1:0]};
    assign patch_bias_wr_addr   = {{(17-BIAS_TILE_WORD_AW){1'b0}}, tile_word_offset[BIAS_TILE_WORD_AW-1:0]};
    assign trans_weight_wr_addr = {{(17-WEIGHT_TILE_WORD_AW){1'b0}}, tile_word_offset[WEIGHT_TILE_WORD_AW-1:0]};
    assign trans_bias_wr_addr   = {{(17-BIAS_TILE_WORD_AW){1'b0}}, tile_word_offset[BIAS_TILE_WORD_AW-1:0]};

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            patch_weight_tile_id_q <= '0;
            patch_bias_tile_id_q <= '0;
            trans_weight_tile_id_q <= '0;
            trans_bias_tile_id_q <= '0;
            patch_weight_valid_q <= 1'b0;
            patch_bias_valid_q <= 1'b0;
            trans_weight_valid_q <= 1'b0;
            trans_bias_valid_q <= 1'b0;
            for (lane = 0; lane < IMAGE_CACHE_SLOTS; lane = lane + 1) begin
                image_page_base_q[lane] <= '0;
            end
            pos_page_base_q <= '0;
            image_page_valid_q <= '0;
            pos_page_valid_q <= 1'b0;
            image_load_slot_q <= '0;
            image_replace_slot_q <= '0;
        end
        else begin
            if (load_start_pulse) begin
                case (load_start_target)
                    T_IMAGE: begin
                        image_load_slot_q <= image_next_load_slot;
                        image_page_valid_q[image_next_load_slot] <= 1'b0;
                    end
                    T_PW_TILE: patch_weight_valid_q <= 1'b0;
                    T_PB_TILE: patch_bias_valid_q <= 1'b0;
                    T_TW_TILE: trans_weight_valid_q <= 1'b0;
                    T_TB_TILE: trans_bias_valid_q <= 1'b0;
                    default: begin
                    end
                endcase
            end

            if (load_wr_en) begin
                case (load_wr_target)
                    T_IMAGE: begin
                        // Image write is handled by u_image_mem.
                    end

                    T_POS: begin
                        // Position write is handled by u_pos_mem.
                    end

                    T_CLS: begin
                        // CLS write is handled by u_cls_mem.
                    end

                    T_INV_RMS: begin
                        // Legacy loader target kept for old host code.
                        // RMS1 statistics now come from X BRAM and are written to Token_Stat_BRAM.
                    end

                    T_GAMMA: begin
                        // Gamma writes are handled by Gamma_Buffer_BRAM below.
                        // One loader DATA beat writes one gamma value in DATA[15:0].
                    end

                    T_PW_TILE: begin
                        // BRAM write handled by u_patch_weight_tile_bram.
                    end

                    T_PB_TILE: begin
                        // Distributed bias tile write is handled below.
                    end

                    T_TW_TILE: begin
                        // BRAM write handled by u_trans_weight_tile_bram.
                    end

                    T_TB_TILE: begin
                        // Distributed bias tile write is handled below.
                    end

                    default: begin
                    end
                endcase
            end

            if (load_done_pulse) begin
                if (load_done_target == T_IMAGE) begin
                    image_page_base_q[image_load_slot_q] <= load_done_base;
                    image_page_valid_q[image_load_slot_q] <= 1'b1;
                    image_replace_slot_q <= image_load_slot_q + 1'b1;
                end
                if (load_done_target == T_POS) begin
                    pos_page_base_q <= load_done_base;
                    pos_page_valid_q <= 1'b1;
                end
                case (load_done_target)
                    T_PW_TILE: begin
                        patch_weight_tile_id_q <= load_done_base;
                        patch_weight_valid_q <= 1'b1;
                    end
                    T_PB_TILE: begin
                        patch_bias_tile_id_q <= load_done_base;
                        patch_bias_valid_q <= 1'b1;
                    end
                    T_TW_TILE: begin
                        trans_weight_tile_id_q <= load_done_base;
                        trans_weight_valid_q <= 1'b1;
                    end
                    T_TB_TILE: begin
                        trans_bias_tile_id_q <= load_done_base;
                        trans_bias_valid_q <= 1'b1;
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    logic [IMG_ADDR_W-1:0] image_rd_addr;
    logic [DATA_W-1:0] image_rd_data;
    logic image_rd_valid;
    logic patch_w_rd_en;
    logic [PE_W_ADDR_W-1:0] patch_w_addr;
    logic [31:0] patch_w_data;
    logic patch_w_valid;
    logic patch_bias_rd_en;
    logic [16:0] patch_bias_addr;
    logic [31:0] patch_bias_data;
    logic [PE_POS_ADDR_W-1:0] patch_pos_addr;
    logic [DATA_W-1:0] patch_pos_data;
    logic patch_pos_valid;
    logic [PE_EMBED_ADDR_W-1:0] cls_addr;
    logic [DATA_W-1:0] cls_data;
    logic cls_valid;
    logic w_bram_rd_en;
    logic bias_bram_rd_en;
    logic [16:0] w_bram_addr;
    logic [16:0] bias_bram_addr;
    logic w_bram_valid;
    logic [31:0] w_bram_data;
    logic [31:0] bias_bram_data;
    logic rms_norm_sel_o;
    logic [8:0] gamma_addr;
    logic signed [15:0] gamma_data;
    logic [ADDR_W-1:0] x_out_raddr;
    logic [DATA_W-1:0] x_out_rdata;
    logic [3:0] debug_rd_target;
    logic [ADDR_W-1:0] debug_rd_addr;
    logic [31:0] debug_rd_data;
    logic [1:0] debug_top_state;
    logic [4:0] debug_phase;
    logic [15:0] debug_tile_count;
    logic [15:0] debug_softmax_count;
    logic ppu_data_tile_valid_o;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_data_tile_o;
    logic stat_valid_o;
    logic [TOKEN_W-1:0] stat_token_idx_o;
    logic [SUM_W-1:0] sum_sq_o;
    logic patch_busy_o;
    logic patch_done_o;
    logic [2:0] patch_debug_state_o;
    logic [15:0] patch_debug_patch_idx_o;
    logic [15:0] patch_debug_channel_idx_o;
    logic [15:0] patch_debug_elem_idx_o;
    logic [7:0] image_perf_bram_rd_words;
    logic [7:0] image_perf_bram_wr_words;
    logic image_perf_bram_active;
    logic [15:0] image_perf_mac_ops;
    logic image_perf_pingpong_wait;
    logic image_perf_pingpong_load;
    logic image_perf_pingpong_overlap;
    logic [7:0] loader_perf_bram_wr_words;

    logic [LOAD_ADDR_W-1:0] patch_weight_req_tile;
    logic [LOAD_ADDR_W-1:0] patch_bias_req_tile;
    logic [LOAD_ADDR_W-1:0] trans_weight_req_tile;
    logic [LOAD_ADDR_W-1:0] trans_bias_req_tile;
    logic patch_weight_match;
    logic patch_bias_match;
    logic trans_weight_match;
    logic trans_bias_match;
    logic trans_external_phase;
    logic patch_external_phase;
    logic patch_request_in_range;

    assign patch_weight_req_tile = {{(LOAD_ADDR_W-PE_W_ADDR_W){1'b0}}, patch_w_addr} >> WEIGHT_TILE_WORD_AW;
    assign patch_bias_req_tile = {{(LOAD_ADDR_W-17){1'b0}}, patch_bias_addr} >> BIAS_TILE_WORD_AW;
    assign trans_weight_req_tile = {{(LOAD_ADDR_W-17){1'b0}}, w_bram_addr} >> WEIGHT_TILE_WORD_AW;
    assign trans_bias_req_tile = {{(LOAD_ADDR_W-17){1'b0}}, bias_bram_addr} >> BIAS_TILE_WORD_AW;

    assign patch_weight_match = patch_weight_valid_q && (patch_weight_tile_id_q == patch_weight_req_tile);
    assign patch_bias_match = patch_bias_valid_q && (patch_bias_tile_id_q == patch_bias_req_tile);
    assign trans_weight_match = trans_weight_valid_q && (trans_weight_tile_id_q == trans_weight_req_tile);
    assign trans_bias_match = trans_bias_valid_q && (trans_bias_tile_id_q == trans_bias_req_tile);

    assign trans_external_phase =
        (debug_phase == 5'd2) || (debug_phase == 5'd6) ||
        (debug_phase == 5'd8) || (debug_phase == 5'd9);
    assign patch_request_in_range =
        (patch_weight_req_tile < PATCH_WEIGHT_TILE_COUNT) &&
        (patch_bias_req_tile < PATCH_BIAS_TILE_COUNT);
    assign patch_external_phase =
        patch_busy_o &&
        patch_request_in_range &&
        ((patch_debug_state_o == 3'd5) || (patch_debug_state_o == 3'd6));

    assign image_req_base = (({{(LOAD_ADDR_W-IMG_ADDR_W){1'b0}}, image_rd_addr} >> 2) & ~{{(LOAD_ADDR_W-PAGE_AW){1'b0}}, {PAGE_AW{1'b1}}});
    assign pos_req_base   = (({{(LOAD_ADDR_W-PE_POS_ADDR_W){1'b0}}, patch_pos_addr} >> 2) & ~{{(LOAD_ADDR_W-PAGE_AW){1'b0}}, {PAGE_AW{1'b1}}});
    assign pos_page_hit   = pos_page_valid_q && (pos_req_base == pos_page_base_q);

    always_comb begin
        image_page_hit = 1'b0;
        image_rd_slot = '0;
        for (int slot = 0; slot < IMAGE_CACHE_SLOTS; slot = slot + 1) begin
            if (image_page_valid_q[slot] && (image_req_base == image_page_base_q[slot])) begin
                image_page_hit = 1'b1;
                image_rd_slot = slot[IMAGE_CACHE_SLOT_W-1:0];
            end
        end
    end

    always_comb begin
        image_next_load_slot = image_replace_slot_q;
        for (int slot = 0; slot < IMAGE_CACHE_SLOTS; slot = slot + 1) begin
            if (!image_page_valid_q[slot]) begin
                image_next_load_slot = slot[IMAGE_CACHE_SLOT_W-1:0];
            end
        end
        for (int slot = 0; slot < IMAGE_CACHE_SLOTS; slot = slot + 1) begin
            if (image_page_valid_q[slot] && (load_start_base == image_page_base_q[slot])) begin
                image_next_load_slot = slot[IMAGE_CACHE_SLOT_W-1:0];
            end
        end
    end

    assign image_mem_rd_addr = {{(17-IMAGE_CACHE_SLOT_W-PAGE_AW){1'b0}},
                                image_rd_slot,
                                image_rd_addr[PAGE_AW+1:2]};
    assign pos_mem_rd_addr   = {7'd0, patch_pos_addr[PAGE_AW+1:2]};
    assign cls_mem_rd_addr   = cls_addr >> 2;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            image_rd_lane_q <= 2'd0;
            pos_rd_lane_q <= 2'd0;
            cls_rd_lane_q <= 2'd0;
            image_rd_in_range_q <= 1'b0;
            pos_rd_in_range_q <= 1'b0;
            cls_rd_in_range_q <= 1'b0;
            image_page_hit_q <= 1'b0;
            pos_page_hit_q <= 1'b0;
        end
        else begin
            image_rd_lane_q <= image_rd_addr[1:0];
            pos_rd_lane_q <= patch_pos_addr[1:0];
            cls_rd_lane_q <= cls_addr[1:0];
            image_rd_in_range_q <= (image_rd_addr < IMAGE_BYTES);
            pos_rd_in_range_q <= (patch_pos_addr < POS_BYTES);
            cls_rd_in_range_q <= (cls_addr < CLS_BYTES);
            image_page_hit_q <= image_page_hit;
            pos_page_hit_q <= pos_page_hit;
        end
    end

    always_comb begin
        case (image_rd_lane_q)
            2'd0: image_rd_data = image_mem_rd_word[7:0];
            2'd1: image_rd_data = image_mem_rd_word[15:8];
            2'd2: image_rd_data = image_mem_rd_word[23:16];
            default: image_rd_data = image_mem_rd_word[31:24];
        endcase
        if (!image_rd_in_range_q) begin
            image_rd_data = 8'd0;
        end
    end

    always_comb begin
        case (pos_rd_lane_q)
            2'd0: patch_pos_data = pos_mem_rd_word[7:0];
            2'd1: patch_pos_data = pos_mem_rd_word[15:8];
            2'd2: patch_pos_data = pos_mem_rd_word[23:16];
            default: patch_pos_data = pos_mem_rd_word[31:24];
        endcase
        if (!pos_rd_in_range_q) begin
            patch_pos_data = 8'd128;
        end
    end

    always_comb begin
        case (cls_rd_lane_q)
            2'd0: cls_data = cls_mem_rd_word[7:0];
            2'd1: cls_data = cls_mem_rd_word[15:8];
            2'd2: cls_data = cls_mem_rd_word[23:16];
            default: cls_data = cls_mem_rd_word[31:24];
        endcase
        if (!cls_rd_in_range_q) begin
            cls_data = 8'd128;
        end
    end

    assign image_rd_valid = image_rd_in_range_q && image_page_hit_q;
    assign patch_pos_valid = pos_rd_in_range_q && pos_page_hit_q;
    assign cls_valid = cls_rd_in_range_q;

    ActivationMem #(
        .INIT_FILE("NONE"),
        .NUM_BANKS(IMAGE_BANKS)
    ) u_image_mem (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(1'b1),
        .rd_addr(image_mem_rd_addr),
        .rd_data(image_mem_rd_word),
        .wr_en(image_mem_wr_en),
        .wr_addr(image_mem_wr_addr),
        .wr_data(load_wr_data),
        .wr_byte_en(load_wr_strb)
    );

    ActivationMem #(
        .INIT_FILE("NONE"),
        .NUM_BANKS(POS_BANKS)
    ) u_pos_mem (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(1'b1),
        .rd_addr(pos_mem_rd_addr),
        .rd_data(pos_mem_rd_word),
        .wr_en(pos_mem_wr_en),
        .wr_addr(pos_mem_wr_addr),
        .wr_data(load_wr_data),
        .wr_byte_en(load_wr_strb)
    );

    (* ram_style = "distributed" *) logic [31:0] cls_mem [0:CLS_WORDS-1];
    logic [CLS_WORD_AW-1:0] cls_mem_wr_word_idx;
    logic [CLS_WORD_AW-1:0] cls_mem_rd_word_idx;
    assign cls_mem_wr_word_idx = cls_mem_wr_addr[CLS_WORD_AW-1:0];
    assign cls_mem_rd_word_idx = cls_mem_rd_addr[CLS_WORD_AW-1:0];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cls_mem_rd_word <= 32'd0;
        end
        else begin
            if (cls_mem_wr_en && (cls_mem_wr_addr < CLS_WORDS)) begin
                if (load_wr_strb[0]) cls_mem[cls_mem_wr_word_idx][7:0]   <= load_wr_data[7:0];
                if (load_wr_strb[1]) cls_mem[cls_mem_wr_word_idx][15:8]  <= load_wr_data[15:8];
                if (load_wr_strb[2]) cls_mem[cls_mem_wr_word_idx][23:16] <= load_wr_data[23:16];
                if (load_wr_strb[3]) cls_mem[cls_mem_wr_word_idx][31:24] <= load_wr_data[31:24];
            end

            if (cls_mem_rd_addr < CLS_WORDS)
                cls_mem_rd_word <= cls_mem[cls_mem_rd_word_idx];
            else
                cls_mem_rd_word <= 32'd0;
        end
    end

    assign patch_w_valid = patch_weight_match && patch_bias_match;
    assign patch_weight_rd_addr = {{(17-WEIGHT_TILE_WORD_AW){1'b0}}, patch_w_addr[WEIGHT_TILE_WORD_AW-1:0]};
    assign patch_bias_rd_addr   = {{(17-BIAS_TILE_WORD_AW){1'b0}}, patch_bias_addr[BIAS_TILE_WORD_AW-1:0]};

    assign w_bram_valid = (!trans_external_phase) || (trans_weight_match && trans_bias_match);
    assign trans_weight_rd_addr = {{(17-WEIGHT_TILE_WORD_AW){1'b0}}, w_bram_addr[WEIGHT_TILE_WORD_AW-1:0]};
    assign trans_bias_rd_addr   = {{(17-BIAS_TILE_WORD_AW){1'b0}}, bias_bram_addr[BIAS_TILE_WORD_AW-1:0]};

    WeightMem #(
        .INIT_FILE("NONE"),
        .NUM_BANKS(1)
    ) u_patch_weight_tile_bram (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(patch_w_rd_en),
        .rd_addr(patch_weight_rd_addr),
        .rd_data(patch_w_data),
        .wr_en(patch_weight_wr_en),
        .wr_addr(patch_weight_wr_addr),
        .wr_data(load_wr_data),
        .wr_byte_en(load_wr_strb)
    );

    (* ram_style = "distributed" *) logic [31:0] patch_bias_tile_mem [0:BIAS_TILE_WORDS-1];
always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            patch_bias_data <= 32'd0;
        end
        else begin
            if (patch_bias_wr_en) begin
                if (load_wr_strb[0]) patch_bias_tile_mem[patch_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][7:0]   <= load_wr_data[7:0];
                if (load_wr_strb[1]) patch_bias_tile_mem[patch_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][15:8]  <= load_wr_data[15:8];
                if (load_wr_strb[2]) patch_bias_tile_mem[patch_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][23:16] <= load_wr_data[23:16];
                if (load_wr_strb[3]) patch_bias_tile_mem[patch_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][31:24] <= load_wr_data[31:24];
            end

            if (patch_bias_rd_en)
                patch_bias_data <= patch_bias_tile_mem[patch_bias_rd_addr[BIAS_TILE_WORD_AW-1:0]];
        end
    end

    WeightMem #(
        .INIT_FILE("NONE"),
        .NUM_BANKS(1)
    ) u_trans_weight_tile_bram (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(w_bram_rd_en),
        .rd_addr(trans_weight_rd_addr),
        .rd_data(w_bram_data),
        .wr_en(trans_weight_wr_en),
        .wr_addr(trans_weight_wr_addr),
        .wr_data(load_wr_data),
        .wr_byte_en(load_wr_strb)
    );

    (* ram_style = "distributed" *) logic [31:0] trans_bias_tile_mem [0:BIAS_TILE_WORDS-1];
always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bias_bram_data <= 32'd0;
        end
        else begin
            if (trans_bias_wr_en) begin
                if (load_wr_strb[0]) trans_bias_tile_mem[trans_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][7:0]   <= load_wr_data[7:0];
                if (load_wr_strb[1]) trans_bias_tile_mem[trans_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][15:8]  <= load_wr_data[15:8];
                if (load_wr_strb[2]) trans_bias_tile_mem[trans_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][23:16] <= load_wr_data[23:16];
                if (load_wr_strb[3]) trans_bias_tile_mem[trans_bias_wr_addr[BIAS_TILE_WORD_AW-1:0]][31:24] <= load_wr_data[31:24];
            end

            if (bias_bram_rd_en)
                bias_bram_data <= trans_bias_tile_mem[trans_bias_rd_addr[BIAS_TILE_WORD_AW-1:0]];
        end
    end

    assign gamma_rd_addr = {rms_norm_sel_o, gamma_addr};

    Gamma_Buffer_BRAM #(
        .CHANNEL_NUM(2 * EMBED_DIM),
        .CHANNEL_AW(10),
        .DATA_W(16)
    ) u_gamma_bram (
        .clk(clk),
        .rst_n(rst_n),
        .wr_valid_i(gamma_wr_valid),
        .wr_addr_i(gamma_wr_addr),
        .wr_data_i(gamma_wr_data),
        .rd_addr_i(gamma_rd_addr),
        .rd_data_o(gamma_data)
    );

    logic busy_exec_int;
    logic done_exec_int;

    assign busy = busy_exec_int;
    assign done_pulse = done_exec_int;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done_sticky <= 1'b0;
        end
        else begin
            if (start || clear_done) begin
                done_sticky <= 1'b0;
            end
            else if (done_exec_int) begin
                done_sticky <= 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            x_out_raddr <= '0;
            debug_rd_target <= 4'd0;
            debug_rd_addr <= '0;
            gelu_store_rd_en <= 1'b0;
            gelu_store_rd_addr <= '0;
            output_host_pending <= 1'b0;
            output_host_target_q <= 4'd0;
            output_host_rd_data <= 32'd0;
        end
        else begin
            gelu_store_rd_en <= 1'b0;
            output_host_pending <= 1'b0;

            if (output_host_en) begin
                output_host_pending <= 1'b1;
                output_host_target_q <= output_host_target;
                if (output_host_target == T_GELU_PAGE) begin
                    gelu_store_rd_en <= 1'b1;
                    gelu_store_rd_addr <= output_host_addr;
                end
                else if ((output_host_target == T_X_BUF) ||
                         (output_host_target == T_NORM_BUF) ||
                         (output_host_target == T_VXMID_BUF) ||
                         (output_host_target == T_SHARED_BUF)) begin
                    debug_rd_target <= output_host_target;
                    debug_rd_addr <= output_host_addr;
                end
                else begin
                    x_out_raddr <= output_host_addr;
                end
            end

            if (output_host_pending) begin
                if (output_host_target_q == T_GELU_PAGE) begin
                    output_host_rd_data <= gelu_page_store_rd_data;
                end
                else if ((output_host_target_q == T_X_BUF) ||
                         (output_host_target_q == T_NORM_BUF) ||
                         (output_host_target_q == T_VXMID_BUF) ||
                         (output_host_target_q == T_SHARED_BUF)) begin
                    output_host_rd_data <= debug_rd_data;
                end
                else begin
                    output_host_rd_data <= {24'd0, x_out_rdata};
                end
            end
        end
    end

    assign patch_request_word = {
        patch_bias_req_tile[15:0],
        patch_weight_req_tile[15:0]
    };

    assign trans_request_word = {
        trans_bias_req_tile[15:0],
        trans_weight_req_tile[15:0]
    };

    assign image_page_req_valid = patch_busy_o && (image_rd_addr < IMAGE_BYTES) && !image_page_hit;
    assign pos_page_req_valid = patch_busy_o && (patch_pos_addr < POS_BYTES) && !pos_page_hit;

    always_comb begin
        page_request_word = 32'd0;
        if (gelu_page_store_req_valid) begin
            page_request_word = {1'b1, 1'b1, 2'b00, T_GELU_PAGE,
                                 {{(24-LOAD_ADDR_W){1'b0}}, gelu_page_store_req_base}};
        end
        else if (gelu_page_load_req_valid) begin
            page_request_word = {1'b1, 1'b0, 2'b00, T_GELU_PAGE,
                                 {{(24-LOAD_ADDR_W){1'b0}}, gelu_page_load_req_base}};
        end
        else if (image_page_req_valid) begin
            page_request_word = {1'b1, 1'b0, 2'b00, T_IMAGE,
                                 {{(24-LOAD_ADDR_W){1'b0}}, image_req_base}};
        end
        else if (pos_page_req_valid) begin
            page_request_word = {1'b1, 1'b0, 2'b00, T_POS,
                                 {{(24-LOAD_ADDR_W){1'b0}}, pos_req_base}};
        end
    end

    assign debug_word = patch_busy_o ? {
        patch_debug_patch_idx_o[7:0],
        patch_debug_elem_idx_o[7:0],
        3'd0,
        {2'd0, patch_debug_state_o},
        6'd0,
        debug_top_state
    } : {
        debug_softmax_count[7:0],
        debug_tile_count[7:0],
        3'd0,
        debug_phase,
        6'd0,
        debug_top_state
    };

    assign status_word = {
        12'd0,
        patch_external_phase,
        trans_external_phase,
        trans_bias_match,
        trans_weight_match,
        patch_bias_match,
        patch_weight_match,
        trans_bias_valid_q,
        trans_weight_valid_q,
        patch_bias_valid_q,
        patch_weight_valid_q,
        (trans_external_phase && !trans_bias_match),
        (trans_external_phase && !trans_weight_match),
        (patch_external_phase && !patch_bias_match),
        (patch_external_phase && !patch_weight_match),
        input_error,
        input_done,
        input_busy,
        done_sticky,
        done_exec_int,
        busy_exec_int
    };

    ViT_Image_Accelerator_Top #(
        .IMG_H(IMG_H),
        .IMG_W(IMG_W),
        .IMG_C(IMG_C),
        .PATCH_SIZE(PATCH_SIZE),
        .EMBED_DIM(EMBED_DIM),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ADDR_W(ADDR_W),
        .HEAD_NUM(HEAD_NUM),
        .HEAD_DIM(HEAD_DIM),
        .FFN_CHANNEL_NUM(FFN_CHANNEL_NUM),
        .SOFTMAX_COLS(SOFTMAX_COLS),
        .SOFTMAX_EXP_LUT_HEX(SOFTMAX_EXP_LUT_HEX),
        .PATCH_GRID_H(PATCH_GRID_H),
        .PATCH_GRID_W(PATCH_GRID_W),
        .PATCH_COUNT(PATCH_COUNT),
        .TOKEN_NUM(TOKEN_NUM),
        .PATCH_ELEMS(PATCH_ELEMS),
        .IMG_ADDR_W(IMG_ADDR_W),
        .PE_W_ADDR_W(PE_W_ADDR_W),
        .PE_POS_ADDR_W(PE_POS_ADDR_W),
        .PE_EMBED_ADDR_W(PE_EMBED_ADDR_W),
        .LOAD_ADDR_W(LOAD_ADDR_W)
    ) u_vit_image_top (
        .clk(clk),
        .rst_n(rst_n),
        .start_exec(start),
        .transformer_only_i(transformer_only),
        .patch_requant_shift_i(patch_requant_shift),
        .busy_exec(busy_exec_int),
        .done_exec(done_exec_int),
        .debug_top_state(debug_top_state),
        .checkpoint_enable_i(stage_checkpoint_enable),
        .checkpoint_resume_i(stage_checkpoint_resume),
        .checkpoint_pending_o(stage_checkpoint_pending),
        .stage_done_pulse_o(stage_done_pulse),
        .stage_id_o(stage_id),
        .stage_phase_o(stage_phase),
        .debug_rd_target_i(debug_rd_target),
        .debug_rd_addr_i(debug_rd_addr),
        .debug_rd_data_o(debug_rd_data),
        .image_rd_addr(image_rd_addr),
        .image_rd_data(image_rd_data),
        .image_rd_valid(image_rd_valid),
        .patch_w_rd_en(patch_w_rd_en),
        .patch_w_addr(patch_w_addr),
        .patch_w_data(patch_w_data),
        .patch_w_valid(patch_w_valid),
        .patch_bias_rd_en(patch_bias_rd_en),
        .patch_bias_addr(patch_bias_addr),
        .patch_bias_data(patch_bias_data),
        .patch_pos_addr(patch_pos_addr),
        .patch_pos_data(patch_pos_data),
        .patch_pos_valid(patch_pos_valid),
        .cls_addr(cls_addr),
        .cls_data(cls_data),
        .cls_valid(cls_valid),
        .x_loader_word_we(x_loader_word_we),
        .x_loader_word_addr(x_loader_word_addr),
        .x_loader_word_data(x_loader_word_data),
        .x_loader_word_byte_en(x_loader_word_byte_en),
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
        .x_out_raddr(x_out_raddr),
        .x_out_rdata(x_out_rdata),
        .gelu_loader_wr_en(gelu_loader_wr_en),
        .gelu_loader_wr_addr(gelu_loader_wr_addr),
        .gelu_loader_wr_data(gelu_loader_wr_data),
        .gelu_loader_wr_strb(gelu_loader_wr_strb),
        .gelu_load_done_i(gelu_load_done_pulse),
        .gelu_load_base_i(load_done_base),
        .gelu_store_done_i(gelu_store_done_i),
        .gelu_store_rd_en(gelu_store_rd_en),
        .gelu_store_rd_addr(gelu_store_rd_addr),
        .gelu_store_rd_data(gelu_page_store_rd_data),
        .gelu_page_load_req_valid(gelu_page_load_req_valid),
        .gelu_page_load_req_base(gelu_page_load_req_base),
        .gelu_page_store_req_valid(gelu_page_store_req_valid),
        .gelu_page_store_req_base(gelu_page_store_req_base),
        .gelu_page_wait_o(gelu_page_wait_o),
        .debug_phase(debug_phase),
        .debug_tile_count(debug_tile_count),
        .debug_softmax_count(debug_softmax_count),
        .ppu_data_tile_valid_o(ppu_data_tile_valid_o),
        .ppu_data_tile_o(ppu_data_tile_o),
        .stat_valid_o(stat_valid_o),
        .stat_token_idx_o(stat_token_idx_o),
        .sum_sq_o(sum_sq_o),
        .patch_busy_o(patch_busy_o),
        .patch_done_o(patch_done_o),
        .patch_debug_state_o(patch_debug_state_o),
        .patch_debug_patch_idx_o(patch_debug_patch_idx_o),
        .patch_debug_channel_idx_o(patch_debug_channel_idx_o),
        .patch_debug_elem_idx_o(patch_debug_elem_idx_o),
        .perf_bram_rd_words_o(image_perf_bram_rd_words),
        .perf_bram_wr_words_o(image_perf_bram_wr_words),
        .perf_bram_active_o(image_perf_bram_active),
        .perf_mac_ops_o(image_perf_mac_ops),
        .perf_pingpong_wait_o(image_perf_pingpong_wait),
        .perf_pingpong_load_o(image_perf_pingpong_load),
        .perf_pingpong_overlap_o(image_perf_pingpong_overlap)
    );

    always_comb begin
        loader_perf_bram_wr_words = 8'd0;
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, image_mem_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, pos_mem_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, cls_mem_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, patch_weight_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, patch_bias_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, trans_weight_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, trans_bias_wr_en};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, gamma_wr_valid};
        loader_perf_bram_wr_words = loader_perf_bram_wr_words + {7'd0, gelu_loader_wr_en};
    end

    assign perf_bram_rd_words_o     = image_perf_bram_rd_words;
    assign perf_bram_wr_words_o     = image_perf_bram_wr_words + loader_perf_bram_wr_words;
    assign perf_bram_active_o       = image_perf_bram_active || |loader_perf_bram_wr_words;
    assign perf_mac_ops_o           = image_perf_mac_ops;
    assign perf_pingpong_wait_o     = image_perf_pingpong_wait;
    assign perf_pingpong_load_o     = image_perf_pingpong_load;
    assign perf_pingpong_overlap_o  = image_perf_pingpong_overlap;

endmodule


