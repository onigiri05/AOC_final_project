`timescale 1ns/1ps

// ============================================================
// Module: Patch_Embedding_Systolic_Top
//
// 功能:
//   使用原本的 16x16 Systolic Array 做 ViT patch embedding。
//   raw image pixel 會先被整理成 Systolic activation BRAM 的 128-bit row，
//   再執行:
//
//       patch_act [16 patches x PATCH_ELEMS]
//         x
//       patch_weight [PATCH_ELEMS x 16 embed channels]
//
//   Systolic 輸出的 opsum 會做 requant，再加 position embedding，
//   最後寫入 ViT_Accelerator_Top 的 X Buffer。
//
// FPGA 接線重點:
//   1. image_rd_* 仍是 raw image scalar pixel 讀取介面。
//   2. patch_w_* 改成 Systolic 原本使用的 128-bit row 介面。
//      patch_w_addr_o 的 layout:
//        n_tile * (PATCH_K_TILE_NUM * 16) + k_tile * 16 + k_inner
//   3. patch_bias_* 也是 Systolic bias row 介面。
//      patch_bias_addr_o = n_tile * 4 + bias_word_idx
//      原本 Systolic.v 的 BRAM 介面相同。若外部使用同步 BRAM，通常是
//      address 出去後下一拍回 row data。
//
// 注意:
//   目前 Systolic.v 的 activation 是 unsigned 8-bit MAC，沒有在 PE 內扣
//   input zero-point。若 patch embedding 權重是用 centered pixel 訓練，
//   zero-point correction 要併入 bias，或之後再改 PE 支援 signed activation。
// ============================================================
// 新版 FPGA 接線補充:
//   patch_w_* / patch_bias_* 已改成新版 Systolic.v 的 32-bit word 介面。
//   patch_w_addr_o = old_128b_row_addr * 4 + word_sel。
//   patch_w_data_i[8*c +: 8] 對應同一 row 裡連續 4 個 output channel。
//   patch_bias_addr_o = n_tile * 16 + output_channel_inner，每個 bias 是 signed INT32。
//   patch_w_valid_i 在目前 systolic 裡是 weight buffer ready，不是單純 data-valid。
module Patch_Embedding_Systolic_Top #(
    parameter int IMG_H        = 224,
    parameter int IMG_W        = 224,
    parameter int IMG_C        = 3,
    parameter int PATCH_SIZE   = 16,
    parameter int EMBED_DIM    = 384,
    parameter int TOKEN_TILE   = 8,
    parameter int CHANNEL_TILE = 8,
    parameter int DATA_W       = 8,
    parameter int ACC_W        = 32,
    parameter int OUTPUT_ZP    = 128,

    parameter int PATCH_GRID_H = IMG_H / PATCH_SIZE,
    parameter int PATCH_GRID_W = IMG_W / PATCH_SIZE,
    parameter int PATCH_COUNT  = PATCH_GRID_H * PATCH_GRID_W,
    parameter int TOKEN_NUM    = PATCH_COUNT + 1,
    parameter int PATCH_ELEMS  = PATCH_SIZE * PATCH_SIZE * IMG_C,

    parameter int PATCH_M_TILE_NUM = (PATCH_COUNT + TOKEN_TILE - 1) / TOKEN_TILE,
    parameter int PATCH_K_TILE_NUM = (PATCH_ELEMS + CHANNEL_TILE - 1) / CHANNEL_TILE,
    parameter int EMBED_TILE_NUM   = (EMBED_DIM + CHANNEL_TILE - 1) / CHANNEL_TILE,
    parameter int ACT_ROW_COUNT    = PATCH_M_TILE_NUM * PATCH_K_TILE_NUM * TOKEN_TILE,

    parameter int IMG_ADDR_W   = $clog2(IMG_H * IMG_W * IMG_C),
    parameter int POS_ADDR_W   = $clog2(TOKEN_NUM * EMBED_DIM),
    parameter int EMBED_ADDR_W = $clog2(EMBED_DIM),
    parameter int ADDR_W       = $clog2(TOKEN_NUM * EMBED_DIM)
)(
    input  logic clk,
    input  logic rst_n,

    input  logic start_i,
    input  logic [5:0] requant_shift_i,
    output logic busy_o,
    output logic done_o,

    output logic [IMG_ADDR_W-1:0] image_rd_addr_o,
    input  logic [DATA_W-1:0]     image_rd_data_i,
    input  logic                  image_rd_valid_i,

    output logic                  patch_w_rd_en_o,
    output logic [16:0]           patch_w_addr_o,
    input  logic [31:0]           patch_w_data_i,
    input  logic                  patch_w_valid_i,

    output logic                  patch_bias_rd_en_o,
    output logic [16:0]           patch_bias_addr_o,
    input  logic [31:0]           patch_bias_data_i,

    output logic [POS_ADDR_W-1:0]   pos_addr_o,
    input  logic [DATA_W-1:0]       pos_data_i,
    input  logic                    pos_valid_i,

    output logic [EMBED_ADDR_W-1:0] cls_addr_o,
    input  logic [DATA_W-1:0]       cls_data_i,
    input  logic                    cls_valid_i,

    output logic                    x_buf_we_o,
    output logic [ADDR_W-1:0]       x_buf_addr_o,
    output logic [DATA_W-1:0]       x_buf_wdata_o,
    output logic                    x_buf_word_we_o,
    output logic [16:0]             x_buf_word_addr_o,
    output logic [31:0]             x_buf_word_data_o,
    output logic [3:0]              x_buf_word_byte_en_o,

    output logic [2:0]              debug_state_o,
    output logic [15:0]             debug_patch_idx_o,
    output logic [15:0]             debug_channel_idx_o,
    output logic [15:0]             debug_elem_idx_o,

    // Shared systolic interface.  Patch embedding does not instantiate a
    // second array; ViT_Image_Accelerator_Top owns the single physical
    // Systolic instance and feeds these request/data ports during TOP_PATCH.
    output logic                    systolic_start_o,
    input  logic                    systolic_module_ready_i,
    output logic [16:0]             systolic_act_base_addr_o,
    output logic [16:0]             systolic_w_base_addr_o,
    output logic [16:0]             systolic_bias_base_addr_o,
    output logic [7:0]              systolic_k_tile_cnt_o,
    output logic [7:0]              systolic_act_zero_point_o,
    input  logic                    systolic_act_bram_rd_en_i,
    input  logic                    systolic_w_bram_rd_en_i,
    input  logic                    systolic_bias_bram_rd_en_i,
    input  logic [16:0]             systolic_act_bram_addr_i,
    input  logic [16:0]             systolic_w_bram_addr_i,
    input  logic [16:0]             systolic_bias_bram_addr_i,
    output logic                    systolic_w_bram_valid_o,
    output logic [31:0]             systolic_act_bram_data_o,
    output logic [31:0]             systolic_w_bram_data_o,
    output logic [31:0]             systolic_bias_bram_data_o,
    output logic                    systolic_opsum_ready_o,
    input  logic                    systolic_opsum_valid_i,
    input  logic [31:0]             systolic_opsum_i
);

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_CLS_READ,
        ST_CLS_WRITE,
        ST_PACK_IMAGE_REQ,
        ST_PACK_IMAGE,
        ST_SYS_START,
        ST_SYS_RUN,
        ST_SYS_ADVANCE,
        ST_DONE
    } state_t;

    state_t state_q;

    localparam int OPS_PER_TILE = TOKEN_TILE * CHANNEL_TILE;
    localparam [7:0] PATCH_K_TILE_CNT = PATCH_K_TILE_NUM;
    localparam int PATCH_TILE_ACT_ROW_COUNT  = PATCH_K_TILE_NUM * TOKEN_TILE;
    localparam int TOKEN_TILE_LOG2 = $clog2(TOKEN_TILE);
    localparam int CHANNEL_TILE_LOG2 = $clog2(CHANNEL_TILE);
    localparam int TILE_ROW_WORDS = CHANNEL_TILE / 4;
    localparam int TILE_ROW_WORD_LOG2 = $clog2(TILE_ROW_WORDS);
    localparam int TILE_WORDS = TOKEN_TILE * TILE_ROW_WORDS;
    localparam int PATCH_TILE_ACT_WORD_COUNT = PATCH_TILE_ACT_ROW_COUNT * 4;
    localparam int PATCH_ACT_BANKS =
        (PATCH_TILE_ACT_WORD_COUNT + 1023) / 1024;

    integer cls_channel_q;
    integer pack_patch_q;
    integer pack_elem_q;
    logic [15:0] pack_patch_row_q;
    logic [15:0] pack_patch_col_q;
    logic [7:0]  pack_local_y_q;
    logic [7:0]  pack_local_x_q;
    logic [3:0]  pack_local_c_q;
    integer m_tile_q;
    integer n_tile_q;

    logic [8:0] psum_count_q;
    logic sys_opsum_valid_d;
    logic signed [ACC_W-1:0] sys_opsum_d;
    logic [15:0] out_patch_d;
    logic [15:0] out_channel_d;

    logic sys_start_pulse;
    logic sys_module_ready;
    logic [16:0] sys_act_base_addr;
    logic [16:0] sys_w_base_addr;
    logic [16:0] sys_bias_base_addr;
    logic sys_act_rd_en;
    logic sys_w_rd_en;
    logic sys_bias_rd_en;
    logic [16:0] sys_act_addr;
    logic [16:0] sys_w_addr;
    logic [16:0] sys_bias_addr;
    logic [31:0] sys_act_word_q;
    logic sys_opsum_valid;
    logic [31:0] sys_opsum;
    logic [16:0] sys_act_row_addr_comb;
    logic [1:0]  sys_act_word_sel_comb;
    logic [TOKEN_TILE*DATA_W-1:0] sys_act_row_comb;

    integer pack_row_addr_comb;
    integer pack_lane_comb;
    logic patch_act_wr_en;
    logic [16:0] patch_act_wr_addr;
    logic [31:0] patch_act_wr_data;
    logic [3:0] patch_act_wr_byte_en;
    logic [31:0] patch_act_rd_data;
    integer tile_last_patch_comb;

    integer out_row_comb;
    integer out_col_comb;
    integer out_patch_comb;
    integer out_channel_comb;

    assign debug_state_o       = state_q[2:0];
    assign debug_patch_idx_o   = ((state_q == ST_PACK_IMAGE) || (state_q == ST_PACK_IMAGE_REQ)) ? pack_patch_q[15:0] : out_patch_comb[15:0];
    assign debug_channel_idx_o = out_channel_comb[15:0];
    assign debug_elem_idx_o    = pack_elem_q[15:0];

    function automatic integer mul_patch_size;
        input integer value;
        begin
            case (PATCH_SIZE)
                2:       mul_patch_size = value << 1;
                16:      mul_patch_size = value << 4;
                default: mul_patch_size = value * PATCH_SIZE;
            endcase
        end
    endfunction

    function automatic integer mul_img_w;
        input integer value;
        begin
            case (IMG_W)
                2:       mul_img_w = value << 1;
                224:     mul_img_w = (value << 8) - (value << 5);
                default: mul_img_w = value * IMG_W;
            endcase
        end
    endfunction

    function automatic integer mul_img_c;
        input integer value;
        begin
            case (IMG_C)
                1:       mul_img_c = value;
                3:       mul_img_c = (value << 1) + value;
                default: mul_img_c = value * IMG_C;
            endcase
        end
    endfunction

    function automatic integer image_addr_from_counters;
        input integer patch_row;
        input integer patch_col;
        input integer local_y;
        input integer local_x;
        input integer local_c;
        integer image_y;
        integer image_x;
        begin
            image_y = mul_patch_size(patch_row) + local_y;
            image_x = mul_patch_size(patch_col) + local_x;
            image_addr_from_counters =
                mul_img_c(mul_img_w(image_y) + image_x) + local_c;
        end
    endfunction

    function automatic integer mul_384;
        input integer value;
        begin
            mul_384 = (value << 8) + (value << 7);
        end
    endfunction

    function automatic integer embed_offset_for;
        input integer token_idx;
        begin
            case (EMBED_DIM)
                16:      embed_offset_for = token_idx << 4;
                384:     embed_offset_for = mul_384(token_idx);
                default: embed_offset_for = token_idx << 4;
            endcase
        end
    endfunction

    function automatic integer x_word_addr_for;
        input integer token_idx;
        input integer channel_idx;
        integer token_tile_idx;
        integer channel_tile_idx;
        integer row_idx;
        begin
            token_tile_idx   = token_idx >> TOKEN_TILE_LOG2;
            channel_tile_idx = channel_idx >> CHANNEL_TILE_LOG2;

            case (EMBED_TILE_NUM)
                1: begin
                    row_idx = (token_tile_idx << TOKEN_TILE_LOG2) +
                              (channel_tile_idx << TOKEN_TILE_LOG2) +
                              (token_idx & (TOKEN_TILE - 1));
                end
                48: begin
                    row_idx = ((token_tile_idx << 8) + (token_tile_idx << 7)) +
                              (channel_tile_idx << TOKEN_TILE_LOG2) +
                              (token_idx & (TOKEN_TILE - 1));
                end
                default: begin
                    row_idx = (token_tile_idx * EMBED_TILE_NUM * TOKEN_TILE) +
                              (channel_tile_idx << TOKEN_TILE_LOG2) +
                              (token_idx & (TOKEN_TILE - 1));
                end
            endcase

            x_word_addr_for = (row_idx << TILE_ROW_WORD_LOG2) +
                              ((channel_idx >> 2) & (TILE_ROW_WORDS - 1));
        end
    endfunction

    function automatic integer patch_w_tile_words;
        input integer tile_idx;
        begin
            patch_w_tile_words = tile_idx * PATCH_K_TILE_NUM * TILE_WORDS;
        end
    endfunction

    function automatic integer patch_act_row_addr_for;
        input integer patch_idx;
        input integer elem_idx;
        begin
            patch_act_row_addr_for =
                ((elem_idx >> CHANNEL_TILE_LOG2) << TOKEN_TILE_LOG2) +
                (patch_idx & (TOKEN_TILE - 1));
        end
    endfunction

    function automatic logic [DATA_W-1:0] requant_to_u8;
        input logic signed [ACC_W-1:0] value;
        input logic [5:0] shift;
        integer signed shifted;
        begin
            shifted = value >>> shift;
            if (shifted > 127) begin
                requant_to_u8 = 8'd255;
            end
            else if (shifted < -128) begin
                requant_to_u8 = 8'd0;
            end
            else begin
                requant_to_u8 = shifted + OUTPUT_ZP;
            end
        end
    endfunction

    function automatic logic [DATA_W-1:0] add_pos_zp128;
        input logic [DATA_W-1:0] token_q;
        input logic [DATA_W-1:0] pos_q;
        integer sum_q;
        begin
            sum_q = integer'(token_q) + integer'(pos_q) - OUTPUT_ZP;
            if (sum_q < 0) begin
                add_pos_zp128 = 8'd0;
            end
            else if (sum_q > 255) begin
                add_pos_zp128 = 8'd255;
            end
            else begin
                add_pos_zp128 = sum_q[7:0];
            end
        end
    endfunction

    always_comb begin
        out_row_comb     = psum_count_q >> CHANNEL_TILE_LOG2;
        out_col_comb     = psum_count_q & (CHANNEL_TILE - 1);
        out_patch_comb   = (m_tile_q << TOKEN_TILE_LOG2) + out_row_comb;
        out_channel_comb = (n_tile_q << CHANNEL_TILE_LOG2) + out_col_comb;

        image_rd_addr_o = image_addr_from_counters(
            pack_patch_row_q,
            pack_patch_col_q,
            pack_local_y_q,
            pack_local_x_q,
            pack_local_c_q
        );
        cls_addr_o      = cls_channel_q;

        if (state_q == ST_CLS_WRITE) begin
            pos_addr_o = cls_channel_q;
        end
        else begin
            pos_addr_o = embed_offset_for(out_patch_comb + 1) + out_channel_comb;
        end

        tile_last_patch_comb = (m_tile_q << TOKEN_TILE_LOG2) + TOKEN_TILE - 1;
        if (tile_last_patch_comb >= PATCH_COUNT) begin
            tile_last_patch_comb = PATCH_COUNT - 1;
        end

        pack_row_addr_comb = patch_act_row_addr_for(pack_patch_q, pack_elem_q);
        pack_lane_comb     = pack_elem_q & (CHANNEL_TILE - 1);
    end

    always_comb begin
        sys_act_row_addr_comb = sys_act_addr >> TILE_ROW_WORD_LOG2;
        sys_act_word_sel_comb = sys_act_addr[TILE_ROW_WORD_LOG2-1:0];
        sys_act_row_comb = '0;
        sys_act_row_comb[sys_act_word_sel_comb*32 +: 32] = patch_act_rd_data;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            sys_act_word_q <= 32'd0;
        end
        else begin
            sys_act_word_q <= patch_act_rd_data;
        end
    end

    assign patch_w_rd_en_o      = sys_w_rd_en;
    assign patch_bias_rd_en_o   = sys_bias_rd_en;
    // Before the shared systolic starts, expose the base addresses for the
    // next patch weight/bias tile.  Otherwise the host can only see the
    // previous systolic read address and may reload tile 0 while ST_SYS_START
    // is actually waiting for a later embedding-channel tile.
    assign patch_w_addr_o       = (state_q == ST_SYS_START) ? sys_w_base_addr    : sys_w_addr;
    assign patch_bias_addr_o    = (state_q == ST_SYS_START) ? sys_bias_base_addr : sys_bias_addr;
    assign sys_act_base_addr    = 17'd0;
    assign sys_w_base_addr      = patch_w_tile_words(n_tile_q);
    assign sys_bias_base_addr   = n_tile_q << CHANNEL_TILE_LOG2;

    assign systolic_start_o           = sys_start_pulse;
    assign systolic_act_base_addr_o   = sys_act_base_addr;
    assign systolic_w_base_addr_o     = sys_w_base_addr;
    assign systolic_bias_base_addr_o  = sys_bias_base_addr;
    assign systolic_k_tile_cnt_o      = PATCH_K_TILE_CNT;
    assign systolic_act_zero_point_o  = 8'd0;
    assign systolic_w_bram_valid_o    = patch_w_valid_i;
    assign systolic_act_bram_data_o   = patch_act_rd_data;
    assign systolic_w_bram_data_o     = patch_w_data_i;
    assign systolic_bias_bram_data_o  = patch_bias_data_i;

    always_comb begin
        patch_act_wr_en      = 1'b0;
        patch_act_wr_addr    = (pack_row_addr_comb << TILE_ROW_WORD_LOG2) + (pack_lane_comb >> 2);
        patch_act_wr_data    = 32'd0;
        patch_act_wr_byte_en = 4'b0000;

        if ((state_q == ST_PACK_IMAGE) && image_rd_valid_i &&
            (pack_patch_q < PATCH_COUNT)) begin
            patch_act_wr_en = 1'b1;
            patch_act_wr_data[8*(pack_lane_comb & 3) +: 8] = image_rd_data_i;
            patch_act_wr_byte_en[pack_lane_comb & 3] = 1'b1;
        end
    end

    ActivationMem #(
        .INIT_FILE("NONE"),
        .NUM_BANKS(PATCH_ACT_BANKS)
    ) u_patch_act_mem (
        .clk(clk),
        .rst_n(rst_n),
        .rd_en(sys_act_rd_en),
        .rd_addr(sys_act_addr),
        .rd_data(patch_act_rd_data),
        .wr_en(patch_act_wr_en),
        .wr_addr(patch_act_wr_addr),
        .wr_data(patch_act_wr_data),
        .wr_byte_en(patch_act_wr_byte_en)
    );
    assign sys_module_ready = systolic_module_ready_i;
    assign sys_act_rd_en    = systolic_act_bram_rd_en_i;
    assign sys_w_rd_en      = systolic_w_bram_rd_en_i;
    assign sys_bias_rd_en   = systolic_bias_bram_rd_en_i;
    assign sys_act_addr     = systolic_act_bram_addr_i;
    assign sys_w_addr       = systolic_w_bram_addr_i;
    assign sys_bias_addr    = systolic_bias_bram_addr_i;
    assign systolic_opsum_ready_o = 1'b1;
    assign sys_opsum_valid  = systolic_opsum_valid_i;
    assign sys_opsum        = systolic_opsum_i;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q       <= ST_IDLE;
            busy_o        <= 1'b0;
            done_o        <= 1'b0;
            x_buf_we_o    <= 1'b0;
            x_buf_addr_o  <= '0;
            x_buf_wdata_o <= '0;
            x_buf_word_we_o      <= 1'b0;
            x_buf_word_addr_o    <= 17'd0;
            x_buf_word_data_o    <= 32'd0;
            x_buf_word_byte_en_o <= 4'b0000;
            cls_channel_q <= 0;
            pack_patch_q  <= 0;
            pack_elem_q   <= 0;
            pack_patch_row_q <= 16'd0;
            pack_patch_col_q <= 16'd0;
            pack_local_y_q   <= 8'd0;
            pack_local_x_q   <= 8'd0;
            pack_local_c_q   <= 4'd0;
            m_tile_q      <= 0;
            n_tile_q      <= 0;
            psum_count_q  <= '0;
            sys_start_pulse <= 1'b0;
            sys_opsum_valid_d <= 1'b0;
            sys_opsum_d <= '0;
            out_patch_d <= 16'd0;
            out_channel_d <= 16'd0;
        end
        else begin
            done_o          <= 1'b0;
            x_buf_we_o      <= 1'b0;
            x_buf_word_we_o      <= 1'b0;
            x_buf_word_data_o    <= 32'd0;
            x_buf_word_byte_en_o <= 4'b0000;
            sys_start_pulse <= 1'b0;
            sys_opsum_valid_d <= 1'b0;

            // Systolic output is delayed by one cycle so position BRAM data
            // from the same output address is available when X is written.
            if (sys_opsum_valid_d &&
                (out_patch_d < PATCH_COUNT) &&
                (out_channel_d < EMBED_DIM) &&
                pos_valid_i) begin
                x_buf_we_o    <= 1'b1;
                x_buf_addr_o  <= embed_offset_for(out_patch_d + 1) + out_channel_d;
                x_buf_wdata_o <= add_pos_zp128(
                    requant_to_u8(sys_opsum_d, requant_shift_i),
                    pos_data_i
                );
                x_buf_word_we_o   <= 1'b1;
                x_buf_word_addr_o <= x_word_addr_for(out_patch_d + 1, out_channel_d);
                x_buf_word_data_o[((out_channel_d & 3) << 3) +: 8] <= add_pos_zp128(
                    requant_to_u8(sys_opsum_d, requant_shift_i),
                    pos_data_i
                );
                x_buf_word_byte_en_o <= 4'b0001 << (out_channel_d & 3);
            end

            case (state_q)
                ST_IDLE: begin
                    busy_o <= 1'b0;
                    if (start_i) begin
                        busy_o        <= 1'b1;
                        cls_channel_q <= 0;
                        pack_patch_q  <= 0;
                        pack_elem_q   <= 0;
                        pack_patch_row_q <= 16'd0;
                        pack_patch_col_q <= 16'd0;
                        pack_local_y_q   <= 8'd0;
                        pack_local_x_q   <= 8'd0;
                        pack_local_c_q   <= 4'd0;
                        m_tile_q      <= 0;
                        n_tile_q      <= 0;
                        psum_count_q  <= '0;
                        state_q       <= ST_CLS_READ;
                    end
                end

                ST_CLS_READ: begin
                    // Hold cls/pos address for one cycle so synchronous BRAM
                    // returns the requested bytes in ST_CLS_WRITE.
                    state_q <= ST_CLS_WRITE;
                end

                ST_CLS_WRITE: begin
                    if (cls_valid_i && pos_valid_i) begin
                        x_buf_we_o    <= 1'b1;
                        x_buf_addr_o  <= cls_channel_q[ADDR_W-1:0];
                        x_buf_wdata_o <= add_pos_zp128(cls_data_i, pos_data_i);
                        x_buf_word_we_o   <= 1'b1;
                        x_buf_word_addr_o <= x_word_addr_for(0, cls_channel_q);
                        x_buf_word_data_o[((cls_channel_q & 3) << 3) +: 8] <=
                            add_pos_zp128(cls_data_i, pos_data_i);
                        x_buf_word_byte_en_o <= 4'b0001 << (cls_channel_q & 3);

                        if (cls_channel_q == (EMBED_DIM - 1)) begin
                            cls_channel_q <= 0;
                            pack_patch_q  <= 0;
                            pack_elem_q   <= 0;
                            pack_patch_row_q <= 16'd0;
                            pack_patch_col_q <= 16'd0;
                            pack_local_y_q   <= 8'd0;
                            pack_local_x_q   <= 8'd0;
                            pack_local_c_q   <= 4'd0;
                            state_q       <= ST_PACK_IMAGE_REQ;
                        end
                        else begin
                            cls_channel_q <= cls_channel_q + 1;
                            state_q       <= ST_CLS_READ;
                        end
                    end
                end

                ST_PACK_IMAGE_REQ: begin
                    // Hold image address for one cycle before consuming data.
                    state_q <= ST_PACK_IMAGE;
                end

                ST_PACK_IMAGE: begin
                    if (image_rd_valid_i) begin

                        if ((pack_patch_q == tile_last_patch_comb) &&
                            (pack_elem_q == (PATCH_ELEMS - 1))) begin
                            pack_elem_q  <= 0;
                            pack_local_y_q <= 8'd0;
                            pack_local_x_q <= 8'd0;
                            pack_local_c_q <= 4'd0;
                            n_tile_q     <= 0;
                            psum_count_q <= '0;
                            state_q      <= ST_SYS_START;
                        end
                        else if (pack_elem_q == (PATCH_ELEMS - 1)) begin
                            pack_elem_q  <= 0;
                            pack_patch_q <= pack_patch_q + 1;
                            pack_local_y_q <= 8'd0;
                            pack_local_x_q <= 8'd0;
                            pack_local_c_q <= 4'd0;
                            if (pack_patch_col_q == (PATCH_GRID_W - 1)) begin
                                pack_patch_col_q <= 16'd0;
                                pack_patch_row_q <= pack_patch_row_q + 1'b1;
                            end
                            else begin
                                pack_patch_col_q <= pack_patch_col_q + 1'b1;
                            end
                            state_q      <= ST_PACK_IMAGE_REQ;
                        end
                        else begin
                            pack_elem_q <= pack_elem_q + 1;
                            if (pack_local_c_q == (IMG_C - 1)) begin
                                pack_local_c_q <= 4'd0;
                                if (pack_local_x_q == (PATCH_SIZE - 1)) begin
                                    pack_local_x_q <= 8'd0;
                                    if (pack_local_y_q == (PATCH_SIZE - 1)) begin
                                        pack_local_y_q <= 8'd0;
                                    end
                                    else begin
                                        pack_local_y_q <= pack_local_y_q + 1'b1;
                                    end
                                end
                                else begin
                                    pack_local_x_q <= pack_local_x_q + 1'b1;
                                end
                            end
                            else begin
                                pack_local_c_q <= pack_local_c_q + 1'b1;
                            end
                            state_q     <= ST_PACK_IMAGE_REQ;
                        end
                    end
                end

                ST_SYS_START: begin
                    if (sys_module_ready) begin
                        sys_start_pulse <= 1'b1;
                        psum_count_q    <= '0;
                        state_q         <= ST_SYS_RUN;
                    end
                end

                ST_SYS_RUN: begin
                    if (sys_opsum_valid) begin
                        sys_opsum_valid_d <= 1'b1;
                        sys_opsum_d       <= $signed(sys_opsum);
                        out_patch_d       <= out_patch_comb[15:0];
                        out_channel_d     <= out_channel_comb[15:0];

                        if (psum_count_q == (OPS_PER_TILE - 1)) begin
                            psum_count_q <= '0;
                            state_q      <= ST_SYS_ADVANCE;
                        end
                        else begin
                            psum_count_q <= psum_count_q + 1'b1;
                        end
                    end
                end

                ST_SYS_ADVANCE: begin
                    if (n_tile_q == (EMBED_TILE_NUM - 1)) begin
                        n_tile_q <= 0;
                        if (m_tile_q == (PATCH_M_TILE_NUM - 1)) begin
                            state_q <= ST_DONE;
                        end
                        else begin
                            m_tile_q     <= m_tile_q + 1;
                            pack_patch_q <= (m_tile_q + 1) << TOKEN_TILE_LOG2;
                            pack_elem_q  <= 0;
                            pack_local_y_q <= 8'd0;
                            pack_local_x_q <= 8'd0;
                            pack_local_c_q <= 4'd0;
                            if (pack_patch_col_q == (PATCH_GRID_W - 1)) begin
                                pack_patch_col_q <= 16'd0;
                                pack_patch_row_q <= pack_patch_row_q + 1'b1;
                            end
                            else begin
                                pack_patch_col_q <= pack_patch_col_q + 1'b1;
                            end
                            state_q      <= ST_PACK_IMAGE;
                        end
                    end
                    else begin
                        n_tile_q <= n_tile_q + 1;
                        state_q  <= ST_SYS_START;
                    end
                end

                ST_DONE: begin
                    busy_o  <= 1'b0;
                    done_o  <= 1'b1;
                    state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
