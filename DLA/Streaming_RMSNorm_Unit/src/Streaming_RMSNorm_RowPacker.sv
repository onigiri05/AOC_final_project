`timescale 1ns/1ps

module Streaming_RMSNorm_RowPacker #(
    parameter int TOKEN_NUM      = 197,
    parameter int CHANNEL_NUM    = 384,
    parameter int TOKEN_TILE     = 16,
    parameter int CHANNEL_TILE   = 16,
    parameter int DATA_W         = 8,
    parameter int ADDR_W         = 17,

    // 1：把 Streaming RMSNorm 的 signed INT8 輸出轉成 uint8 zero-point 128
    // 0：直接把輸入 8-bit 原樣 pack 起來
    parameter bit SIGNED_TO_ZP128 = 1'b1
)(
    input  logic clk,
    input  logic rst_n,

    // ------------------------------------------------------------
    // control
    // ------------------------------------------------------------
    input  logic start_i,

    // base_addr_i 是 32-bit word address
    input  logic [ADDR_W-1:0] base_addr_i,

    // ------------------------------------------------------------
    // input stream from Streaming_RMSNorm_Unit
    // one 8-bit activation per transfer
    // ------------------------------------------------------------
    input  logic signed [DATA_W-1:0] s_data_i,
    input  logic                     s_valid_i,
    output logic                     s_ready_o,
    input  logic                     s_last_i,

    // ------------------------------------------------------------
    // output write interface to ActivationMem / 32-bit BRAM
    // one word = 4 INT8 activations
    // ------------------------------------------------------------
    output logic                     act_wr_valid_o,
    input  logic                     act_wr_ready_i,
    output logic [ADDR_W-1:0]        act_wr_addr_o,
    output logic [31:0]              act_wr_data_o,
    output logic [3:0]               act_wr_byte_en_o,
    output logic                     act_wr_last_o
);

    localparam int K_TILE_NUM = (CHANNEL_NUM + CHANNEL_TILE - 1) / CHANNEL_TILE;
    localparam int TOKEN_AW   = (TOKEN_NUM <= 1) ? 1 : $clog2(TOKEN_NUM);
    localparam int CHANNEL_AW = (CHANNEL_NUM <= 1) ? 1 : $clog2(CHANNEL_NUM);

    logic [TOKEN_AW-1:0]   token_cnt;
    logic [CHANNEL_AW-1:0] channel_cnt;

    logic [31:0] pack_reg;
    logic [31:0] pack_next;

    logic input_fire;
    logic row_complete;
    logic word_complete;

    logic [3:0] tile_lane;
    logic [1:0] word_lane;

    assign tile_lane = channel_cnt[3:0];
    assign word_lane = channel_cnt[1:0];

    // 當目前有一個 32-bit word 還沒被 BRAM 接收時，先 stall 上游 RMSNorm。
    assign s_ready_o = (!act_wr_valid_o) || act_wr_ready_i;
    assign input_fire = s_valid_i && s_ready_o;

    assign row_complete = (tile_lane == CHANNEL_TILE-1) ||
                          (channel_cnt == CHANNEL_NUM-1);
    assign word_complete = (word_lane == 2'd3) || row_complete;

    // ------------------------------------------------------------
    // signed INT8 -> uint8 zero-point 128
    // -128 -> 0
    // 0    -> 128
    // 127  -> 255
    // ------------------------------------------------------------
    function automatic logic [7:0] convert_to_act_u8;
        input logic signed [7:0] x;
        begin
            if (SIGNED_TO_ZP128) begin
                convert_to_act_u8 = {~x[7], x[6:0]};
            end
            else begin
                convert_to_act_u8 = x[7:0];
            end
        end
    endfunction

    function automatic logic [3:0] byte_mask_for_lane;
        input logic [1:0] lane;
        begin
            case (lane)
                2'd0: byte_mask_for_lane = 4'b0001;
                2'd1: byte_mask_for_lane = 4'b0011;
                2'd2: byte_mask_for_lane = 4'b0111;
                default: byte_mask_for_lane = 4'b1111;
            endcase
        end
    endfunction

    // ------------------------------------------------------------
    // Calculate ActivationMem 32-bit word address for Systolic.v
    //
    // 128-bit row layout:
    //   row = m_tile * (K_TILE_NUM * TOKEN_TILE)
    //       + k_tile * TOKEN_TILE
    //       + m_inner
    //
    // 32-bit word layout:
    //   word_addr = base_addr_i + row * 4 + word_sel
    //   word_sel  = channel_inner[3:2]
    // ------------------------------------------------------------
    function automatic logic [ADDR_W-1:0] calc_act_word_addr;
        input logic [TOKEN_AW-1:0]   token_idx;
        input logic [CHANNEL_AW-1:0] channel_idx;

        int unsigned m_tile;
        int unsigned m_inner;
        int unsigned k_tile;
        int unsigned word_sel;
        int unsigned row_int;
        int unsigned addr_int;
        begin
            m_tile   = int'(token_idx) / TOKEN_TILE;
            m_inner  = int'(token_idx) % TOKEN_TILE;
            k_tile   = int'(channel_idx) / CHANNEL_TILE;
            word_sel = (int'(channel_idx) % CHANNEL_TILE) / 4;

            row_int = m_tile * (K_TILE_NUM * TOKEN_TILE)
                    + k_tile * TOKEN_TILE
                    + m_inner;

            addr_int = int'(base_addr_i) + row_int * 4 + word_sel;
            calc_act_word_addr = addr_int[ADDR_W-1:0];
        end
    endfunction

    always_comb begin
        pack_next = pack_reg;
        pack_next[DATA_W*word_lane +: DATA_W] = convert_to_act_u8(s_data_i);
    end

    // ------------------------------------------------------------
    // Main sequential logic
    // ------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            token_cnt        <= '0;
            channel_cnt      <= '0;
            pack_reg         <= '0;

            act_wr_valid_o   <= 1'b0;
            act_wr_addr_o    <= '0;
            act_wr_data_o    <= 32'd0;
            act_wr_byte_en_o <= 4'b0000;
            act_wr_last_o    <= 1'b0;
        end
        else begin
            // start_i 代表新一個 RMSNorm stream 開始
            if (start_i) begin
                token_cnt        <= '0;
                channel_cnt      <= '0;
                pack_reg         <= '0;

                act_wr_valid_o   <= 1'b0;
                act_wr_addr_o    <= '0;
                act_wr_data_o    <= 32'd0;
                act_wr_byte_en_o <= 4'b0000;
                act_wr_last_o    <= 1'b0;
            end
            else begin
                // BRAM 接收目前 word
                if (act_wr_valid_o && act_wr_ready_i) begin
                    act_wr_valid_o   <= 1'b0;
                    act_wr_byte_en_o <= 4'b0000;
                    act_wr_last_o    <= 1'b0;
                end

                if (input_fire) begin
                    pack_reg <= pack_next;

                    // 收滿 4 個 channel，或遇到 row/stream 尾端，形成一個 32-bit word。
                    if (word_complete) begin
                        act_wr_valid_o   <= 1'b1;
                        act_wr_data_o    <= pack_next;
                        act_wr_byte_en_o <= byte_mask_for_lane(word_lane);
                        act_wr_addr_o    <= calc_act_word_addr(token_cnt, channel_cnt);
                        act_wr_last_o    <= s_last_i;

                        // 下一個 word 重新 pack
                        pack_reg <= 32'd0;
                    end

                    // update token/channel counters
                    if (channel_cnt == CHANNEL_NUM-1) begin
                        channel_cnt <= '0;

                        if (token_cnt == TOKEN_NUM-1) begin
                            token_cnt <= '0;
                        end
                        else begin
                            token_cnt <= token_cnt + 1'b1;
                        end
                    end
                    else begin
                        channel_cnt <= channel_cnt + 1'b1;
                    end
                end
            end
        end
    end

endmodule
