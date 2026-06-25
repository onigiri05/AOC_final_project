`timescale 1ns/1ps

// ============================================================
// 檔名  : RMS_Stat_Accumulator.sv
// 模組  : RMS_Stat_Accumulator
// 功能  : 16x16 tile 版本 RMS statistic accumulator
//
// 對應 project dataflow：
//   Systolic Array / PPU 一次輸出 [16 tokens x 16 channels] tile。
//   你們目前 FC2 / projection dataflow 是 output-channel tile 為外層，
//   所以同一個 token 的 384 channels 不是連續 384 cycle 進來，
//   而是分成 24 個 channel tile：
//
//      channel_tile_idx = 0  : channel 0~15
//      channel_tile_idx = 1  : channel 16~31
//      ...
//      channel_tile_idx = 23 : channel 368~383
//
//   因此本模組使用 partial_sum_mem[token] 保存每個 token 的中途累加值。
//
// RMSNorm 需要的統計量：
//   sum_sq[t] = Σ x[t,c]^2, c = 0~383
//
// 因為 activation 是 uint8, zero point = 128：
//   centered_x = q - 128
//   sum_sq[t] += centered_x * centered_x
//
// 輸出行為：
//   當 channel_tile_idx_i == 23 時，代表該 tile 的 16 個 token row
//   都已經看完最後 16 個 channels。
//   本模組會把最多 16 個 token 的完整 sum_sq 暫存在 pending queue，
//   再用 stat_valid_o/stat_ready_i 一筆一筆輸出給 Token Stat SRAM
//   或後面的 inv-sqrt LUT / quantization unit。
//
// 注意：
//   sum_sq_o 是 32-bit raw Σx²，不是 8-bit inv_rms。
//   若你的 Token Stat SRAM 最後只存 8-bit，後面還需要接 LUT/近似轉換。
// ============================================================

module RMS_Stat_Accumulator #(
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
    input  logic rst_n,

    // -------------------------
    // Tile input handshake
    // -------------------------
    input  logic tile_valid_i,
    output logic tile_ready_o,

    // acc_en_i = 1：累加 X_mid / X_out 的 Σx²
    // acc_en_i = 0：例如 FC1 bypass 階段，不累加 statistic
    input  logic acc_en_i,

    // 16x16 output activation tile，已經是 residual add 後的 X_mid / X_out
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_i,

    // base_token_idx_i 是此 tile 第 0 row 對應的 token index。
    // 例如 m_tile=0 時 base=0，m_tile=12 時 base=192。
    input  logic [TOKEN_W-1:0] base_token_idx_i,

    // 目前是第幾個 output-channel tile。
    // ViT-Small D=384, CHANNEL_TILE=16，所以範圍是 0~23。
    input  logic [CHANNEL_TILE_W-1:0] channel_tile_idx_i,

    // 有效 token row mask。
    // 最後一個 token tile 只有 token 192~196 有效，所以通常 mask = 16'h001F。
    input  logic [TOKEN_TILE-1:0] token_valid_mask_i,

    // -------------------------
    // Statistic output handshake
    // -------------------------
    output logic stat_valid_o,
    input  logic stat_ready_i,
    output logic [TOKEN_W-1:0] stat_token_idx_o,
    output logic [SUM_W-1:0]   sum_sq_o
);

    localparam int NUM_CHANNEL_TILES = CHANNEL_NUM / CHANNEL_TILE;

    // 每個 token 一個 partial sum。
    // 實際硬體可以視資源改成 SRAM / BRAM；此處先用 RTL array 描述行為。
    logic [SUM_W-1:0] partial_sum_mem [0:TOKEN_NUM-1];

    // 最後一個 channel tile 到來時，會一次完成最多 16 個 token 的 sum_sq。
    // 但 stat output 是單筆握手介面，所以先放到 pending queue 再逐筆送出。
    logic [SUM_W-1:0]   pending_sum   [0:TOKEN_TILE-1];
    logic [TOKEN_W-1:0] pending_token [0:TOKEN_TILE-1];
    logic [TOKEN_TILE-1:0] pending_valid;

    // row_sum[r] = 目前 16-channel tile 中，第 r 個 token row 的平方和。
    logic [SUM_W-1:0] row_sum [0:TOKEN_TILE-1];

    logic pending_busy;
    logic input_fire;
    logic last_channel_tile;

    integer r;
    integer c;
    integer rr;
    integer pp;
    integer token_idx_int;
    logic pop_found;

    assign pending_busy      = |pending_valid;
    assign last_channel_tile = (channel_tile_idx_i == (NUM_CHANNEL_TILES - 1));

    // 當 pending queue 還沒送完，或 stat output 正被 backpressure 卡住時，
    // 先不要接下一個 tile，避免 statistic 覆蓋。
    assign tile_ready_o = (!pending_busy) && ((!stat_valid_o) || stat_ready_i);
    assign input_fire   = tile_valid_i && tile_ready_o;

    // ------------------------------------------------------------
    // 單一 uint8 activation 轉 centered value 後平方。
    // ------------------------------------------------------------
    function automatic logic [SUM_W-1:0] square_u8_zp128;
        input logic [7:0] q;
        logic signed [9:0]  centered_x;
        logic signed [19:0] square_x;
        begin
            centered_x = $signed({1'b0, q}) - $signed({2'b00, ZERO_POINT});
            square_x   = centered_x * centered_x;
            square_u8_zp128 = {{(SUM_W-20){1'b0}}, square_x[19:0]};
        end
    endfunction

    // ------------------------------------------------------------
    // 對 16x16 tile 的每一個 token row，先算出 16 個 channels 的平方和。
    // row_sum[r] 還不是完整 384 channels 的 sum，只是目前 channel tile 的貢獻。
    // ------------------------------------------------------------
    always_comb begin
        for (r = 0; r < TOKEN_TILE; r = r + 1) begin
            row_sum[r] = {SUM_W{1'b0}};
            for (c = 0; c < CHANNEL_TILE; c = c + 1) begin
                row_sum[r] = row_sum[r] + square_u8_zp128(
                    data_tile_i[(r*CHANNEL_TILE + c)*DATA_W +: DATA_W]
                );
            end
        end
    end

    // ------------------------------------------------------------
    // 主時序邏輯：
    //   1. 接收 tile 並更新 partial_sum_mem
    //   2. 若是最後 channel tile，產生 pending statistic
    //   3. 將 pending statistic 逐筆送出
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            stat_valid_o     <= 1'b0;
            stat_token_idx_o <= {TOKEN_W{1'b0}};
            sum_sq_o         <= {SUM_W{1'b0}};
            pending_valid    <= {TOKEN_TILE{1'b0}};

            for (rr = 0; rr < TOKEN_NUM; rr = rr + 1) begin
                partial_sum_mem[rr] <= {SUM_W{1'b0}};
            end
            for (rr = 0; rr < TOKEN_TILE; rr = rr + 1) begin
                pending_sum[rr]   <= {SUM_W{1'b0}};
                pending_token[rr] <= {TOKEN_W{1'b0}};
            end
        end
        else begin
            // ----------------------------------------------------
            // Step A：若 stat output 空著，或目前這筆已被接走，
            // 就從 pending queue 取下一筆 statistic 出來。
            // ----------------------------------------------------
            if ((!stat_valid_o) || stat_ready_i) begin
                stat_valid_o <= 1'b0;
                pop_found = 1'b0;

                for (pp = 0; pp < TOKEN_TILE; pp = pp + 1) begin
                    if ((!pop_found) && pending_valid[pp]) begin
                        stat_valid_o     <= 1'b1;
                        stat_token_idx_o <= pending_token[pp];
                        sum_sq_o         <= pending_sum[pp];
                        pending_valid[pp] <= 1'b0;
                        pop_found = 1'b1;
                    end
                end
            end

            // ----------------------------------------------------
            // Step B：接收新的 16x16 tile。
            // acc_en_i=0 時只吃掉 tile，不更新 statistic。
            // ----------------------------------------------------
            if (input_fire && acc_en_i) begin
                for (rr = 0; rr < TOKEN_TILE; rr = rr + 1) begin
                    token_idx_int = base_token_idx_i + rr;

                    if (token_valid_mask_i[rr] && (token_idx_int < TOKEN_NUM)) begin
                        if (last_channel_tile) begin
                            // 這是最後 16 個 channels，完整 sum_sq 完成。
                            pending_sum[rr]   <= partial_sum_mem[token_idx_int] + row_sum[rr];
                            pending_token[rr] <= token_idx_int[TOKEN_W-1:0];
                            pending_valid[rr] <= 1'b1;

                            // 該 token 已經完成，partial sum 清零，準備下一次 block 使用。
                            partial_sum_mem[token_idx_int] <= {SUM_W{1'b0}};
                        end
                        else begin
                            // 還沒到最後 channel tile，先存在 partial_sum_mem。
                            partial_sum_mem[token_idx_int] <= partial_sum_mem[token_idx_int] + row_sum[rr];
                        end
                    end
                end
            end
        end
    end

endmodule
