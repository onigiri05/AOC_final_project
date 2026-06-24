`timescale 1ns/1ps

// ============================================================
// 檔名  : Residual_Add_Unit.sv
// 模組  : Residual_Add_Unit
// 功能  : 16x16 tile 版本的 shortcut / residual add
//
// 對應 project dataflow：
//   1. Attention 後：X_mid = X + O
//      main_tile_i     = Requant 後的 Attention output O
//      residual_tile_i = Activation-Residual Buffer 裡保留的 X
//
//   2. FC2 後：X_out = X_mid + MLP_out
//      main_tile_i     = Requant 後的 MLP_out
//      residual_tile_i = Residual Buffer 裡保留的 X_mid
//
// 資料格式：
//   根據目前 Requant_Unit，輸出是 uint8 且 zero point = 128。
//   因此 8'd128 代表真實值 0。
//
// 公式：
//   q_out = clamp(q_main + q_residual - 128, 0, 255)
//
// Tile 排列：
//   data[(row*CHANNEL_TILE + col)*8 +: 8]
//   row = token row，0~15
//   col = output channel row，0~15
// ============================================================

module Residual_Add_Unit #(
    parameter int TOKEN_TILE   = 16,
    parameter int CHANNEL_TILE = 16,
    parameter int DATA_W       = 8,
    parameter logic [7:0] ZERO_POINT = 8'd128
)(
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] main_tile_i,
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_tile_i,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o
);

    localparam int TILE_ELEMS = TOKEN_TILE * CHANNEL_TILE;

    integer idx;

    // ------------------------------------------------------------
    // 單一 INT8 element 的 residual add。
    // 這裡不使用 signed INT8 解讀，因為目前 project 的 Requant_Unit
    // 已經把 signed value 轉成 zero-point=128 的 uint8 格式。
    // ------------------------------------------------------------
    function automatic logic [7:0] add_u8_zp128_clamp;
        input logic [7:0] main_q;
        input logic [7:0] residual_q;
        logic signed [9:0] sum_q;
        begin
            // main_q 與 residual_q 都是 0~255 的 uint8。
            // 兩個 zero-point activation 相加時，zero point 會被加兩次，
            // 所以要扣掉一次 ZERO_POINT。
            sum_q = $signed({1'b0, main_q})
                  + $signed({1'b0, residual_q})
                  - $signed({2'b00, ZERO_POINT});

            // saturation / clamp 到 uint8 範圍。
            if (sum_q < 10'sd0) begin
                add_u8_zp128_clamp = 8'd0;
            end
            else if (sum_q > 10'sd255) begin
                add_u8_zp128_clamp = 8'd255;
            end
            else begin
                add_u8_zp128_clamp = sum_q[7:0];
            end
        end
    endfunction

    // ------------------------------------------------------------
    // 16x16 tile 內的 256 個 elements 平行做 residual add。
    // 這個 module 是 combinational，方便接在 Requant 後面。
    // ------------------------------------------------------------
    always_comb begin
        for (idx = 0; idx < TILE_ELEMS; idx = idx + 1) begin
            data_tile_o[idx*DATA_W +: DATA_W] = add_u8_zp128_clamp(
                main_tile_i[idx*DATA_W +: DATA_W],
                residual_tile_i[idx*DATA_W +: DATA_W]
            );
        end
    end

endmodule
