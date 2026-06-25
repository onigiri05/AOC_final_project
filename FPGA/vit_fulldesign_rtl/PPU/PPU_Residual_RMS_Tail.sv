`timescale 1ns/1ps

// ============================================================
// 檔名  : PPU_Residual_RMS_Tail.sv
// 模組  : PPU_Residual_RMS_Tail
// 功能  : 接在 Requant_Unit 後面的 PPU tail
//         依照 mode 做 Residual Add / FC1 bypass，並在產生 X_mid/X_out 時累加 RMS statistic。
//
// 對應 project PPU dataflow：
//   ppu_mode_i = 2'b00：Attention output phase
//       main_tile_i     = Requant 後的 Attention output O
//       residual_tile_i = X
//       data_tile_o     = X_mid = X + O
//       RMS Stat Accumulator 啟用，累加 Σ(X_mid - 128)^2
//
//   ppu_mode_i = 2'b01：FFN FC1 phase
//       main_tile_i     = GELU + Requant 後的 FC1 output
//       residual_tile_i = don't care
//       data_tile_o     = main_tile_i，直接送 FC2
//       RMS Stat Accumulator 不啟用
//
//   ppu_mode_i = 2'b10：FFN FC2 phase
//       main_tile_i     = Requant 後的 MLP_out
//       residual_tile_i = X_mid
//       data_tile_o     = X_out = X_mid + MLP_out
//       RMS Stat Accumulator 啟用，累加 Σ(X_out - 128)^2
//
// Tile 排列：
//   data[(row*CHANNEL_TILE + col)*8 +: 8]
//   row = token row，0~15
//   col = output channel，0~15
// ============================================================

module PPU_Residual_RMS_Tail #(
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

    // 00: Attention output, 01: FFN FC1, 10: FFN FC2
    input  logic [1:0] ppu_mode_i,

    // -------------------------
    // Tile input handshake
    // -------------------------
    input  logic tile_valid_i,
    output logic tile_ready_o,

    // Requant 後的 main branch tile
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] main_tile_i,

    // Residual shortcut tile，Attention/FC2 mode 使用
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_tile_i,

    // 給 RMS_Stat_Accumulator 判斷 token / channel 位置
    input  logic [TOKEN_W-1:0]        base_token_idx_i,
    input  logic [CHANNEL_TILE_W-1:0] channel_tile_idx_i,
    input  logic [TOKEN_TILE-1:0]     token_valid_mask_i,

    // -------------------------
    // Tile output handshake
    // -------------------------
    output logic data_tile_valid_o,
    input  logic data_tile_ready_i,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o,

    // -------------------------
    // Statistic output handshake
    // -------------------------
    output logic stat_valid_o,
    input  logic stat_ready_i,
    output logic [TOKEN_W-1:0] stat_token_idx_o,
    output logic [SUM_W-1:0]   sum_sq_o
);

    localparam logic [1:0] PPU_MODE_ATTN = 2'b00;
    localparam logic [1:0] PPU_MODE_FC1  = 2'b01;
    localparam logic [1:0] PPU_MODE_FC2  = 2'b10;

    logic residual_mode;
    logic rms_acc_en;
    logic output_stage_ready;
    logic input_fire;

    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_add_tile;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] selected_tile;

    logic rms_tile_valid;
    logic rms_tile_ready;

    assign residual_mode      = (ppu_mode_i == PPU_MODE_ATTN) || (ppu_mode_i == PPU_MODE_FC2);
    assign rms_acc_en         = residual_mode;
    assign output_stage_ready = (!data_tile_valid_o) || data_tile_ready_i;

    // 本 wrapper 只有在 output register 有空，且 RMS accumulator 可接收時，才吃新的 tile。
    assign tile_ready_o = output_stage_ready && ((!rms_acc_en) || rms_tile_ready);
    assign input_fire   = tile_valid_i && tile_ready_o;

    // ------------------------------------------------------------
    // Stage 1：Residual Add Unit
    //   Attention / FC2 mode 使用。
    //   FC1 mode 不使用 residual_add_tile，直接 bypass main_tile_i。
    // ------------------------------------------------------------
    Residual_Add_Unit #(
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .ZERO_POINT(ZERO_POINT)
    ) u_Residual_Add_Unit (
        .main_tile_i(main_tile_i),
        .residual_tile_i(residual_tile_i),
        .data_tile_o(residual_add_tile)
    );

    always_comb begin
        if (residual_mode) begin
            selected_tile = residual_add_tile;
        end
        else begin
            // FC1：GELU + Requant 後直接送 FC2，不做 shortcut add。
            selected_tile = main_tile_i;
        end
    end

    // ------------------------------------------------------------
    // Stage 2：Output register
    //   將 X_mid / FC1 output / X_out 送回 GLB 或下一級 buffer。
    // ------------------------------------------------------------
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            data_tile_valid_o <= 1'b0;
            data_tile_o       <= {(TOKEN_TILE*CHANNEL_TILE*DATA_W){1'b0}};
        end
        else begin
            if (output_stage_ready) begin
                data_tile_valid_o <= input_fire;
                if (input_fire) begin
                    data_tile_o <= selected_tile;
                end
            end
        end
    end

    // ------------------------------------------------------------
    // Stage 3：RMS Stat Accumulator
    //   只在 Attention / FC2 mode 對 X_mid / X_out 累加 Σx²。
    //   FC1 mode 不累加。
    // ------------------------------------------------------------
    assign rms_tile_valid = tile_valid_i && output_stage_ready && rms_acc_en;

    RMS_Stat_Accumulator #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ZERO_POINT(ZERO_POINT)
    ) u_RMS_Stat_Accumulator (
        .clk(clk),
        .rst_n(rst_n),
        .tile_valid_i(rms_tile_valid),
        .tile_ready_o(rms_tile_ready),
        .acc_en_i(rms_acc_en),
        .data_tile_i(selected_tile),
        .base_token_idx_i(base_token_idx_i),
        .channel_tile_idx_i(channel_tile_idx_i),
        .token_valid_mask_i(token_valid_mask_i),
        .stat_valid_o(stat_valid_o),
        .stat_ready_i(stat_ready_i),
        .stat_token_idx_o(stat_token_idx_o),
        .sum_sq_o(sum_sq_o)
    );

endmodule
