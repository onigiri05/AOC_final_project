`timescale 1ns/1ps
`ifndef STREAMING_RMSNORM_DEFS_SVH
`define STREAMING_RMSNORM_DEFS_SVH

// ============================================================
// 可調整規格
// ============================================================
`define RMS_TOKEN_NUM       197
`define RMS_CHANNEL_NUM     384

`define RMS_X_W             8
`define RMS_SCALE_W         16

// fixed-point 小數位數
// 目前暫定 inv_rms / gamma 都是 16-bit fixed-point with FRAC=14
// 實際 FRAC 要等 software calibration 決定
`define RMS_FRAC            14

// 額外 output scaling shift
// 若 software 決定 RMSNorm 後還要再縮放，可調整這個值
`define RMS_OUT_SHIFT       0

`endif


// ============================================================
// Module: Streaming_RMSNorm_Core
//
// 功能：
//   對單一 activation element 做 RMSNorm 運算：
//
//   y = clamp_int8((x_in * inv_rms_in * gamma_in) >>> SHIFT)
//
// 資料格式暫定：
//   x_in        : signed INT8
//   inv_rms_in  : unsigned 16-bit fixed-point，目前暫定 Q?.14
//   gamma_in    : signed 16-bit fixed-point，目前暫定 Q?.14
//
// 注意：
//   inv_rms 通常是正數，所以這裡用 unsigned input。
//   gamma 是 learnable parameter，保留 signed 格式比較安全。
//
// Pipeline:
//   Stage 1: x * inv_rms
//   Stage 2: result * gamma
//   Stage 3: shift + clamp
// ============================================================

module Streaming_RMSNorm_Core #(
    parameter int X_W       = `RMS_X_W,
    parameter int SCALE_W   = `RMS_SCALE_W,
    parameter int FRAC      = `RMS_FRAC,
    parameter int OUT_SHIFT = `RMS_OUT_SHIFT
)(
    input  logic clk,
    input  logic rst_n,

    // ----------------------------
    // input stream
    // ----------------------------
    input  logic                         in_valid,
    output logic                         in_ready,
    input  logic                         in_last,

    input  logic signed [X_W-1:0]         x_in,
    input  logic        [SCALE_W-1:0]     inv_rms_in,
    input  logic signed [SCALE_W-1:0]     gamma_in,

    // ----------------------------
    // output stream
    // ----------------------------
    output logic                         out_valid,
    input  logic                         out_ready,
    output logic                         out_last,

    output logic signed [X_W-1:0]         y_out
);

    localparam int INV_EXT_W = SCALE_W + 1;
    localparam int PROD1_W   = X_W + INV_EXT_W;
    localparam int PROD2_W   = PROD1_W + SCALE_W;
    localparam int SHIFT     = 2 * FRAC + OUT_SHIFT;

    // INT8 max/min sign-extend 到 PROD2_W 方便比較
    localparam logic signed [PROD2_W-1:0] INT8_MAX_EXT =
        {{(PROD2_W-X_W){1'b0}}, 1'b0, {(X_W-1){1'b1}}};

    localparam logic signed [PROD2_W-1:0] INT8_MIN_EXT =
        {{(PROD2_W-X_W){1'b1}}, 1'b1, {(X_W-1){1'b0}}};

    // ------------------------------------------------------------
    // Pipeline registers
    // ------------------------------------------------------------
    logic vld_s1, vld_s2, vld_s3;
    logic last_s1, last_s2, last_s3;

    logic signed [PROD1_W-1:0] prod1_s1;
    logic signed [SCALE_W-1:0] gamma_s1;

    logic signed [PROD2_W-1:0] prod2_s2;
    logic signed [X_W-1:0]     y_s3;

    // ------------------------------------------------------------
    // Combinational multiply path
    // ------------------------------------------------------------
    logic signed [INV_EXT_W-1:0] inv_rms_signed;
    logic signed [PROD1_W-1:0]   prod1_next;
    logic signed [PROD2_W-1:0]   prod2_next;
    logic signed [PROD2_W-1:0]   scaled_next;

    logic pipe_advance;

    // inv_rms 是 unsigned，前面補 0 轉成 signed positive
    assign inv_rms_signed = $signed({1'b0, inv_rms_in});

    assign prod1_next  = $signed(x_in) * inv_rms_signed;
    assign prod2_next  = prod1_s1 * $signed(gamma_s1);
    assign scaled_next = prod2_s2 >>> SHIFT;

    // 如果最後一級是空的，或下游願意接收，pipeline 可以前進
    assign pipe_advance = (~vld_s3) | out_ready;

    assign in_ready  = pipe_advance;
    assign out_valid = vld_s3;
    assign out_last  = last_s3;
    assign y_out     = y_s3;

    // ------------------------------------------------------------
    // clamp to signed INT8
    // ------------------------------------------------------------
    function automatic logic signed [X_W-1:0] clamp_int8;
        input logic signed [PROD2_W-1:0] v;
        begin
            if (v > INT8_MAX_EXT)
                clamp_int8 = {1'b0, {(X_W-1){1'b1}}};   // +127 when X_W=8
            else if (v < INT8_MIN_EXT)
                clamp_int8 = {1'b1, {(X_W-1){1'b0}}};   // -128 when X_W=8
            else
                clamp_int8 = v[X_W-1:0];
        end
    endfunction

    // ------------------------------------------------------------
    // 3-stage pipeline
    // ------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vld_s1   <= 1'b0;
            vld_s2   <= 1'b0;
            vld_s3   <= 1'b0;

            last_s1  <= 1'b0;
            last_s2  <= 1'b0;
            last_s3  <= 1'b0;

            prod1_s1 <= '0;
            gamma_s1 <= '0;
            prod2_s2 <= '0;
            y_s3     <= '0;
        end
        else begin
            if (pipe_advance) begin
                // ----------------------------
                // Stage 3: shift + clamp
                // ----------------------------
                vld_s3  <= vld_s2;
                last_s3 <= last_s2;

                if (vld_s2) begin
                    y_s3 <= clamp_int8(scaled_next);
                end

                // ----------------------------
                // Stage 2: prod1 * gamma
                // ----------------------------
                vld_s2  <= vld_s1;
                last_s2 <= last_s1;

                if (vld_s1) begin
                    prod2_s2 <= prod2_next;
                end

                // ----------------------------
                // Stage 1: x * inv_rms
                // ----------------------------
                vld_s1  <= in_valid;
                last_s1 <= in_last;

                if (in_valid) begin
                    prod1_s1 <= prod1_next;
                    gamma_s1 <= gamma_in;
                end
            end
        end
    end

endmodule


// ============================================================
// Module: Streaming_RMSNorm_Unit
//
// 功能：
//   外層控制 module。
//   負責：
//     1. token counter
//     2. channel counter
//     3. Token Stat SRAM address
//     4. Gamma Buffer address
//     5. start / done 控制
//     6. valid / ready handshake
//
// 資料順序：
//   token 0, channel 0~383
//   token 1, channel 0~383
//   ...
//   token 196, channel 0~383
//
// address:
//   inv_rms_addr = token_cnt
//   gamma_addr   = channel_cnt
//
// 注意：
//   這一版假設 inv_rms_data / gamma_data 已經和 x_in 對齊。
//   如果未來 Token Stat SRAM / Gamma Buffer 是同步 BRAM，
//   則需要在外層多加 1-cycle pipeline 對齊 x_in 和 data。
// ============================================================

module Streaming_RMSNorm_Unit #(
    parameter int TOKEN_NUM   = `RMS_TOKEN_NUM,
    parameter int CHANNEL_NUM = `RMS_CHANNEL_NUM,

    parameter int TOKEN_AW    = $clog2(TOKEN_NUM),
    parameter int CHANNEL_AW  = $clog2(CHANNEL_NUM),

    parameter int X_W         = `RMS_X_W,
    parameter int SCALE_W     = `RMS_SCALE_W,
    parameter int FRAC        = `RMS_FRAC,
    parameter int OUT_SHIFT   = `RMS_OUT_SHIFT
)(
    input  logic clk,
    input  logic rst_n,

    // ----------------------------
    // control
    // ----------------------------
    input  logic start,
    output logic busy,
    output logic done,

    // ----------------------------
    // input activation stream
    // 來源：Activation-Residual Buffer / Global Buffer
    // ----------------------------
    input  logic                         x_valid,
    output logic                         x_ready,
    input  logic signed [X_W-1:0]         x_in,

    // ----------------------------
    // Token Stat SRAM read port
    // 存 inv_rms[t]
    // ----------------------------
    output logic [TOKEN_AW-1:0]           inv_rms_addr,
    input  logic        [SCALE_W-1:0]     inv_rms_data,

    // ----------------------------
    // Gamma Buffer read port
    // 存 gamma[c]
    // ----------------------------
    output logic [CHANNEL_AW-1:0]         gamma_addr,
    input  logic signed [SCALE_W-1:0]     gamma_data,

    // ----------------------------
    // output normalized activation stream
    // 目的地：Activation FIFO
    // ----------------------------
    output logic                         y_valid,
    input  logic                         y_ready,
    output logic                         y_last,
    output logic signed [X_W-1:0]         y_out
);

    logic running;
    logic input_done;

    logic [TOKEN_AW-1:0]   token_cnt;
    logic [CHANNEL_AW-1:0] channel_cnt;

    logic core_in_valid;
    logic core_in_ready;
    logic core_in_last;

    logic core_out_valid;
    logic core_out_ready;
    logic core_out_last;

    logic accept_input;
    logic last_input;

    assign busy = running;

    assign inv_rms_addr = token_cnt;
    assign gamma_addr   = channel_cnt;

    assign core_in_valid = running & (~input_done) & x_valid;
    assign x_ready       = running & (~input_done) & core_in_ready;

    assign accept_input = core_in_valid & core_in_ready;

    assign last_input =
        (token_cnt == TOKEN_NUM-1) &&
        (channel_cnt == CHANNEL_NUM-1);

    assign core_in_last  = last_input;
    assign core_out_ready = y_ready;

    assign y_valid = core_out_valid;
    assign y_last  = core_out_last;

    Streaming_RMSNorm_Core #(
        .X_W(X_W),
        .SCALE_W(SCALE_W),
        .FRAC(FRAC),
        .OUT_SHIFT(OUT_SHIFT)
    ) u_core (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(core_in_valid),
        .in_ready(core_in_ready),
        .in_last(core_in_last),

        .x_in(x_in),
        .inv_rms_in(inv_rms_data),
        .gamma_in(gamma_data),

        .out_valid(core_out_valid),
        .out_ready(core_out_ready),
        .out_last(core_out_last),

        .y_out(y_out)
    );

    // ------------------------------------------------------------
    // token / channel counter
    //
    // 只有 valid && ready 成立，也就是真的吃進一筆 x_in 時，
    // counter 才會前進。
    // ------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            running     <= 1'b0;
            input_done  <= 1'b0;
            done        <= 1'b0;
            token_cnt   <= '0;
            channel_cnt <= '0;
        end
        else begin
            done <= 1'b0;

            // start 只在 idle 時接受
            if (start && !running) begin
                running     <= 1'b1;
                input_done  <= 1'b0;
                token_cnt   <= '0;
                channel_cnt <= '0;
            end

            // 接收一筆 input element
            if (accept_input) begin
                if (last_input) begin
                    input_done <= 1'b1;
                end

                if (channel_cnt == CHANNEL_NUM-1) begin
                    channel_cnt <= '0;

                    if (token_cnt == TOKEN_NUM-1)
                        token_cnt <= '0;
                    else
                        token_cnt <= token_cnt + 1'b1;
                end
                else begin
                    channel_cnt <= channel_cnt + 1'b1;
                end
            end

            // 最後一筆 output 被下游接收，整個 RMSNorm 完成
            if (core_out_valid && core_out_ready && core_out_last) begin
                running    <= 1'b0;
                input_done <= 1'b0;
                done       <= 1'b1;
            end
        end
    end

endmodule
