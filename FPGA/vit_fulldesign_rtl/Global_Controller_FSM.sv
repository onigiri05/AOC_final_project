`timescale 1ns/1ps

// ============================================================
// Module: Global_Controller_FSM
// Function:
//   High-level ViT block scheduler. FlashAttention has been removed; the
//   attention path is now explicit: QKV -> QK^T -> Softmax -> A*V -> OutProj.
//
//   The top-level owns tile loops, buffers, and datapath handshakes. This
//   controller starts one architectural phase at a time and waits for the top
//   to report phase_done_i.
//
// 中文說明：
//   這個 FSM 只負責「phase 順序」與每個 phase 的基本設定。
//   實際 tile loop、buffer 寫回、PPU/stat 等等待條件都在
//   ViT_Accelerator_Top.sv 裡處理。
// ============================================================
module Global_Controller_FSM #(
    parameter logic [16:0] W_QKV_BASE      = 17'h00000,
    parameter logic [16:0] B_QKV_BASE      = 17'h00000,
    parameter logic [16:0] W_OUT_BASE      = 17'h02000,
    parameter logic [16:0] B_OUT_BASE      = 17'h00100,
    parameter logic [16:0] W_FC1_BASE      = 17'h04000,
    parameter logic [16:0] B_FC1_BASE      = 17'h00200,
    parameter logic [16:0] W_FC2_BASE      = 17'h08000,
    parameter logic [16:0] B_FC2_BASE      = 17'h00300,

    parameter logic [5:0]  SCALE_QKV       = 6'd2,
    parameter logic [5:0]  SCALE_ATTN_V    = 6'd2,
    parameter logic [5:0]  SCALE_OUT_PROJ  = 6'd1,
    parameter logic [5:0]  SCALE_FC1       = 6'd0,
    parameter logic [5:0]  SCALE_FC2       = 6'd1,

    // 這四個 tile 數不可寫死，one-tile TB 會縮成 1，
    // 完整 ViT-Small 則分別是 24/4/13/96。
    parameter int CHANNEL_TILE_NUM         = 24,
    parameter int HEAD_DIM_TILE_NUM        = 4,
    parameter int SCORE_TILE_NUM           = 13,
    parameter int FFN_CHANNEL_TILE_NUM     = 96
)(
    input  logic        clk,
    input  logic        rst_n,

    input  logic        start_exec,
    input  logic        phase_done_i,
    // ViT top 若只配置單 head 的 score/A BRAM，會在 ATTN_V 完成後要求
    // controller 回到 QK^T，繼續處理下一個 head。
    input  logic        mhsa_repeat_i,

    output logic        busy_exec,
    output logic        done_exec,

    output logic [4:0]  phase_o,
    output logic        phase_start_o,

    // Legacy/debug decoded pulses.
    // 可接出到 ILA，快速判斷 controller 是否有發出 phase start。
    output logic        rms_start,
    output logic        sys_en,
    output logic        softmax_start,

    // Phase configuration consumed by ViT_Accelerator_Top.
    // sys_k_tile_cnt 會交給 Systolic，ppu_mode/scale 會交給 PPU。
    // 新版 Systolic 使用 32-bit word address。完整權重不應全部常駐在
    // 17-bit on-chip word space；FPGA 版應改用 ping-pong tile loader。
    output logic [1:0]  ppu_mode_o,
    output logic [5:0]  ppu_scaling_factor_o,
    output logic [7:0]  sys_k_tile_cnt,
    output logic [16:0] sys_w_base_addr,
    output logic [16:0] sys_bias_base_addr
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
    localparam logic [4:0] PH_DONE      = 5'd10;

    logic [4:0] phase_q;

    assign phase_o   = phase_q;
    assign busy_exec = (phase_q != PH_IDLE) && (phase_q != PH_DONE);

    function automatic logic [4:0] next_phase;
        input logic [4:0] phase_i;
        begin
            case (phase_i)
                PH_RMS1:     next_phase = PH_QKV;
                PH_QKV:      next_phase = PH_QKT;
                PH_QKT:      next_phase = PH_SOFTMAX;
                PH_SOFTMAX:  next_phase = PH_ATTN_V;
                PH_ATTN_V:   next_phase = PH_OUT_PROJ;
                PH_OUT_PROJ: next_phase = PH_RMS2;
                PH_RMS2:     next_phase = PH_FC1;
                PH_FC1:      next_phase = PH_FC2;
                PH_FC2:      next_phase = PH_DONE;
                default:     next_phase = PH_IDLE;
            endcase
        end
    endfunction

    function automatic logic is_rms_phase;
        input logic [4:0] phase_i;
        begin
            is_rms_phase = (phase_i == PH_RMS1) || (phase_i == PH_RMS2);
        end
    endfunction

    function automatic logic is_systolic_phase;
        input logic [4:0] phase_i;
        begin
            is_systolic_phase =
                (phase_i == PH_QKV)      ||
                (phase_i == PH_QKT)      ||
                (phase_i == PH_ATTN_V)   ||
                (phase_i == PH_OUT_PROJ) ||
                (phase_i == PH_FC1)      ||
                (phase_i == PH_FC2);
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            phase_q       <= PH_IDLE;
            phase_start_o <= 1'b0;
            done_exec     <= 1'b0;
        end
        else begin
            phase_start_o <= 1'b0;
            done_exec     <= 1'b0;

            case (phase_q)
                PH_IDLE: begin
                    if (start_exec) begin
                        phase_q       <= PH_RMS1;
                        phase_start_o <= 1'b1;
                    end
                end

                PH_DONE: begin
                    done_exec <= 1'b1;
                    phase_q   <= PH_IDLE;
                end

                default: begin
                    if (phase_done_i) begin
                        if ((phase_q == PH_ATTN_V) && mhsa_repeat_i) begin
                            phase_q       <= PH_QKV;
                            phase_start_o <= 1'b1;
                        end
                        else begin
                            phase_q <= next_phase(phase_q);
                            if (next_phase(phase_q) != PH_DONE) begin
                                phase_start_o <= 1'b1;
                            end
                        end
                    end
                end
            endcase
        end
    end

    always_comb begin
        rms_start     = phase_start_o && is_rms_phase(phase_q);
        sys_en        = phase_start_o && is_systolic_phase(phase_q);
        softmax_start = phase_start_o && (phase_q == PH_SOFTMAX);

        ppu_mode_o           = 2'b00;
        ppu_scaling_factor_o = 6'd0;
        sys_k_tile_cnt       = 8'd0;
        sys_w_base_addr      = 17'd0;
        sys_bias_base_addr   = 17'd0;

        case (phase_q)
            PH_QKV: begin
                // Pure requant path: top feeds residual zero-point 128.
                ppu_mode_o           = 2'b00;
                ppu_scaling_factor_o = SCALE_QKV;
                sys_k_tile_cnt       = CHANNEL_TILE_NUM;
                sys_w_base_addr      = W_QKV_BASE;
                sys_bias_base_addr   = B_QKV_BASE;
            end

            PH_QKT: begin
                // Internal K buffer supplies weights. No PPU.
                sys_k_tile_cnt       = HEAD_DIM_TILE_NUM;
            end

            PH_ATTN_V: begin
                // Pure requant path to create O_attn buffer.
                ppu_mode_o           = 2'b00;
                ppu_scaling_factor_o = SCALE_ATTN_V;
                sys_k_tile_cnt       = SCORE_TILE_NUM;
            end

            PH_OUT_PROJ: begin
                ppu_mode_o           = 2'b00; // Residual Add 1 + X_mid stats
                ppu_scaling_factor_o = SCALE_OUT_PROJ;
                sys_k_tile_cnt       = CHANNEL_TILE_NUM;
                sys_w_base_addr      = W_OUT_BASE;
                sys_bias_base_addr   = B_OUT_BASE;
            end

            PH_FC1: begin
                ppu_mode_o           = 2'b01; // GELU + requant
                ppu_scaling_factor_o = SCALE_FC1;
                sys_k_tile_cnt       = CHANNEL_TILE_NUM;
                sys_w_base_addr      = W_FC1_BASE;
                sys_bias_base_addr   = B_FC1_BASE;
            end

            PH_FC2: begin
                ppu_mode_o           = 2'b10; // Residual Add 2
                ppu_scaling_factor_o = SCALE_FC2;
                sys_k_tile_cnt       = FFN_CHANNEL_TILE_NUM;
                sys_w_base_addr      = W_FC2_BASE;
                sys_bias_base_addr   = B_FC2_BASE;
            end

            default: begin
                // Defaults above.
            end
        endcase
    end

endmodule
