`timescale 1ns/1ps

// ============================================================
// 模組：Global_Controller_FSM
// 功能：使用固定狀態機 (FSM) 依序調度 ViT 加速器的各個硬體模組
// ============================================================
module Global_Controller_FSM (
    input  logic        clk,
    input  logic        rst_n,

    // 系統外部控制 (來自 PS 端的 ARM 處理器)
    input  logic        start_exec, // 啟動推論 (Pulse)
    output logic        done_exec,  // 推論完成 (Pulse)

    // 控制 RMSNorm
    output logic        rms_start,
    input  logic        rms_done,

    // 控制 Systolic Array
    output logic        sys_en,
    input  logic        sys_module_ready,
    output logic [16:0] sys_w_base_addr,
    output logic [16:0] sys_bias_base_addr,
    output logic [6:0]  sys_k_tile_cnt,

    // 控制 PPU
    output logic [1:0]  ppu_mode_o,
    output logic [5:0]  ppu_scaling_factor_o,

    // 控制 Softmax
    output logic        softmax_start,
    input  logic        softmax_done
);

    // ==========================================
    // FSM 狀態定義 (ViT Block 執行順序)
    // ==========================================
    typedef enum logic [3:0] {
        ST_IDLE,
        ST_RMSNORM,        // 啟動 RMSNorm
        ST_WAIT_RMS,       // 等待 RMSNorm 結束
        
        ST_QKV_PROJ,       // 啟動 QKV 矩陣乘法
        ST_WAIT_QKV_START, // 等待 Systolic Array ready 拉低
        ST_WAIT_QKV_DONE,  // 等待 Systolic Array ready 拉高
        
        ST_SOFTMAX,        // 啟動 FlashAttention Softmax
        ST_WAIT_SOFTMAX,   // 等待 Softmax 結束
        
        ST_ATTN_OUT,       // 啟動 Attention Output 矩陣乘法
        ST_WAIT_ATTN_START,
        ST_WAIT_ATTN_DONE,
        
        ST_FC1,            // 啟動 MLP FC1 矩陣乘法
        ST_WAIT_FC1_START,
        ST_WAIT_FC1_DONE,
        
        ST_FC2,            // 啟動 MLP FC2 矩陣乘法
        ST_WAIT_FC2_START,
        ST_WAIT_FC2_DONE,
        
        ST_DONE            // 執行完畢
    } state_t;

    state_t state, next_state;

    // ==========================================
    // FSM State Register
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
        end else begin
            state <= next_state;
        end
    end

    // ==========================================
    // FSM Next State Logic (狀態轉換)
    // ==========================================
    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE: begin
                if (start_exec) next_state = ST_RMSNORM;
            end
            
            // --- 1. RMSNorm ---
            ST_RMSNORM:  next_state = ST_WAIT_RMS;
            ST_WAIT_RMS: if (rms_done) next_state = ST_QKV_PROJ;
            
            // --- 2. QKV Projection ---
            ST_QKV_PROJ:       next_state = ST_WAIT_QKV_START;
            ST_WAIT_QKV_START: if (!sys_module_ready) next_state = ST_WAIT_QKV_DONE;
            ST_WAIT_QKV_DONE:  if (sys_module_ready)  next_state = ST_SOFTMAX;
            
            // --- 3. Softmax ---
            ST_SOFTMAX:      next_state = ST_WAIT_SOFTMAX;
            ST_WAIT_SOFTMAX: if (softmax_done) next_state = ST_ATTN_OUT;
            
            // --- 4. Attention Output (包含 Residual Add) ---
            ST_ATTN_OUT:        next_state = ST_WAIT_ATTN_START;
            ST_WAIT_ATTN_START: if (!sys_module_ready) next_state = ST_WAIT_ATTN_DONE;
            ST_WAIT_ATTN_DONE:  if (sys_module_ready)  next_state = ST_FC1;

            // --- 5. MLP FC1 (包含 GELU) ---
            ST_FC1:            next_state = ST_WAIT_FC1_START;
            ST_WAIT_FC1_START: if (!sys_module_ready) next_state = ST_WAIT_FC1_DONE;
            ST_WAIT_FC1_DONE:  if (sys_module_ready)  next_state = ST_FC2;

            // --- 6. MLP FC2 (包含 Residual Add) ---
            ST_FC2:            next_state = ST_WAIT_FC2_START;
            ST_WAIT_FC2_START: if (!sys_module_ready) next_state = ST_WAIT_FC2_DONE;
            ST_WAIT_FC2_DONE:  if (sys_module_ready)  next_state = ST_DONE;

            // --- 結束 ---
            ST_DONE: next_state = ST_IDLE;
            
            default: next_state = ST_IDLE;
        endcase
    end

    // ==========================================
    // FSM Output Logic (輸出控制訊號與參數)
    // ==========================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rms_start            <= 1'b0;
            sys_en               <= 1'b0;
            softmax_start        <= 1'b0;
            done_exec            <= 1'b0;
            
            ppu_mode_o           <= 2'b00;
            ppu_scaling_factor_o <= 6'd0;
            sys_k_tile_cnt       <= 7'd0;
            sys_w_base_addr      <= 17'd0;
            sys_bias_base_addr   <= 17'd0;
        end else begin
            // 預設將 Pulse 訊號清零
            rms_start     <= 1'b0;
            sys_en        <= 1'b0;
            softmax_start <= 1'b0;
            done_exec     <= 1'b0;

            case (state)
                ST_RMSNORM: begin
                    rms_start <= 1'b1;
                end
                
                ST_QKV_PROJ: begin
                    sys_en               <= 1'b1;
                    ppu_mode_o           <= 2'b01;  // 根據你的規劃設定 (若QKV不進GELU/Residual需確認模式)
                    ppu_scaling_factor_o <= 6'd2;   // 填入 QKV 對應的右移位數
                    sys_k_tile_cnt       <= 7'd24;  // 384/16 = 24 個 K-tile
                    sys_w_base_addr      <= 17'h0000; // QKV Weight 位址
                    sys_bias_base_addr   <= 17'h0000;
                end
                
                ST_SOFTMAX: begin
                    softmax_start <= 1'b1;
                end
                
                ST_ATTN_OUT: begin
                    sys_en               <= 1'b1;
                    ppu_mode_o           <= 2'b00;  // 00: Attention Output (Residual Add + RMS)
                    ppu_scaling_factor_o <= 6'd1;   
                    sys_k_tile_cnt       <= 7'd24;  
                    sys_w_base_addr      <= 17'h1000; // Attn Out Weight 位址 (範例)
                    sys_bias_base_addr   <= 17'h0010;
                end

                ST_FC1: begin
                    sys_en               <= 1'b1;
                    ppu_mode_o           <= 2'b01;  // 01: FFN FC1 (GELU)
                    ppu_scaling_factor_o <= 6'd0;   
                    sys_k_tile_cnt       <= 7'd24;  // Input 384/16 = 24
                    sys_w_base_addr      <= 17'h2000; // FC1 Weight 位址 (範例)
                    sys_bias_base_addr   <= 17'h0020;
                end

                ST_FC2: begin
                    sys_en               <= 1'b1;
                    ppu_mode_o           <= 2'b10;  // 10: FFN FC2 (Residual Add + RMS)
                    ppu_scaling_factor_o <= 6'd1;   
                    sys_k_tile_cnt       <= 7'd96;  // Input 1536/16 = 96
                    sys_w_base_addr      <= 17'h3000; // FC2 Weight 位址 (範例)
                    sys_bias_base_addr   <= 17'h0030;
                end
                
                ST_DONE: begin
                    done_exec <= 1'b1; // 發出結束 Pulse
                end
            endcase
        end
    end

endmodule