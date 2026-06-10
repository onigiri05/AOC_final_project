`include "ASIC.svh"

module PPU (
    input  logic                         clk,
    input  logic                         rst,
    
    // 全域控制與模式配置
    input  logic [1:0]                   ppu_mode,          // 00: Attn Out, 01: FFN FC1, 10: FFN FC2
    input  logic [5:0]                   scaling_factor,    // Requant 移位值 n
    input  logic                         data_in_valid,     // 輸入資料有效flag
    input  logic                         token_start,       // 當前 Token 串流起點 (用於初始化累加器)
    input  logic                         token_end,         // 當前 Token 串流終點 (用於觸發寫入 SRAM)
    input  logic [7:0]                   current_token_idx, // Token 索引 (0~196)

    // 數據輸入通路
    input  logic signed [`DATA_BITS-1:0] data_in,           // 來自 Systolic Array 的成果 (INT32)
    input  logic [7:0]                   residual_in,       // 來自 Global Buffer 的 Shortcut (INT8)

    // 數據輸出通路 (特徵圖)
    output logic [7:0]                   data_out,          // 輸出至 L2 Global Buffer / 下級 Activation Buffer
    output logic                         data_out_valid,    // 特徵圖輸出有效flag
    
    // Token Stat SRAM 寫入介面 (RMSNorm Fusion 第一階段)
    output logic [7:0]                   token_stat_addr,
    output logic [7:0]                   token_stat_data,
    output logic                         token_stat_we
);

    // 內部子模組互連訊號線
    logic signed [`DATA_BITS-1:0] gelu_to_requant;
    logic signed [`DATA_BITS-1:0] requant_in_mux;
    logic [7:0]                   requant_to_residual;
    logic [7:0]                   res_add_to_mux;
    
    logic                         rms_acc_en;

    // ─────────────────────────────────────────────────────────────────
    // [Unit 2] GELU Unit 實例化 (非線性激活)
    // ─────────────────────────────────────────────────────────────────
    GELU_Unit u_GELU_Unit (
        .clk(clk),
        .rst(rst),
        .en(data_in_valid && (ppu_mode == 2'b01)), // 僅在 FFN FC1 階段啟用
        .data_in(data_in),
        .data_out(gelu_to_requant)
    );

    // Requant 輸入多路選擇器：FC1 走 GELU 查表線；其餘運算走原始 data_in 線
    assign requant_in_mux = (ppu_mode == 2'b01) ? gelu_to_requant : data_in;

    // ─────────────────────────────────────────────────────────────────
    // [Unit 1] Requant Unit 實例化 (INT32 -> uint8 再量化)
    // ─────────────────────────────────────────────────────────────────
    Requant_Unit u_Requant_Unit (
        .data_in(requant_in_mux),
        .scaling_factor(scaling_factor),
        .data_out(requant_to_residual)
    );

    // ─────────────────────────────────────────────────────────────────
    // [Unit 3] Residual Add Unit 組合邏輯實現 (Shortcut 相加)
    // ─────────────────────────────────────────────────────────────────
    // 承接自 Residual_Add_Unit.sv 的 uint8 (Zero-point=128) 飽和相加算法
    // 公式：q_out = clamp(q_main + q_residual - 128, 0, 255)
    logic signed [9:0] sum_q;
    always_comb begin
        sum_q = $signed({1'b0, requant_to_residual})
              + $signed({1'b0, residual_in})
              - $signed(10'sd128); // 減去一個 ZERO_POINT

        if (sum_q < 10'sd0) begin
            res_add_to_mux = 8'd0;    // 下溢飽和截斷
        end else if (sum_q > 10'sd255) begin
            res_add_to_mux = 8'd255;  // 上溢飽和截斷
        end else begin
            res_add_to_mux = sum_q[7:0];
        end
    end

    // ─────────────────────────────────────────────────────────────────
    // 頂層 Datapath MUX 輸出排程
    // ─────────────────────────────────────────────────────────────────
    always_comb begin
        case (ppu_mode)
            2'b00:   data_out = res_add_to_mux;      // 狀況一：Attn Out + X = X_mid
            2'b01:   data_out = requant_to_residual; // 狀況二：FC1 + GELU -> Requant -> 直通下一級
            2'b10:   data_out = res_add_to_mux;      // 狀況三：FC2 + X_mid = X_out
            default: data_out = requant_to_residual;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) data_out_valid <= 1'b0;
        else     data_out_valid <= data_in_valid;
    end

    // ─────────────────────────────────────────────────────────────────
    // [Unit 4] RMS Stat Accumulator 串流內部邏輯 (Σx² 累加與 inv_rms 查表)
    // ─────────────────────────────────────────────────────────────────
    assign rms_acc_en = (ppu_mode == 2'b00) || (ppu_mode == 2'b10);

    logic signed [8:0]  centered_x;
    logic [15:0]        square_x;
    
    // 1. 還原 uint8 偏移量至 signed 數值，並執行單週期平方計算
    assign centered_x = $signed({1'b0, data_out}) - $signed(9'sd128);
    assign square_x   = centered_x * centered_x;

    // 2. 利用記憶體陣列保存 197 個 Token 分開動態累積的中途 Partial Sum (解決通道交織交錯進來的問題)
    logic [31:0] partial_sum_mem [0:196];
    logic [31:0] current_sum;

    always_comb begin
        if (data_out_valid && rms_acc_en) begin
            if (token_start)
                current_sum = {16'b0, square_x};
            else
                current_sum = partial_sum_mem[current_token_idx] + square_x;
        end else begin
            current_sum = partial_sum_mem[current_token_idx];
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            integer i;
            for (i = 0; i < 197; i = i + 1) begin
                partial_sum_mem[i] <= 32'd0;
            end
        end else begin
            if (data_out_valid && rms_acc_en) begin
                partial_sum_mem[current_token_idx] <= current_sum;
            end else if (token_end && rms_acc_en) begin
                // 當前 Token 處理結束並寫入 SRAM 後，將快取清零以供下個 Block 使用
                partial_sum_mem[current_token_idx] <= 32'd0;
            end
        end
    end

    // 3. 倒數方均根常數 ROM 查找表 (256 x 8-bit)
    logic [7:0] inv_rms_rom [0:255];
    initial begin
        // 此處依據 Python 腳本導出的常數寫入實體硬體 ROM 近似值
    end

    // 4. 當 token_end 脈衝到來時，單週期完成查表並對接寫入 Token Stat SRAM
    always_ff @(posedge clk) begin
        if (rst) begin
            token_stat_we   <= 1'b0;
            token_stat_addr <= 8'd0;
            token_stat_data <= 8'd0;
        end else if (token_end && rms_acc_en) begin
            token_stat_we   <= 1'b1;
            token_stat_addr <= current_token_idx;
            // 採用 current_sum 的高位元區段作為動態映射 ROM 索引 (Index Mapping)
            token_stat_data <= inv_rms_rom[current_sum[23:16]]; 
        end else begin
            token_stat_we   <= 1'b0;
        end
    end

endmodule