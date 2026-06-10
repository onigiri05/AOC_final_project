`timescale 1ns/1ps
`include "PPU.sv"
module tb_PPU;

    // ─────────────────────────────────────────────────────────────────
    // 1. 參數與訊號宣告
    // ─────────────────────────────────────────────────────────────────
    localparam CLK_PERIOD = 10; // 100MHz
    
    logic                         clk;
    logic                         rst;
    logic [1:0]                   ppu_mode;
    logic [5:0]                   scaling_factor;
    logic                         data_in_valid;
    logic                         token_start;
    logic                         token_end;
    logic [7:0]                   current_token_idx;
    logic signed [`DATA_BITS-1:0] data_in;
    logic [7:0]                   residual_in;

    logic [7:0]                   data_out;
    logic                         data_out_valid;
    logic [7:0]                   token_stat_addr;
    logic [7:0]                   token_stat_data;
    logic                         token_stat_we;

    // 驗證計數器
    int out_cycle_cnt;

    // ─────────────────────────────────────────────────────────────────
    // 2. 被測元件實例化 (DUT Instantiation)
    // ─────────────────────────────────────────────────────────────────
    PPU dut (
        .clk(clk),
        .rst(rst),
        .ppu_mode(ppu_mode),
        .scaling_factor(scaling_factor),
        .data_in_valid(data_in_valid),
        .token_start(token_start),
        .token_end(token_end),
        .current_token_idx(current_token_idx),
        .data_in(data_in),
        .residual_in(residual_in),
        .data_out(data_out),
        .data_out_valid(data_out_valid),
        .token_stat_addr(token_stat_addr),
        .token_stat_data(token_stat_data),
        .token_stat_we(token_stat_we)
    );

    // ─────────────────────────────────────────────────────────────────
    // 3. 時脈產生器 (Clock Generator)
    // ─────────────────────────────────────────────────────────────────
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // ─────────────────────────────────────────────────────────────────
    // 4. 下游連續收料與監控邏輯 (Scoreboard / Monitor)
    // ─────────────────────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            out_cycle_cnt <= 0;
        end else if (data_out_valid) begin
            out_cycle_cnt <= out_cycle_cnt + 1;
            // 隨機抽樣列印輸出，確認資料正確性
            if (out_cycle_cnt % 50 == 0 || out_cycle_cnt == 255) begin
                $display("[Monitor] Time=%0tds | Out#%0d: Mode=%b, uint8_out=%d (hex=0x%h)", 
                         $time, out_cycle_cnt, ppu_mode, data_out, data_out);
            end
        end
    end

    // 監控 RMS 累加器寫入 Token Stat SRAM 的行為
    always_ff @(posedge clk) begin
        if (!rst && token_stat_we) begin
            $display("[SRAM Write] Token=%0d 處理完畢! 寫入 SRAM 地址=%0d, inv_rms 近似值=0x%h", 
                     current_token_idx, token_stat_addr, token_stat_data);
        end
    end

    // ─────────────────────────────────────────────────────────────────
    // 5. 測試激勵程序 (Test Stimulus)
    // ─────────────────────────────────────────────────────────────────
    initial begin
        // --- 初始化訊號 ---
        rst                = 1;
        ppu_mode           = 2'b00;
        scaling_factor     = 6'd0;
        data_in_valid      = 0;
        token_start        = 0;
        token_end          = 0;
        current_token_idx  = 8'd0;
        data_in            = '0;
        residual_in        = 8'd128; // 實體 0 偏置值

        #(CLK_PERIOD * 5);
        @(negedge clk);
        rst = 0; // 解除重設
        #(CLK_PERIOD * 2);

        // =============================================================
        // 測試場景一：ppu_mode = 2'b01 (FFN FC1 + GELU 激活 + Requant)
        // 期望行為：1 cycle pipeline 延遲後，連續 256 個 cycle 吐出資料
        // =============================================================
        $display("\n====== [Scenario 1] Testing FFN FC1 -> GELU -> Requant Pipeline ======");
        ppu_mode       = 2'b01; // 切換至 FC1 模式
        scaling_factor = 6'd4;  // 設定 Power-of-two 算術右移 4 位元
        out_cycle_cnt  = 0;

        @(posedge clk);
        // 開始連續 256 個 cycle 灌入資料
        for (int i = 0; i < 256; i = i + 1) begin
            data_in_valid = 1;
            // 模擬隨機變動的 FC1 密集 INT32 部分和
            data_in       = (i * 128) - 4096; 
            residual_in   = 8'dx; // FC1 階段不關心殘差輸入
            @(posedge clk);
        end
        
        // 關閉輸入，確認 PPU 是否能把最後一筆 Pipeline 資料清出來
        data_in_valid = 0;
        #(CLK_PERIOD * 3);

        // 驗證連續吞吐量
        if (out_cycle_cnt == 256) begin
            $display("[Result] Scenario 1 SUCCESS: 完美連續消化並輸出 256 個 Opsum!");
        end else begin
            $display("[Result] Scenario 1 ERROR: 輸出計數 %0d 與預期 256 不符!", out_cycle_cnt);
        end


        // =============================================================
        // 測試場景二：ppu_mode = 2'b00 (Attention 結束 + Residual Add + RMS 累加)
        // 期望行為：連續 256 cycles 輸入，且在特定通道邊界觸發 SRAM 寫入
        // =============================================================
        $display("\n====== [Scenario 2] Testing Attn Out + Residual Add + RMS Accumulator ======");
        ppu_mode       = 2'b00; // 切換至 Attention 殘差模式
        scaling_factor = 6'd2;  // 右移 2 位元
        out_cycle_cnt  = 0;
        current_token_idx = 8'd45; // 假設目前正流過第 45 號 Token 的區塊

        @(posedge clk);
        for (int i = 0; i < 256; i = i + 1) begin
            data_in_valid = 1;
            data_in       = (i * 64);      // Main branch 成果
            residual_in   = 8'd150;        // Residual branch 捷徑特徵 (uint8 zp=128)
            
            // 模擬第 1 個 channel 進來的邊界脈衝
            token_start   = (i == 0) ? 1'b1 : 1'b0;
            // 模擬第 256 個 channel 流完（通道流結束）的邊界脈衝
            token_end     = (i == 255) ? 1'b1 : 1'b0;
            
            @(posedge clk);
        end
        
        data_in_valid = 0;
        token_end     = 0;
        #(CLK_PERIOD * 5);

        if (out_cycle_cnt == 256) begin
            $display("[Result] Scenario 2 SUCCESS: 殘差相加通路連續吞吐驗證成功!");
        end else begin
            $display("[Result] Scenario 2 ERROR: 輸出計數不符!");
        end

        // --- 結束測試 ---
        $display("\n====== PPU Testbench Completed ======");
        $finish;
    end

endmodule