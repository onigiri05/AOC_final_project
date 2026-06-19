`timescale 1ns/1ps

module Top_FC1_TB;

    // =================================================================
    // 1. 參數與訊號宣告
    // =================================================================
    parameter int TOKEN_TILE   = 16;
    parameter int CHANNEL_TILE = 16;
    parameter int DATA_W       = 8;
    
    logic clk;
    logic rst;

    // Systolic 脈動陣列控制與配置介面
    logic en;
    logic module_ready;
    logic [16:0] act_base_addr;
    logic [16:0] w_base_addr;
    logic [16:0] bias_base_addr;
    logic [6:0]  k_tile_cnt;

    // BRAM 埠介面 (由控制邏輯或 TB 驅動)
    logic [16:0] act_bram_addr;
    logic [16:0] w_bram_addr;
    logic [16:0] bias_bram_addr;
    logic act_bram_valid;
    logic w_bram_valid;
    logic bias_bram_valid;
    logic [31:0] act_bram_data;
    logic [31:0] w_bram_data;
    logic [31:0] bias_bram_data;

    // PPU 後處理單元控制介面
    logic [1:0]  ppu_mode_i;
    logic [5:0]  scaling_factor_i;
    logic [7:0]  base_token_idx_i;
    logic [4:0]  channel_tile_idx_i;
    logic [15:0] token_valid_mask_i;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_tile_i;

    // PPU 最終特徵圖輸出與交握介面
    logic data_tile_valid_o;
    logic data_tile_ready_i;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o;

    // 串接內部線路：用於捕捉 Systolic 串流輸出的訊號
    wire        sys_opsum_valid;
    wire [31:0] sys_opsum;

    // 串轉並（SIPO）暫存硬體邏輯宣告
    reg  [8191:0] psum_tile_reg;
    reg  [7:0]    opsum_cnt;
    reg           ppu_tile_valid;
    wire          ppu_tile_ready;

    // =================================================================
    // 2. 實例化子模組與級聯電路連接
    // =================================================================
    
    // (A) 實例化 16x16 脈動陣列核心
    Systolic u_Systolic (
        .clk(clk),
        .rst(rst),
        .en(en),
        .module_ready(module_ready),
        .act_base_addr(act_base_addr),
        .w_base_addr(w_base_addr),
        .bias_base_addr(bias_base_addr),
        .k_tile_cnt(k_tile_cnt),
        .act_bram_addr(act_bram_addr),
        .w_bram_addr(w_bram_addr),
        .bias_bram_addr(bias_bram_addr),
        .act_bram_valid(act_bram_valid),
        .w_bram_valid(w_bram_valid),
        .bias_bram_valid(bias_bram_valid),
        .act_bram_data(act_bram_data),
        .w_bram_data(w_bram_data),
        .bias_bram_data(bias_bram_data),
        .opsum_valid(sys_opsum_valid),  // 串流輸出有效訊號
        .opsum(sys_opsum)              // 32-bit 串流部分和
    );

    // (B) 串轉並（SIPO）收集邏輯：將 256 個 32-bit 點打包成一個 8192-bit Tile
    always_ff @(posedge clk) begin
        if (rst) begin
            opsum_cnt      <= 8'd0;
            psum_tile_reg  <= 8192'd0;
            ppu_tile_valid <= 1'b0;
        end else begin
            if (sys_opsum_valid) begin
                // 依據吐出順序，將元素依序填入對應的 bit 區段中
                // 順序由 psum_buffer[0][0] 排至 psum_buffer[15][15]
                psum_tile_reg[opsum_cnt * 32 +: 32] <= sys_opsum;
                opsum_cnt                           <= opsum_cnt + 8'd1;
                
                if (opsum_cnt == 8'd255) begin
                    ppu_tile_valid <= 1'b1; // 256 個點收集齊全，拉高有效信號
                end
            end
            
            // 當 PPU 成功收走這一整塊 Tile 時，清除有效信號並重置計數器
            if (ppu_tile_valid && ppu_tile_ready) begin
                ppu_tile_valid <= 1'b0;
                opsum_cnt      <= 8'd0;
            end
        end
    end

    // (C) 實例化後處理單元 (PPU頂層)
    PPU #(
        .TOKEN_NUM(197),
        .CHANNEL_NUM(384),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .SUM_W(32)
    ) u_PPU (
        .clk(clk),
        .rst(rst),
        .ppu_mode_i(ppu_mode_i),
        .scaling_factor_i(scaling_factor_i),
        .tile_valid_i(ppu_tile_valid),     // 連接到 SIPO 緩衝器有效信號
        .tile_ready_o(ppu_tile_ready),     
        .psum_tile_i(psum_tile_reg),       // 打包完成的 8192-bit 數據
        .residual_tile_i(residual_tile_i),
        .base_token_idx_i(base_token_idx_i),
        .channel_tile_idx_i(channel_tile_idx_i),
        .token_valid_mask_i(token_valid_mask_i),
        .data_tile_valid_o(data_tile_valid_o),
        .data_tile_ready_i(data_tile_ready_i),
        .data_tile_o(data_tile_o),
        
        // 旁路統計交握（因為 FC1 模式下不啟用 RMS 累加，直接給 ready 即可）
        .stat_valid_o(),
        .stat_ready_i(1'b1),
        .stat_token_idx_o(),
        .sum_sq_o()
    );

    // =================================================================
    // 3. 模擬片上 BRAM 記憶體空間與行為 (32-bit 連續存放)
    // =================================================================
    logic [31:0] act_bram_mem  [0:4095];
    logic [31:0] w_bram_mem    [0:4095];
    logic [31:0] bias_bram_mem [0:255];
    
    // Golden 黃金矩陣預期輸出驗證陣列 (256 筆 uint8 資料)
    logic [7:0]  golden_fc1_mem [0:255];

    // 時脈產生邏輯 (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // 對齊 BRAM 讀取行為：Data 比 Addr 晚一個 Cycle 吐出
    always_ff @(posedge clk) begin
        if (rst) begin
            act_bram_data  <= 32'd0;
            w_bram_data    <= 32'd0;
            bias_bram_data <= 32'd0;
        end else begin
            act_bram_data  <= act_bram_mem[act_bram_addr];
            w_bram_data    <= w_bram_mem[w_bram_addr];
            bias_bram_data <= bias_bram_mem[bias_bram_addr];
        end
    end

    // =================================================================
    // 4. Stimulus Process
    // =================================================================
    initial begin
        // 波形導出設定
        $dumpfile("top_fc1_pipeline.vcd");
        $dumpvars(0, Top_FC1_TB);

        // 加載由 Python 生成的測試資料
        $readmemh("tb_act.hex",    act_bram_mem);
        $readmemh("tb_weight.hex", w_bram_mem);
        $readmemh("tb_bias.hex",   bias_bram_mem);
        $readmemh("tb_golden.hex", golden_fc1_mem);
        $display("[System-TB] All unified verification hex files loaded.");

        // 初始化系統控制訊號
        rst = 1;
        en = 0;
        act_base_addr = 17'd0;
        w_base_addr = 17'd0;
        bias_base_addr = 17'd0;
        k_tile_cnt = 7'd1;          // 執行 1 次 16x16 GEMM 區塊乘法

        // 控制端通知 BRAM 資料已載入完畢，拉高 valid
        act_bram_valid = 1;
        w_bram_valid = 1;
        bias_bram_valid = 1;

        // 配置 PPU 為 FFN FC1 測試模式
        ppu_mode_i = 2'b01;          // 2'b01 代表 FC1 階段 (啟動 GELU 查表，旁路殘差)
        scaling_factor_i = 6'd2;     // 設定硬體算術右移因子 n = 2 (即除以 4)
        base_token_idx_i = 8'd0;
        channel_tile_idx_i = 5'd0;
        token_valid_mask_i = 16'hFFFF; // 16 行 token 全數有效
        residual_tile_i = '0;        // FC1 旁路殘差，直接給 0
        data_tile_ready_i = 1;       // 模擬後級 Global Buffer 隨時處於 Ready 狀態

        #20;
        rst = 0;                     // 解除重置
        @(posedge clk);
        wait(module_ready == 1'b1);  // 等待脈動陣列初始化完成
        $display("[System-TB] Systolic Array is idle and ready. Asserting 'en'...");

        // 發送致能信號
        en = 1'b1;
        @(posedge clk);
        en = 1'b0;                   // en 僅需維持 1 cycle，硬體內部會鎖存參數

        // 捕捉最終輸出
        fork
            begin
                // 防呆機制：若電路因交握阻塞陷入死鎖，15,000ns 後強制終止
                #15000;
                $error("[System-TB] Simulation Timeout! Pipeline handshake broke down.");
                $finish;
            end
            begin
                // 等待 PPU 完成 Stage 1 查表與 Stage 2 流水線，拉高輸出有效信號
                wait(data_tile_valid_o && data_tile_ready_i);
                $display("[System-TB] Output Valid detected from PPU wrapper! Starting bit-level analysis...");
                // 呼叫比對驗證任務
                check_result();
                $finish;
            end
        join
    end

    // =================================================================
    // 5. 自動化資料比對與驗證任務 (Verification Task)
    // =================================================================
    task automatic check_result();
        int error_count = 0;
        logic [7:0] current_hardware_out;
        
        for (int i = 0; i < 256; i++) begin
            // 從 PPU 輸出的 2048-bit 總向量中，切片分出當前索引的 8-bit uint8 特徵值
            current_hardware_out = data_tile_o[i*8 +: 8];
            
            if (current_hardware_out !== golden_fc1_mem[i]) begin
                if (error_count < 12) begin // 限制列印前幾筆錯誤資訊，避免洗版
                    $error("Mismatch at Index %0d: Expected [0x%02X], Got [0x%02X]", 
                            i, golden_fc1_mem[i], current_hardware_out);
                end
                error_count++;
            end
        end

        $display("\n====================================================");
        if (error_count == 0) begin
            $display(">> [TEST PASSED] SUCCESS: All 256 cascaded elements perfectly match the Golden Model!");
        end else begin
            $display(">> [TEST FAILED] ERROR: Found %0d total mismatches between Systolic-PPU pipeline and Golden Model.", error_count);
        end
        $display("====================================================\n");
    endtask

endmodule