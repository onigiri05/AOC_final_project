`timescale 1ns/1ps

module Top_Patch_Embed_TB;

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
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] residual_tile_i; // 2048-bit 殘差/底色輸入

    // PPU 最終特徵圖輸出與交握介面
    logic data_tile_valid_o;
    logic data_tile_ready_i;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o;

    // RMSNorm 統計交握介面 (修復 Packed 類型，單個 32-bit Packed 連線)
    logic        stat_valid_o;
    logic        stat_ready_i;
    logic [7:0]  stat_token_idx_o;
    logic [31:0] sum_sq_o;           

    // TB 內部獨立的 Unpacked 暫存陣列，用來動態收錄 16 個 Token 的 RMS 結果
    logic [31:0] tb_sum_sq_captured [0:15]; 

    // 串接內部線線：用於捕捉 Systolic 串流輸出的訊號
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
        .opsum_valid(sys_opsum_valid),  
        .opsum(sys_opsum)              
    );

    // (B) 串轉並（SIPO）收集邏輯
    always_ff @(posedge clk) begin
        if (rst) begin
            opsum_cnt      <= 8'd0;
            psum_tile_reg  <= 8192'd0;
            ppu_tile_valid <= 1'b0;
        end else begin
            if (sys_opsum_valid) begin
                psum_tile_reg[opsum_cnt * 32 +: 32] <= sys_opsum;
                opsum_cnt                           <= opsum_cnt + 8'd1;
                if (opsum_cnt == 8'd255) begin
                    ppu_tile_valid <= 1'b1;
                end
            end
            
            if (ppu_tile_valid && ppu_tile_ready) begin
                ppu_tile_valid <= 1'b0;
                opsum_cnt      <= 8'd0;
            end
        end
    end

    // (C) 實例化後處理單元 (PPU頂層，切換為 Patch_Embed.Proj 運算模式)
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
        .tile_valid_i(ppu_tile_valid),     
        .tile_ready_o(ppu_tile_ready),     
        .psum_tile_i(psum_tile_reg),       
        .residual_tile_i(residual_tile_i), 
        .base_token_idx_i(base_token_idx_i),
        .channel_tile_idx_i(channel_tile_idx_i),
        .token_valid_mask_i(token_valid_mask_i),
        .data_tile_valid_o(data_tile_valid_o),
        .data_tile_ready_i(data_tile_ready_i),
        .data_tile_o(data_tile_o),
        
        // 統計交握連線
        .stat_valid_o(stat_valid_o),
        .stat_ready_i(stat_ready_i),
        .stat_token_idx_o(stat_token_idx_o),
        .sum_sq_o(sum_sq_o)                 
    );

    // =================================================================
    // 3. 模擬片上 BRAM 記憶體空間與行為
    // =================================================================
    logic [31:0] act_bram_mem      [0:4095];
    logic [31:0] w_bram_mem        [0:4095];
    logic [31:0] bias_bram_mem     [0:255];
    
    logic [31:0] residual_bram_mem [0:63];   
    logic [7:0]  golden_pe_mem     [0:255];  // Patch_Embed 預期輸出特徵圖
    logic [31:0] golden_sum_sq     [0:15];   // 預期 16 個 Token 的 RMS 平方和

    // 時脈產生邏輯 (100MHz)
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // BRAM 讀取行為對齊
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
    // 4. 動態捕捉處理 (Capture Logic)
    //    時序修正：利用非阻塞賦值安全收集每筆 Token 的 RMS 結果
    // =================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int k = 0; k < 16; k++) begin
                tb_sum_sq_captured[k] <= 32'd0;
            end
        end else begin
            if (stat_valid_o && stat_ready_i) begin
                tb_sum_sq_captured[stat_token_idx_o[3:0]] <= sum_sq_o;
            end
        end
    end

    // =================================================================
    // 5. Stimulus Process (測試主流程)
    // =================================================================
    initial begin
        $dumpfile("top_patch_embed_pipeline.vcd");
        $dumpvars(0, Top_Patch_Embed_TB);

        // 加載 Patch_Embed 專用的測試測資
        $readmemh("tb_pe_act.hex",      act_bram_mem);
        $readmemh("tb_pe_weight.hex",   w_bram_mem);
        $readmemh("tb_pe_bias.hex",     bias_bram_mem);
        $readmemh("tb_pe_residual.hex", residual_bram_mem);
        $readmemh("tb_pe_golden.hex",   golden_pe_mem);
        $readmemh("tb_pe_sum_sq.hex",   golden_sum_sq);
        $display("[System-TB] Patch_Embed verification hex files loaded.");

        // 初始化系統控制訊號
        rst = 1;
        en = 0;
        act_base_addr = 17'd0;
        w_base_addr = 17'd0;
        bias_base_addr = 17'd0;
        k_tile_cnt = 7'd1;

        act_bram_valid = 1;
        w_bram_valid = 1;
        bias_bram_valid = 1;

        // 配置 PPU 為 Patch Embedding 模式
        ppu_mode_i = 2'b00;          // 2'b00 代表 Attention Output / 初始輸入階段 (啟動殘差與 RMS 統計)
        scaling_factor_i = 6'd2;     
        base_token_idx_i = 8'd0;     
        channel_tile_idx_i = 5'd23;  // 模擬最後一組 Channel Tile 以順利觸發隨後的 RMS 輸出
        token_valid_mask_i = 16'hFFFF; 
        
        // 載入殘差/基礎特徵向量
        for(int i=0; i<64; i++) begin
            residual_tile_i[i*32 +: 32] = residual_bram_mem[i];
        end

        data_tile_ready_i = 1;       
        stat_ready_i = 1;            

        #20;
        rst = 0;                     
        @(posedge clk);
        wait(module_ready == 1'b1);
        
        $display("[System-TB] Systolic Array is IDLE. Asserting 'en' for Patch_Embed.Proj...");
        en = 1'b1;
        @(posedge clk);
        en = 1'b0;

        // 捕捉與驗證輸出
        fork
            begin
                #20000;
                $error("[System-TB] Simulation Timeout! Pipeline hung up.");
                $finish;
            end
            begin
                // 1. 驗證 PPU 最終特徵圖輸出與殘差加法結果
                wait(data_tile_valid_o && data_tile_ready_i);
                $display("[System-TB] Output Data Tile Valid detected! Checking Pixels...");
                check_data_result();
                
                // 2. 等待 16 筆 Token 的 RMS 統計量全部透過交握捕捉完成
                for (int count = 0; count < 16; count++) begin
                    @(posedge clk);
                    while (!(stat_valid_o && stat_ready_i)) @(posedge clk);
                end
                
                
                @(posedge clk); 
                
                $display("[System-TB] All 16 RMSNorm Stats captured! Checking Square Sums...");
                check_stat_result();
                
                $finish;
            end
        join
    end

    // =================================================================
    // 6. 自動化資料比對與驗證任務 (修復 begin/end 與 task/endtask 配對)
    // =================================================================
    
    task automatic check_data_result();
        int error_count = 0;
        logic [7:0] current_hardware_out;
        
        begin
            for (int i = 0; i < 256; i++) begin
                current_hardware_out = data_tile_o[i*8 +: 8];
                if (current_hardware_out !== golden_pe_mem[i]) begin
                    if (error_count < 12) begin
                        $error("Data Mismatch at Pixel %0d: Expected [0x%02X], Got [0x%02X]", 
                                i, golden_pe_mem[i], current_hardware_out);
                    end
                    error_count++;
                end
            end

            if (error_count == 0) begin
                $display(">> [PASSED] SUCCESS: Patch_Embed Output Feature Map matches Golden Model perfectly!");
            end else begin
                $display(">> [FAILED] ERROR: Found %0d feature map mismatches.", error_count);
            end
        end
    endtask

    task automatic check_stat_result();
        int error_count = 0;
        begin
            for (int t = 0; t < 16; t++) begin
                if (tb_sum_sq_captured[t] !== golden_sum_sq[t]) begin
                    $error("RMS Stat Mismatch at Token %0d: Expected [%0d], Got [%0d]", 
                            t, golden_sum_sq[t], tb_sum_sq_captured[t]);
                    error_count++;
                end
            end

            if (error_count == 0) begin
                $display(">> [PASSED] SUCCESS: Patch_Embed RMSNorm Accumulation Sum-of-Squares matches Golden Model perfectly!");
            end else begin
                $display(">> [FAILED] ERROR: Found %0d RMS statistics mismatches.", error_count);
            end
        end
    endtask

endmodule