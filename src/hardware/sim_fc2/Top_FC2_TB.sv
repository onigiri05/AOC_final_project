`timescale 1ns/1ps

module Top_FC2_TB;

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
    // 2. 實例化子模組與級聯電路連接 (與 FC1 完全相同)
    // =================================================================
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
        .stat_valid_o(),
        .stat_ready_i(1'b1),
        .stat_token_idx_o(),
        .sum_sq_o()
    );

    // =================================================================
    // 3. 模擬片上 BRAM 與記憶體空間
    // =================================================================
    logic [31:0] act_bram_mem  [0:4095];
    logic [31:0] w_bram_mem    [0:4095];
    logic [31:0] bias_bram_mem [0:255];
    
    // FC2 專用：殘差輸入矩陣與黃金比對矩陣 (各 256 筆 uint8 資料)
    logic [7:0]  residual_mem   [0:255];
    logic [7:0]  golden_fc2_mem [0:255];
    logic [2047:0] packed_residual_tile; // 打包後的 2048-bit 殘差區塊

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

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
    // 4. 主測試激勵流程
    // =================================================================
    initial begin
        $dumpfile("top_fc2_pipeline.vcd");
        $dumpvars(0, Top_FC2_TB);

        // 加載 FC2 專用的十六進位測試資料
        // 測試時根據資料，自行修改路徑或名稱
        $readmemh("tb_fc2_act.hex",      act_bram_mem);
        $readmemh("tb_fc2_weight.hex",   w_bram_mem);
        $readmemh("tb_fc2_bias.hex",     bias_bram_mem);
        $readmemh("tb_fc2_residual.hex", residual_mem);   // 載入殘差
        $readmemh("tb_fc2_golden.hex",   golden_fc2_mem);
        $display("[FC2-TB] All FC2 verification hex files loaded.");

        // 將 256 個 8-bit 的殘差資料打包成 2048-bit 的一維向量
        for (int i = 0; i < 256; i++) begin
            packed_residual_tile[i*8 +: 8] = residual_mem[i];
        end

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

        // 配置 PPU 為 FFN FC2 測試模式
        ppu_mode_i = 2'b10;          // 2'b10 代表 FC2 階段 (Bypass GELU, 啟用 Residual Add)
        scaling_factor_i = 6'd2;     
        base_token_idx_i = 8'd0;
        channel_tile_idx_i = 5'd0;
        token_valid_mask_i = 16'hFFFF; 
        residual_tile_i = packed_residual_tile; // 餵入打包好的殘差
        data_tile_ready_i = 1;       

        #20;
        rst = 0;                     
        @(posedge clk);
        wait(module_ready == 1'b1);  
        $display("[FC2-TB] Systolic Array is idle and ready. Asserting 'en'...");

        en = 1'b1;
        @(posedge clk);
        en = 1'b0;                   

        fork
            begin
                #15000;
                $error("[FC2-TB] Simulation Timeout! Pipeline handshake broke down.");
                $finish;
            end
            begin
                wait(data_tile_valid_o && data_tile_ready_i);
                $display("[FC2-TB] Output Valid detected! Starting bit-level analysis against FC2 Golden...");
                check_result();
                $finish;
            end
        join
    end

    // =================================================================
    // 5. 自動化資料比對與驗證任務
    // =================================================================
    task automatic check_result();
        int error_count = 0;
        logic [7:0] current_hardware_out;
        
        for (int i = 0; i < 256; i++) begin
            current_hardware_out = data_tile_o[i*8 +: 8];
            
            if (current_hardware_out !== golden_fc2_mem[i]) begin
                if (error_count < 12) begin 
                    $error("Mismatch at Index %0d: Expected [0x%02X], Got [0x%02X]", 
                            i, golden_fc2_mem[i], current_hardware_out);
                end
                error_count++;
            end
        end

        $display("\n====================================================");
        if (error_count == 0) begin
            $display(">> [TEST PASSED] SUCCESS: All 256 elements perfectly match the FC2 Golden Model (Residual Add + Requant)!");
        end else begin
            $display(">> [TEST FAILED] ERROR: Found %0d total mismatches between FC2 pipeline and Golden Model.", error_count);
        end
        $display("====================================================\n");
    endtask

endmodule