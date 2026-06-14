`timescale 1ns/1ps
`include "ASIC.svh"

module PPU_TB;

    // ==========================================
    // 參數設定 (與 PPU 頂層一致)
    // ==========================================
    parameter int TOKEN_NUM       = 197;
    parameter int CHANNEL_NUM     = 384;
    parameter int TOKEN_TILE      = 16;
    parameter int CHANNEL_TILE    = 16;
    parameter int DATA_W          = 8;
    parameter int SUM_W           = 32;
    parameter int TOKEN_W         = 8;
    parameter int CHANNEL_TILE_W  = 5;
    parameter logic [7:0] ZERO_POINT = 8'd128;

    localparam int TILE_ELEMS = TOKEN_TILE * CHANNEL_TILE;

    // ==========================================
    // 訊號宣告
    // ==========================================
    logic clk;
    logic rst;

    logic [1:0] ppu_mode_i;
    logic [5:0] scaling_factor_i;

    logic tile_valid_i;
    logic tile_ready_o;
    logic [TOKEN_TILE*CHANNEL_TILE*`DATA_BITS-1:0] psum_tile_i;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0]     residual_tile_i;

    logic [TOKEN_W-1:0]        base_token_idx_i;
    logic [CHANNEL_TILE_W-1:0] channel_tile_idx_i;
    logic [TOKEN_TILE-1:0]     token_valid_mask_i;

    logic data_tile_valid_o;
    logic data_tile_ready_i;
    logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o;

    logic stat_valid_o;
    logic stat_ready_i;
    logic [TOKEN_W-1:0] stat_token_idx_o;
    logic [SUM_W-1:0]   sum_sq_o;

    // ==========================================
    // 實例化待測物 (DUT)
    // ==========================================
    PPU #(
        .TOKEN_NUM(TOKEN_NUM),
        .CHANNEL_NUM(CHANNEL_NUM),
        .TOKEN_TILE(TOKEN_TILE),
        .CHANNEL_TILE(CHANNEL_TILE),
        .DATA_W(DATA_W),
        .SUM_W(SUM_W),
        .TOKEN_W(TOKEN_W),
        .CHANNEL_TILE_W(CHANNEL_TILE_W),
        .ZERO_POINT(ZERO_POINT)
    ) dut (
        .* // 隱式連接所有同名訊號
    );

    // ==========================================
    // 時脈產生 (100MHz)
    // ==========================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ==========================================
    // 輔助任務 (Helper Tasks) - 打包 Tile 資料
    // ==========================================
    
    // 將整個 16x16 Psum Tile 填入相同的測試數值
    task pack_psum_tile(input logic signed [`DATA_BITS-1:0] val);
        integer i;
        begin
            for (i = 0; i < TILE_ELEMS; i++) begin
                psum_tile_i[i*`DATA_BITS +: `DATA_BITS] = val;
            end
        end
    endtask

    // 將整個 16x16 Residual Tile 填入相同的測試數值
    task pack_residual_tile(input logic [DATA_W-1:0] val);
        integer i;
        begin
            for (i = 0; i < TILE_ELEMS; i++) begin
                residual_tile_i[i*DATA_W +: DATA_W] = val;
            end
        end
    endtask

    // 發送一個 Tile 進 PPU
    task send_tile();
        begin
            tile_valid_i = 1'b1;
            wait(tile_ready_o == 1'b1);
            @(posedge clk);
            tile_valid_i = 1'b0;
        end
    endtask

    // ==========================================
    // 主測試流程
    // ==========================================
    initial begin
        // 波形匯出設定 (相容於一般模擬器與 Verilator)
        $dumpfile("ppu_waveform.vcd");
        $dumpvars(0, PPU_TB);

        // 1. 系統重置
        $display("----------------------------------------");
        $display("[System] Reset initializing...");
        rst = 1;
        ppu_mode_i = 2'b00;
        scaling_factor_i = 6'd0;
        tile_valid_i = 0;
        psum_tile_i = '0;
        residual_tile_i = '0;
        base_token_idx_i = 8'd0;
        channel_tile_idx_i = 5'd0;
        token_valid_mask_i = 16'hFFFF; // 全 token 有效
        data_tile_ready_i = 1;         // 模擬後級 Buffer 隨時 ready
        stat_ready_i = 1;              // 模擬 Stat SRAM 隨時 ready
        
        #20 rst = 0;
        @(posedge clk);
        $display("[System] Reset complete.");

        // ---------------------------------------------------------
        // 測試場景 1：Attention Output Phase (Mode = 2'b00)
        // 目標：驗證 Requant -> Residual Add (X + O) -> RMS Acc 觸發
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 1] Attention Output Phase (Mode 00)");
        ppu_mode_i = 2'b00;
        scaling_factor_i = 6'd2; // 右移 2 位
        base_token_idx_i = 8'd0;
        
        // 假設 Systolic Array 算出的 Psum=16，右移2位變 4，Requant後(加上128)為 132
        pack_psum_tile(32'sd16);  
        // 假設 Residual Buffer 讀出的 X=138 (真實值10)
        pack_residual_tile(8'd138); 
        // 預期結果: 132 + 138 - 128 = 142 (真實值14)

        // 送出連續的 Channel Tiles 來觸發 RMS 統計完成 (需送滿 24 個 Tile)
        for (int c = 0; c < 24; c++) begin
            channel_tile_idx_i = c[4:0];
            send_tile();
            @(posedge clk); // 模擬暫態延遲
        end
        $display("[Test 1] Sent 24 channel tiles. Waiting for Stat Output...");
        
        // 等待管線清空
        repeat(10) @(posedge clk);


        // ---------------------------------------------------------
        // 測試場景 2：FFN FC1 Phase (Mode = 2'b01)
        // 目標：驗證 GELU 啟動 -> Requant -> 無 Residual -> 無 RMS Acc
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 2] FFN FC1 Phase (Mode 01)");
        ppu_mode_i = 2'b01;
        scaling_factor_i = 6'd0; 
        
        // 傳入數值去戳 GELU 的 LUT。
        // data_in[15:8] 為索引。假設我們給 32'h0000_0500，索引為 5。
        // GELU ROM index 5 預設值為 8'h73。
        pack_psum_tile(32'h0000_0500); 
        pack_residual_tile(8'd0); // FC1 不使用 Residual

        channel_tile_idx_i = 5'd0;
        send_tile();
        $display("[Test 2] Sent FC1 tile. Output should bypass residual add.");
        
        repeat(5) @(posedge clk);


        // ---------------------------------------------------------
        // 測試場景 3：FFN FC2 Phase (Mode = 2'b10)
        // 目標：驗證 Requant -> Residual Add (X_mid + MLP_out) -> RMS Acc
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 3] FFN FC2 Phase (Mode 10)");
        ppu_mode_i = 2'b10;
        scaling_factor_i = 6'd1; // 右移 1 位
        
        // 假設 Psum=-20，右移1位=-10，Requant(加128)為 118
        pack_psum_tile(-32'sd20); 
        // 假設 X_mid=158 (真實值 30)
        pack_residual_tile(8'd158);
        // 預期結果: 118 + 158 - 128 = 148

        channel_tile_idx_i = 5'd23; // 直接測最後一個 Tile 確保 stat 會被觸發
        send_tile();
        
        repeat(10) @(posedge clk);

        $display("----------------------------------------");
        $display("[System] Simulation Finished.");
        $finish;
    end

endmodule