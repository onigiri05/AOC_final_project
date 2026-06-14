
`timescale 1ns/1ps
`include "../ASIC.svh"

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
    // 宣告測資記憶體 (儲存 256 個 elements)
    // ==========================================
    logic [31:0] psum_mem        [0:255];
    logic [7:0]  residual_mem    [0:255];
    logic [7:0]  golden_attn_mem [0:255];
    logic [7:0]  golden_fc1_mem  [0:255];

    // ==========================================
    // 讀取 Hex 測資檔案 (防呆檢查版)
    // ==========================================
    initial begin
        int fd;
        // 注意：這裡使用你電腦上的路徑，確保 Vivado 找得到。
        // 若路徑有變，請同步修改字串。
        
        fd = $fopen("psum_in.hex", "r");
        if (fd == 0) $fatal(1, "ERROR: Could not open psum_in.hex");
        $fclose(fd);
        
        fd = $fopen("residual_in.hex", "r");
        if (fd == 0) $fatal(1, "ERROR: residual_in.hex");
        $fclose(fd);
        
        fd = $fopen("golden_attn.hex", "r");
        if (fd == 0) $fatal(1, "ERROR: golden_attn.hex");
        $fclose(fd);
        
        fd = $fopen("golden_fc1.hex", "r");
        if (fd == 0) $fatal(1, "ERROR: golden_fc1.hex");
        $fclose(fd);

        // 確認檔案存在後，才將資料讀入記憶體
        $readmemh("psum_in.hex",     psum_mem);
        $readmemh("residual_in.hex", residual_mem);
        $readmemh("golden_attn.hex", golden_attn_mem);
        $readmemh("golden_fc1.hex",  golden_fc1_mem);
        
        $display("[System] Golden Patterns Loaded Successfully.");
    end

    // ==========================================
    // 輔助任務 (Helper Tasks)
    // ==========================================
    
    // 將陣列資料打包成 1D Vector 送給硬體
    task automatic load_tile_from_mem();
        for (int i = 0; i < TILE_ELEMS; i++) begin
            psum_tile_i[i*32 +: 32] = psum_mem[i];
            residual_tile_i[i*8 +: 8] = residual_mem[i];
        end
    endtask

    // 發送一個 Tile 進 PPU
    task automatic send_tile();
        begin
            tile_valid_i = 1'b1;
            wait(tile_ready_o == 1'b1);
            @(posedge clk);
            tile_valid_i = 1'b0;
        end
    endtask

    // 自動比對結果的 Task
    task automatic check_output(input logic [7:0] golden_mem [0:255], input string phase_name);
        int err_cnt = 0;
        
        // 等待硬體吐出有效資料
        wait(data_tile_valid_o && data_tile_ready_i);
        
        for (int i = 0; i < TILE_ELEMS; i++) begin
            if (data_tile_o[i*8 +: 8] !== golden_mem[i]) begin
                // 只印出前幾個錯誤，避免洗版
                if (err_cnt < 10) begin
                    $error("[%s] Mismatch at index %0d: Expected %02X, Got %02X", 
                            phase_name, i, golden_mem[i], data_tile_o[i*8 +: 8]);
                end else if (err_cnt == 10) begin
                    $display("[%s] ...more mismatches hidden.", phase_name);
                end
                err_cnt++;
            end
        end
        
        if (err_cnt == 0)
            $display(">> [%s] PASS: All 256 elements match the Golden Model!", phase_name);
        else
            $display(">> [%s] FAIL: Found %0d mismatches total.", phase_name, err_cnt);
            
        // 等待這個 clock 結束，避免重複觸發
        @(posedge clk);
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
        // 測試場景 1：Attention Output Phase
        // 目標：驗證 Requant -> Residual Add (X + O) -> RMS Acc 觸發
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 1] Attention Output Phase (Mode 00)");
        ppu_mode_i = 2'b00;
        scaling_factor_i = 6'd2; 
        base_token_idx_i = 8'd0;
        
        // 將記憶體的測資打入
        load_tile_from_mem();

        // 開啟 Fork-Join：一邊打入訊號，一邊等待接收並驗證
        fork
            begin
                // 送出第 0 個 Channel Tile (用來核對答案)
                channel_tile_idx_i = 5'd0;
                send_tile(); 
                
                // 為了觸發 RMS 統計完成，繼續送出剩餘的 23 個 Tile
                for (int c = 1; c < 24; c++) begin
                    channel_tile_idx_i = c[4:0];
                    send_tile();
                end
                $display("[Test 1] Sent 24 channel tiles. Waiting for Stat Output...");
            end
            begin
                // 呼叫比對 Task，傳入 Test 1 的黃金解答
                check_output(golden_attn_mem, "Attention Phase");
            end
        join

        // 檢查 RMS 統計輸出
        wait(stat_valid_o);
        $display(">> [Attention Phase] First Token RMS Stat: %0d", sum_sq_o);

        repeat(10) @(posedge clk);


        // ---------------------------------------------------------
        // 測試場景 2：FFN FC1 Phase 
        // 目標：驗證 GELU 啟動 -> Requant -> 無 Residual -> 無 RMS Acc
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 2] FFN FC1 Phase (Mode 01)");
        ppu_mode_i = 2'b01;
        scaling_factor_i = 6'd0; 
        
        // Psum 不變，沿用剛剛載入的 Tile，FC1 模式下會自動繞過 Residual Add
        
        fork
            begin
                channel_tile_idx_i = 5'd0;
                send_tile();
            end
            begin
                // 呼叫比對 Task，傳入 Test 2 的黃金解答
                check_output(golden_fc1_mem, "FC1 Phase (GELU)");
            end
        join
        
        repeat(10) @(posedge clk);


        // ---------------------------------------------------------
        // 測試場景 3：FFN FC2 Phase (Mode = 2'b10)
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 3] FFN FC2 Phase (Mode 10)");
        ppu_mode_i = 2'b10;
        scaling_factor_i = 6'd1; 
        
        // 這裡暫不比對解答，純粹測試管線能順利打通並產生資料
        for (int i = 0; i < TILE_ELEMS; i++) begin
            psum_tile_i[i*`DATA_BITS +: `DATA_BITS] = -32'sd20;
            residual_tile_i[i*DATA_W +: DATA_W] = 8'd158;
        end

        channel_tile_idx_i = 5'd23; // 直接測最後一個 Tile，確保能觸發 stat
        send_tile();
        
        repeat(15) @(posedge clk);

        $display("----------------------------------------");
        $display("[System] Simulation Finished.");
        $finish;
    end

endmodule