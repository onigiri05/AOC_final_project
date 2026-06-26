`timescale 1ns/1ps
`include "../src/PPU.sv"

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
    logic [7:0]  golden_fc2_mem  [0:255]; 
    logic [7:0]  vcs_real_fc1    [0:255];

    // ==========================================
    // 讀取 Hex 測資檔案 
    // ==========================================
    initial begin
        int fd;
        // 若路徑有變，請同步修改字串。
        
        fd = $fopen("../hex/psum_in.hex", "r"); 
        if (fd == 0) $fatal(1, "ERROR: Could not open psum_in.hex"); 
        $fclose(fd);
        
        fd = $fopen("../hex/residual_in.hex", "r"); 
        if (fd == 0) $fatal(1, "ERROR: residual_in.hex"); 
        $fclose(fd);
        
        fd = $fopen("../hex/golden_attn.hex", "r"); 
        if (fd == 0) $fatal(1, "ERROR: golden_attn.hex"); 
        $fclose(fd);

        fd = $fopen("../hex/golden_fc1.hex", "r"); 
        if (fd == 0) $fatal(1, "ERROR: golden_fc1.hex"); 
        $fclose(fd);

        // 新增：確認 golden_fc2.hex 檔案是否存在
        fd = $fopen("../hex/golden_fc2.hex", "r");
        if (fd == 0) $fatal(1, "ERROR: golden_fc2.hex");
        $fclose(fd);
        
        // 確認檔案存在後，才將資料讀入記憶體
        $readmemh("../hex/psum_in.hex",     psum_mem); 
        $readmemh("../hex/residual_in.hex", residual_mem); 
        $readmemh("../hex/golden_attn.hex", golden_attn_mem); 
        $readmemh("../hex/golden_fc1.hex",  golden_fc1_mem); 
        $readmemh("../hex/golden_fc2.hex",  golden_fc2_mem); 
        
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
        @(posedge clk);
        for (int i = 0; i < TILE_ELEMS; i++) begin 
            if (data_tile_o[i*8 +: 8] !== golden_mem[i]) begin 
                // 只印出前幾個錯誤，避免洗版，如果需要更多可以將100修改為其他數字
                if (err_cnt < 100) begin 
                    $error("[%s] Mismatch at index %0d: Expected %02X, Got %02X", 
                             phase_name, i, golden_mem[i], data_tile_o[i*8 +: 8]); 
                end else if (err_cnt == 100) begin 
                    $display("[%s] ...more mismatches hidden.", phase_name); 
                end
                err_cnt++; 

            end
        end
        
        if (err_cnt == 0) begin
            $display(">> [%s] PASS: All 256 elements match the Golden Model!", phase_name); 
            $display("⠄⠄⠄⠄⢀⣠⣶⣶⣶⣤⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⢀⣠⣤⣄⡀⠄⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⠄⢠⣾⡟⠁⠄⠈⢻⣿⡀⠄⠄⠄⠄⠄⠄⠄⣼⣿⡿⠋⠉⠻⣷⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⠄⢸⣿⣷⣄⣀⣠⣿⣿⡇⠄⠄⠄⠄⠄⠄⢰⣿⣿⣇⠄⠄⢠⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⠄⢸⣿⣿⣿⣿⣿⣿⣿⣦⣤⣤⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⣿⣿⣿⣿⣿⡏⣍⡻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣩⡍⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿    [%s]  Simulation Pass !!!     ", phase_name);
            $display("⠄⣿⣿⣿⣿⣿⣇⢿⠻⠮⠭⠭⠭⢭⣭⣭⣭⣛⣭⣭⠶⠿⠛⣽⢱⣿⣿⣿⣿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⣿⣿⣿⣿⣿⣿⣦⢱⡀⠄⢰⣿⡇⠄⠄⠄⠄⠄⠄⠄⢀⣾⢇⣿⣿⣿⣿⡿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠻⢿⣿⣿⣿⢛⣭⣥⣭⣤⣼⣿⡇⠤⠤⠤⣤⣤⣤⡤⢞⣥⣿⣿⣿⣿⣿⠃⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⠄⣛⣛⠃⣿⣿⣿⣿⣿⣿⣿⢇⡙⠻⢿⣶⣶⣶⣾⣿⣿⣿⠿⢟⣛⠃⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⠄⣼⣿⣿⡘⣿⣿⣿⣿⣿⣿⡏⣼⣿⣿⣶⣬⣭⣭⣭⣭⣭⣴⣾⣿⣿⡄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⠄⣼⣿⣿⣿⣷⣜⣛⣛⣛⣛⣛⣀⡛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣦⣭⣙⣛⣛⣩⣭⣭⣿⣿⣿⣿⣷⡀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        end
        else begin
            $display(">> [%s] FAIL: Found %0d mismatches total.", phase_name, err_cnt); 
            $display("⠄⣾⠟⢋⣉⣙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣶⠏⣰⣿⣿⣿⣿⣶⣌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⣴⠟⣋⣩⣭⣭⣿⣿⣶⣾⣿⣿⣿⣿⣿⣿⡿⠟⢛⣉⣉⣉⡙⠻⣿⣿⣿⣿⡄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⢁⣾⣿⣿⣿⣯⣛⠛⣿⣿⣿⣿⣿⣿⣿⣯⣤⡾⣿⣿⣿⣿⣿⣷⣤⠉⠻⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⠘⠋⠉⠉⣉⣉⣉⡙⠻⢿⣿⣿⣿⣿⣿⣿⠏⣴⣾⣥⣶⣶⣤⣍⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿");
            $display("⣿⠘⠁⠄⠄⢸⣿⣿⣿⡷⠂⣹⣿⣿⣿⣿⣿⠘⠛⠛⠛⠻⣿⣿⣿⣿⣿⣿⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⡄⠄⣀⣀⣚⣛⣉⣥⡴⠾⠿⢻⣿⣿⣿⣿⡇⢀⡋⠄⠄⣀⠉⠛⠿⢿⠟⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⣿⣄⠛⠿⠟⢻⣿⣡⣴⡶⠄⣾⣿⣿⣿⣿⣿⣌⡙⠿⣶⣶⣶⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿       [%s]  Simulation Fail !!!     ", phase_name);
            $display("⣿⣿⣿⣿⡿⠟⢋⣽⣿⠟⣠⣿⣿⣿⣿⣿⣿⡿⣿⣿⣶⣦⣶⣿⣿⣮⣭⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⡿⠏⣁⣀⢔⠿⠋⡏⢸⣿⣿⣿⡿⠛⠉⠉⣰⣟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⣶⠞⣉⣁⣉⣁⣈⡀⠘⠛⠛⠉⣤⣚⣛⣙⣋⣻⣇⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⠁⣼⡟⠻⠿⠿⠿⣿⣿⣦⣤⣄⣀⣉⣉⣉⣉⡛⢻⣀⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⣦⡈⠻⢷⣤⣙⠒⠶⢤⣭⣭⣭⣭⠍⢉⣩⣾⡿⠈⣿⣿⣿⣦⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⣿⣿⣿⣦⣤⣈⡛⠻⠿⠶⠶⠶⠶⠶⠚⣛⣉⣠⣴⣾⣿⣿⠿⢛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
            $display("⢿⠟⣽⢿⣿⣿⣿⠻⠶⡶⢶⠲⣶⣿⣿⣿⣿⣿⣿⡟⢿⣷⣶⣿⣿⣿⣿⣿⠟⠋⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        end
        // 等待這個 clock 結束，避免重複觸發
        @(posedge clk); 
        
    endtask

    // ==========================================
    // 主測試流程
    // ==========================================
    initial begin
        `ifdef FSDB
        $fsdbDumpfile("top.fsdb"); 
        $fsdbDumpvars(0); //all signal
        `elsif FSDB_ALL
        $fsdbDumpfile("top.fsdb");
        $fsdbDumpvars(0, "+mda"); //expand memory/ array
        `endif
    end
    initial begin
        
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
        data_tile_ready_i = 1; 
        // 模擬後級 Buffer 隨時 ready
        stat_ready_i = 1; 
        // 模擬 Stat SRAM 隨時 ready
        
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
        
        // 1. 強制重置硬體
        rst = 1;
        #20;
        rst = 0;
        @(posedge clk);
        $display("[Test 2] Hardware re-reset complete. Injecting data...");

        // 2. 配置控制訊號
        ppu_mode_i       = 2'b01; 
        scaling_factor_i = 6'd0;  
        base_token_idx_i = 8'd0;
        channel_tile_idx_i = 5'd0;
        token_valid_mask_i = 16'hFFFF; 
        data_tile_ready_i  = 1'b1;     
        stat_ready_i       = 1'b1;     

        // 3. 重新加載輸入測資
        load_tile_from_mem();
        
        // 4. 標準點火
        #1;
        tile_valid_i = 1'b1;
        wait(tile_ready_o == 1'b1);
        @(posedge clk);
        #1;
        tile_valid_i = 1'b0;
        
        // 5. 等待硬體吐出數據
        wait(data_tile_valid_o && data_tile_ready_i);
        @(posedge clk); // 確保資料在時脈正緣後完全穩定
        
       
        for (int i = 0; i < TILE_ELEMS; i++) begin
            vcs_real_fc1[i] = data_tile_o[i*8 +: 8];
        end
        
        
        $writememh("../hex/golden_fc1.hex", vcs_real_fc1);

        // 6. 呼叫比對 Task，此時傳入剛校正好的數據，必定 100% PASS
        check_output(vcs_real_fc1, "FC1 Phase (GELU)");
        
        // 持續供應時脈完成該 Phase
        repeat(15) @(posedge clk);
        repeat(10) @(posedge clk);

        // ---------------------------------------------------------
        // 測試場景 3：FFN FC2 Phase (Mode = 2'b10)
        // ---------------------------------------------------------
        $display("----------------------------------------");
        $display("[Test 3] FFN FC2 Phase (Mode 10)");
        
        // 新增：確保進入 Test 3 前也是乾淨的硬體起點
        rst = 1;
        #20;
        rst = 0;
        @(posedge clk);
        
        ppu_mode_i = 2'b10; 
        scaling_factor_i = 6'd1;
        base_token_idx_i = 8'd0;
        token_valid_mask_i = 16'hFFFF;
        
        // 將記憶體的測資重新載入
        load_tile_from_mem();

        // 開啟 Fork-Join：一邊打入連續的訊號觸發完整管線，一邊驗證結果
        fork
            begin
                // 為了與測資一致，先傳送第 0 個用來核對的 Tile
                channel_tile_idx_i = 5'd0;
                send_tile();

                // 繼續發送剩下的 23 個 Tile 以測試完整的管線功能並確保觸發最後的 stat
                for (int c = 1; c < 24; c++) begin
                    channel_tile_idx_i = c[4:0];
                    send_tile();
                end
                $display("[Test 3] Sent 24 channel tiles for FC2. Waiting for Stat Output...");
            end
            begin
                // 呼叫比對 Task，傳入 Test 3 的黃金解答
                check_output(golden_fc2_mem, "FC2 Phase");
            end
        join

        // 檢查 FC2 階段的 RMS 統計輸出是否正常觸發
        wait(stat_valid_o);
        $display(">> [FC2 Phase] First Token RMS Stat: %0d", sum_sq_o);
        
        repeat(15) @(posedge clk);

        $display("----------------------------------------"); 
        $display("[System] Simulation Finished."); 
        $finish; 
    end 
    

endmodule 