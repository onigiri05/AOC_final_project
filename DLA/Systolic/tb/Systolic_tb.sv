`timescale 1ns/1ps
`define MAX 100000
`define CYC 10 //10 ns
`include "Systolic.v"

module Systolic_tb;
reg clk, rst;

// ================ 模仿 BRAM 行為, 加上BRAM後拿掉 ================
reg [31:0] act_bram [6143:0];             //16 * 1536個 8'act
reg [31:0] w_bram [6143:0];
reg [31:0] bias_bram [15:0];                //32', 16 bias
reg [31:0] act_latch;                      //data晚addr一個cycle
reg [31:0] w_latch;                        //data晚addr一個cycle
reg [31:0] bias_latch;
// ==============================================================
wire [16:0] act_read_addr, w_read_addr, bias_read_addr;

reg en;
wire module_ready;
reg [6:0] K_CNT;
reg [6:0] k_tile_cnt;
reg [16:0] act_base_addr, w_base_addr, bias_base_addr;
reg [31:0] golden_rom [255:0];
wire opsum_valid;
wire [31:0] opsum;
integer cnt, err_cnt;
reg act_bram_valid, w_bram_valid, bias_bram_valid;

Systolic DUT (
    .clk(clk),
    .rst(rst),
    
    // ctrl & config for each m_tile
    .en(en),
    .module_ready(module_ready),      // module able to use
    .act_base_addr(act_base_addr), // BRAM total: 4096 * 140 B, / 8B per word, 17' addr
    .w_base_addr(w_base_addr),
    .bias_base_addr(bias_base_addr),
    .k_tile_cnt(k_tile_cnt), //at most 96, at FC2

    // BRAM(SRAM) input data, for each k-tile
    .act_bram_addr(act_read_addr), 
    .w_bram_addr(w_read_addr),
    .bias_bram_addr(bias_read_addr),
    .act_bram_valid(act_bram_valid), 
    .w_bram_valid(w_bram_valid),
    .bias_bram_valid(bias_bram_valid),
    .act_bram_data(act_latch),
    .w_bram_data(w_latch),
    .bias_bram_data(bias_latch),

    //ppu.sv
    .opsum_valid(opsum_valid),
    .opsum(opsum)
);

initial begin //clk
    clk = 1'b1;
    forever
        #(`CYC/2) clk = ~clk;
end

initial begin
    err_cnt = 0;
    rst = 1'b0;
    en = 1'b0;
    act_bram_valid = 1'b0;
    w_bram_valid = 1'b0;
    bias_bram_valid = 1'b0;


    #(`CYC) rst = 1'b1;
    #(`CYC*2) rst = 1'b0;
    
    // =========================================
    // 1. 觸發輸入 (在 negedge 給予，完美 Setup time)
    // =========================================
    @(negedge clk);
    en = 1'b1;
    k_tile_cnt = K_CNT;
    act_base_addr = 17'b0;
    w_base_addr = 17'b0;
    bias_base_addr = 17'b0;

    // =========================================
    // 2. 經過一個 Cycle 後拉低 (製造 1-cycle Pulse)
    // =========================================
    @(negedge clk); 
    en = 1'b0;
    // 故意給無效值，驗證硬體不會吃到錯的資料
    k_tile_cnt = 7'hf;
    act_base_addr = 17'hFFFF;
    w_base_addr = 17'hFFFF;
    
    //測中途valid改變會不會影響運算
    #(`CYC*5)act_bram_valid = 1'b1;
    #(`CYC)w_bram_valid = 1'b1;
    #(`CYC)bias_bram_valid = 1'b1;
    #(`CYC*18)act_bram_valid = 1'b0;
    #(`CYC*2)act_bram_valid = 1'b1;
    
    // =========================================
    // 3. 等待與比對輸出 (在 negedge 取樣，避開正緣資料變動)
    // =========================================
    
    // 在負緣等待 opsum_valid 變高
    while (!opsum_valid) @(negedge clk); 
    
    // 執行到這裡時，必定是某個 opsum_valid == 1 的「負緣」
    for(cnt=0; cnt < 256; cnt = cnt + 1) begin
        
        // 在負緣比對資料，此時 opsum 絕對已經穩定半個 cycle 了
        if(opsum !== golden_rom[cnt]) begin
            err_cnt = err_cnt + 1;
            $display("At cnt %d, opsum got 0x%x, golden got 0x%x", 
                            cnt, opsum, golden_rom[cnt]);
        end
        `ifdef SHOW
            $display("At cnt %d, opsum got 0x%x, golden got 0x%x", 
                            cnt, opsum, golden_rom[cnt]);
        `endif
        // 推進到「下一個負緣」，準備看下一筆資料
        @(negedge clk); 
    end
    
    if(err_cnt != 0) begin
        $display("⠄⣾⠟⢋⣉⣙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣶⠏⣰⣿⣿⣿⣿⣶⣌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⣴⠟⣋⣩⣭⣭⣿⣿⣶⣾⣿⣿⣿⣿⣿⣿⡿⠟⢛⣉⣉⣉⡙⠻⣿⣿⣿⣿⡄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⢁⣾⣿⣿⣿⣯⣛⠛⣿⣿⣿⣿⣿⣿⣿⣯⣤⡾⣿⣿⣿⣿⣿⣷⣤⠉⠻⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⠘⠋⠉⠉⣉⣉⣉⡙⠻⢿⣿⣿⣿⣿⣿⣿⠏⣴⣾⣥⣶⣶⣤⣍⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿");
        $display("⣿⠘⠁⠄⠄⢸⣿⣿⣿⡷⠂⣹⣿⣿⣿⣿⣿⠘⠛⠛⠛⠻⣿⣿⣿⣿⣿⣿⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⡄⠄⣀⣀⣚⣛⣉⣥⡴⠾⠿⢻⣿⣿⣿⣿⡇⢀⡋⠄⠄⣀⠉⠛⠿⢿⠟⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⣿⣄⠛⠿⠟⢻⣿⣡⣴⡶⠄⣾⣿⣿⣿⣿⣿⣌⡙⠿⣶⣶⣶⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿            Simulation Fail !!!     ⣿⣿⣿⣿⣿⣿");
        $display("⣿⣿⣿⣿⡿⠟⢋⣽⣿⠟⣠⣿⣿⣿⣿⣿⣿⡿⣿⣿⣶⣦⣶⣿⣿⣮⣭⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⡿⠏⣁⣀⢔⠿⠋⡏⢸⣿⣿⣿⡿⠛⠉⠉⣰⣟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⣶⠞⣉⣁⣉⣁⣈⡀⠘⠛⠛⠉⣤⣚⣛⣙⣋⣻⣇⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⠁⣼⡟⠻⠿⠿⠿⣿⣿⣦⣤⣄⣀⣉⣉⣉⣉⡛⢻⣀⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⣦⡈⠻⢷⣤⣙⠒⠶⢤⣭⣭⣭⣭⠍⢉⣩⣾⡿⠈⣿⣿⣿⣦⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⣿⣿⣿⣦⣤⣈⡛⠻⠿⠶⠶⠶⠶⠶⠚⣛⣉⣠⣴⣾⣿⣿⠿⢛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⢿⠟⣽⢿⣿⣿⣿⠻⠶⡶⢶⠲⣶⣿⣿⣿⣿⣿⣿⡟⢿⣷⣶⣿⣿⣿⣿⣿⠟⠋⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("err %d", err_cnt);
    end
    else begin
        $display("⠄⠄⠄⠄⢀⣠⣶⣶⣶⣤⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⢀⣠⣤⣄⡀⠄⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⠄⢠⣾⡟⠁⠄⠈⢻⣿⡀⠄⠄⠄⠄⠄⠄⠄⣼⣿⡿⠋⠉⠻⣷⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⠄⢸⣿⣷⣄⣀⣠⣿⣿⡇⠄⠄⠄⠄⠄⠄⢰⣿⣿⣇⠄⠄⢠⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⠄⢸⣿⣿⣿⣿⣿⣿⣿⣦⣤⣤⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⣿⣿⣿⣿⣿⡏⣍⡻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣩⡍⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿         Simulation Pass !!!     ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⣿⣿⣿⣿⣿⣇⢿⠻⠮⠭⠭⠭⢭⣭⣭⣭⣛⣭⣭⠶⠿⠛⣽⢱⣿⣿⣿⣿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⣿⣿⣿⣿⣿⣿⣦⢱⡀⠄⢰⣿⡇⠄⠄⠄⠄⠄⠄⠄⢀⣾⢇⣿⣿⣿⣿⡿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠻⢿⣿⣿⣿⢛⣭⣥⣭⣤⣼⣿⡇⠤⠤⠤⣤⣤⣤⡤⢞⣥⣿⣿⣿⣿⣿⠃⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⠄⣛⣛⠃⣿⣿⣿⣿⣿⣿⣿⢇⡙⠻⢿⣶⣶⣶⣾⣿⣿⣿⠿⢟⣛⠃⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⠄⣼⣿⣿⡘⣿⣿⣿⣿⣿⣿⡏⣼⣿⣿⣶⣬⣭⣭⣭⣭⣭⣴⣾⣿⣿⡄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⠄⣼⣿⣿⣿⣷⣜⣛⣛⣛⣛⣛⣀⡛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
        $display("⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣦⣭⣙⣛⣛⣩⣭⣭⣿⣿⣿⣿⣷⡀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿");
    end
    $finish();
end

always @(posedge clk) begin
    act_latch <= act_bram[act_read_addr];
    w_latch <= w_bram[w_read_addr];
    bias_latch <= bias_bram[bias_read_addr];
end

initial begin
    #(`MAX * `CYC)
    $display("=========================================================");
    $display("Time out! Please check your code and waveforms carefully.");
    $display("=========================================================");
    $finish();
end

initial begin
    `ifdef FSDB
    $fsdbDumpfile("top.fsdb"); 
    $fsdbDumpvars(0); //all signal
    `elsif FSDB_ALL
    $fsdbDumpfile("top.fsdb");
    $fsdbDumpvars(0, "+mda"); //expand memory/ array
    `endif

    `ifdef CASE1
        K_CNT = 7'd1;
        $readmemh("../hex/case1/act.hex", act_bram);
        $readmemh("../hex/case1/weight.hex", w_bram);
        $readmemh("../hex/case1/bias.hex", bias_bram);
        $readmemh("../hex/case1/golden.hex", golden_rom);
    `elsif CASE2
        K_CNT = 7'd2;
        $readmemh("../hex/case2/act.hex", act_bram);
        $readmemh("../hex/case2/weight.hex", w_bram);
        $readmemh("../hex/case2/bias.hex", bias_bram);
        $readmemh("../hex/case2/golden.hex", golden_rom);
    `elsif CASE3
        K_CNT = 7'd4;
        $readmemh("../hex/case3/act.hex", act_bram);
        $readmemh("../hex/case3/weight.hex", w_bram);
        $readmemh("../hex/case3/bias.hex", bias_bram);
        $readmemh("../hex/case3/golden.hex", golden_rom);
    `elsif CASE4
        K_CNT = 7'd96;
        $readmemh("../hex/case4/act.hex", act_bram);
        $readmemh("../hex/case4/weight.hex", w_bram);
        $readmemh("../hex/case4/bias.hex", bias_bram);
        $readmemh("../hex/case4/golden.hex", golden_rom);
    `else
        K_CNT = 7'd1;
        $readmemh("../hex/case0/act.hex", act_bram);
        $readmemh("../hex/case0/weight.hex", w_bram);
        $readmemh("../hex/case0/bias.hex", bias_bram);
        $readmemh("../hex/case0/golden.hex", golden_rom);
    `endif
end

endmodule