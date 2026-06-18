`timescale 1ns/1ps

// ============================================================
// 模組：ViT_Accelerator_Top
// 功能：整合 RMSNorm, RowPacker, Systolic Array, PPU 與 Softmax
// ============================================================
module ViT_Accelerator_Top #(
    parameter int TOKEN_NUM       = 197,
    parameter int CHANNEL_NUM     = 384,
    parameter int TOKEN_TILE      = 16,
    parameter int CHANNEL_TILE    = 16,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int CHANNEL_TILE_W  = 5
)(
    input  logic clk,
    input  logic rst_n, // Active-low reset (系統主要使用)

    // ==========================================
    // 1. RMSNorm Input Interface (From GLB)
    // ==========================================
    input  logic               rms_start,
    output logic               rms_busy,   
    output logic               rms_done,   
    input  logic               rms_x_valid,
    output logic               rms_x_ready,
    input  logic [7:0]         rms_x_in,
    
    // SRAM Interfaces for RMSNorm
    output logic [7:0]         inv_rms_addr,
    input  logic [15:0]        inv_rms_data,
    output logic [8:0]         gamma_addr,
    input  logic signed [15:0] gamma_data,

    // ==========================================
    // 2. Systolic Array Control & Weight BRAM
    // ==========================================
    input  logic               sys_en,
    output logic               sys_module_ready,
    input  logic [16:0]        sys_w_base_addr,
    input  logic [16:0]        sys_bias_base_addr,
    input  logic [6:0]         sys_k_tile_cnt,

    output logic [16:0]        w_bram_addr,
    output logic [16:0]        bias_bram_addr,
    input  logic               w_bram_valid,
    input  logic               bias_bram_valid,
    input  logic [127:0]       w_bram_row,
    input  logic [127:0]       bias_bram_row,

    // ==========================================
    // 3. PPU Interface
    // ==========================================
    input  logic [1:0]         ppu_mode_i,
    input  logic [5:0]         ppu_scaling_factor_i,
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_residual_tile_i,
    
    input  logic [TOKEN_W-1:0]        ppu_base_token_idx_i,
    input  logic [CHANNEL_TILE_W-1:0] ppu_channel_tile_idx_i,
    input  logic [TOKEN_TILE-1:0]     ppu_token_valid_mask_i,
    
    output logic               ppu_data_tile_valid_o,
    input  logic               ppu_data_tile_ready_i,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] ppu_data_tile_o,
    
    output logic               stat_valid_o,
    input  logic               stat_ready_i,
    output logic [TOKEN_W-1:0] stat_token_idx_o,
    output logic [SUM_W-1:0]   sum_sq_o,

    // ==========================================
    // 4. Softmax Interface (For FlashAttention)
    // ==========================================
    input  logic               softmax_start,
    input  logic signed [31:0] softmax_score_row [0:207],
    input  logic [5:0]         softmax_q_shift,
    input  logic [5:0]         softmax_k_shift,
    input  logic [207:0]       softmax_mask,
    output logic signed [7:0]  softmax_attention_row [0:207],
    output logic               softmax_done
);

    // 反向 Reset 供 Systolic Array 使用 (它使用 active-high)
    logic rst_high;
    assign rst_high = ~rst_n;

    // -----------------------------------------------------------------
    // [區塊 A] Streaming RMSNorm -> RowPacker -> Activation BRAM
    // -----------------------------------------------------------------
    logic rms_y_valid, rms_y_ready, rms_y_last;
    logic [7:0] rms_y_out;

    Streaming_RMSNorm_Unit u_rmsnorm (
        .clk(clk),
        .rst_n(rst_n),
        .start(rms_start),
        .busy(rms_busy),       
        .done(rms_done),
        .x_valid(rms_x_valid),
        .x_ready(rms_x_ready),
        .x_in(rms_x_in),
        .inv_rms_addr(inv_rms_addr),
        .inv_rms_data(inv_rms_data),
        .gamma_addr(gamma_addr),
        .gamma_data(gamma_data),
        .y_valid(rms_y_valid),
        .y_ready(rms_y_ready),
        .y_last(rms_y_last),
        .y_out(rms_y_out)
    );

    logic        act_wr_valid;
    logic        act_wr_ready;
    logic [16:0] act_wr_addr;
    logic [127:0] act_wr_row;

    Streaming_RMSNorm_RowPacker u_packer (
        .clk(clk),
        .rst_n(rst_n),
        .start_i(rms_start),
        .base_addr_i(17'd0), // 預設從 0 開始寫
        .s_data_i(rms_y_out),
        .s_valid_i(rms_y_valid),
        .s_ready_o(rms_y_ready),
        .s_last_i(rms_y_last),
        .act_wr_valid_o(act_wr_valid),
        .act_wr_ready_i(act_wr_ready),
        .act_wr_addr_o(act_wr_addr),
        .act_wr_row_o(act_wr_row),
        .act_wr_last_o()
    );

    // 內部實例化 Activation BRAM (扮演 L2 Buffer 的角色)
    // 大小設定為 4096 * 128-bit
    logic [127:0] activation_bram [0:4095];
    logic [16:0]  sys_act_bram_addr;
    logic [127:0] sys_act_bram_row;
    logic         sys_act_bram_valid;

    assign act_wr_ready = 1'b1; // BRAM always ready to write

    always_ff @(posedge clk) begin
        // Write Port (From Packer)
        if (act_wr_valid && act_wr_ready) begin
            activation_bram[act_wr_addr[11:0]] <= act_wr_row;
        end
        // Read Port (To Systolic)
        sys_act_bram_row <= activation_bram[sys_act_bram_addr[11:0]];
        sys_act_bram_valid <= 1'b1; // Read takes 1 cycle
    end

    // -----------------------------------------------------------------
    // [區塊 B] Systolic Array
    // -----------------------------------------------------------------
    logic        sys_opsum_valid;
    logic [31:0] sys_opsum;

    Systolic u_systolic (
        .clk(clk),
        .rst(rst_high),             // Active-high rst
        .en(sys_en),
        .module_ready(sys_module_ready),
        .act_base_addr(17'd0),      // 從 BRAM offset 0 讀取
        .w_base_addr(sys_w_base_addr),
        .bias_base_addr(sys_bias_base_addr),
        .k_tile_cnt(sys_k_tile_cnt),
        
        .act_bram_addr(sys_act_bram_addr),
        .w_bram_addr(w_bram_addr),
        .bias_bram_addr(bias_bram_addr),
        
        .act_bram_valid(sys_act_bram_valid),
        .w_bram_valid(w_bram_valid),
        .bias_bram_valid(bias_bram_valid),
        
        .act_bram_row(sys_act_bram_row),
        .w_bram_row(w_bram_row),
        .bias_bram_row(bias_bram_row),
        
        .opsum_valid(sys_opsum_valid),
        .opsum(sys_opsum)
    );

    // -----------------------------------------------------------------
    // [區塊 C] Psum Deserializer (Systolic 轉 PPU 橋接器)
    // 將 256 個連續的 32-bit opsum 收集成一個 8192-bit 的 Tile
    // -----------------------------------------------------------------
    logic [TOKEN_TILE*CHANNEL_TILE*32-1:0] ppu_psum_tile_reg;
    logic [8:0] psum_counter; // 0~255
    logic       ppu_tile_valid_reg;
    logic       ppu_tile_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            psum_counter <= '0;
            ppu_tile_valid_reg <= 1'b0;
        end else begin
            if (sys_opsum_valid) begin
                // 依序將 32-bit 資料存入 8192-bit 暫存器的對應位置
                ppu_psum_tile_reg[psum_counter * 32 +: 32] <= sys_opsum;
                
                if (psum_counter == 9'd255) begin
                    psum_counter <= '0;
                    ppu_tile_valid_reg <= 1'b1; // 收集滿 256 個，觸發 PPU
                end else begin
                    psum_counter <= psum_counter + 1'b1;
                end
            end else if (ppu_tile_ready && ppu_tile_valid_reg) begin
                // PPU 成功接收後，降下 valid 訊號
                ppu_tile_valid_reg <= 1'b0;
            end
        end
    end

    // -----------------------------------------------------------------
    // [區塊 D] Post-Processing Unit (PPU)
    // -----------------------------------------------------------------
    PPU u_ppu (
        .clk(clk),
        .rst(rst_high), //  PPU 是用 rst (active-high)
        .ppu_mode_i(ppu_mode_i),
        .scaling_factor_i(ppu_scaling_factor_i),
        
        .tile_valid_i(ppu_tile_valid_reg),
        .tile_ready_o(ppu_tile_ready),
        .psum_tile_i(ppu_psum_tile_reg),
        .residual_tile_i(ppu_residual_tile_i),
        
        .base_token_idx_i(ppu_base_token_idx_i),
        .channel_tile_idx_i(ppu_channel_tile_idx_i),
        .token_valid_mask_i(ppu_token_valid_mask_i),
        
        .data_tile_valid_o(ppu_data_tile_valid_o),
        .data_tile_ready_i(ppu_data_tile_ready_i),
        .data_tile_o(ppu_data_tile_o),
        
        .stat_valid_o(stat_valid_o),
        .stat_ready_i(stat_ready_i),
        .stat_token_idx_o(stat_token_idx_o),
        .sum_sq_o(sum_sq_o)
    );

    // -----------------------------------------------------------------
    // [區塊 E] Softmax Unit (For MHSA FlashAttention)
    // 依據架構圖，此單元獨立接在 Attention Controller 旁
    // -----------------------------------------------------------------
    Softmax_Unit u_softmax (
        .clk(clk),
        .rst_n(rst_n),
        .start(softmax_start),
        .score_row(softmax_score_row),
        .q_shift(softmax_q_shift),
        .k_shift(softmax_k_shift),
        .mask(softmax_mask),
        .attention_row(softmax_attention_row),
        .done(softmax_done)
    );

endmodule