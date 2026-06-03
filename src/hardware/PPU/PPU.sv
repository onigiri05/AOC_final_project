`include "ASIC.svh"

module PPU (
    input  logic                         clk,
    input  logic                         rst,
    
    // 全域控制與模式配置
    input  logic [1:0]                   ppu_mode,          // 00: Attn Out, 01: FFN FC1, 10: FFN FC2
    input  logic [5:0]                   scaling_factor,    // Requant 移位值 n
    input  logic                         data_in_valid,     // 輸入資料有效flag
    input  logic                         token_start,       // 當前 Token 串流起點
    input  logic                         token_end,         // 當前 Token 串流終點
    input  logic [7:0]                   current_token_idx, // Token 索引 (0~196)

    // 數據輸入通路
    input  logic signed [`DATA_BITS-1:0] data_in,           // 來自 Systolic Array 的成果 (INT32)
    input  logic [7:0]                   residual_in,       // 來自 Global Buffer 的 Shortcut (INT8)

    // 數據輸出通路 (特徵圖)
    output logic [7:0]                   data_out,          // 輸出至 L2 Global Buffer / 下級 Activation Buffer
    output logic                         data_out_valid,    // 特徵圖輸出有效flag
    
    // Token Stat SRAM 寫入介面 (RMSNorm Fusion)
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
    // [Unit 2] GELU Unit 
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
    // [Unit 1] Requant Unit 
    // ─────────────────────────────────────────────────────────────────
    Requant_Unit u_Requant_Unit (
        .data_in(requant_in_mux),
        .scaling_factor(scaling_factor),
        .data_out(requant_to_residual)
    );

 

    // ─────────────────────────────────────────────────────────────────
    // 頂層 Datapath MUX 輸出排程
    // ─────────────────────────────────────────────────────────────────
    always_comb begin
        case (ppu_mode)
            2'b00:   data_out = res_add_to_mux;      // 狀況一：Attn Out + X = X_mid
            2'b01:   data_out = requant_to_residual; // 狀況二：FC1 + GELU -> Requant -> 直通下一級 (Zero BRAM Traffic)
            2'b10:   data_out = res_add_to_mux;      // 狀況三：FC2 + X_mid = X_out
            default: data_out = requant_to_residual;
        endcase
    end

    always_ff @(posedge clk) begin
        if (rst) data_out_valid <= 1'b0;
        else     data_out_valid <= data_in_valid;
    end


endmodule