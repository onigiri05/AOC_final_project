module Requant_Unit (
input  logic signed [`DATA_BITS-1:0] data_in,         // 來自 GEMM 或 前級電路的 INT32 部分和
input  logic [5:0]                   scaling_factor,  // 右移位數暫存器 n (2^n)
output logic [7:0]                   data_out         // 輸出為以 128 為零點的 uint8 格式
);

logic signed [`DATA_BITS-1:0] data_shifted;
logic overflow_pos;
logic overflow_neg;

always_comb begin
    // 1. 執行高效率的 Power-of-two 算術右移
    data_shifted = data_in >>> scaling_factor;
    
    // 2. 正負溢位偵測 (以 8-bit 有符號數範圍 [-128, 127] 為硬體切分基準)
    overflow_pos = (~data_shifted[`DATA_BITS-1]) && (|(data_shifted[`DATA_BITS-2:7]));
    overflow_neg = (data_shifted[`DATA_BITS-1])  && ~(&(data_shifted[`DATA_BITS-2:7]));
    
    // 3. 飽和截斷並轉換為以 128 為零點的 uint8 格式
    if (overflow_pos) begin
        data_out = 8'hFF; // 上溢飽和最大值 255
    end else if (overflow_neg) begin
        data_out = 8'd0;  // 下溢飽和最小值 0
    end else begin
        // 翻轉最高位元 (Bit 7)，等價於將 2's complement 有符號數精準加上 128 偏移量
        data_out = {~data_shifted[7], data_shifted[6:0]};
    end
end

endmodule