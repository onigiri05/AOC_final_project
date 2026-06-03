module GELU_Unit (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         en,              // FC1 輸出時的致能信號
    input  logic signed [`DATA_BITS-1:0] data_in,         // 來自 FC1 的原始高精度輸出
    output logic signed [`DATA_BITS-1:0] data_out         // 輸出給 Requant Unit 的中間估計值
);

    logic [7:0] gelu_in_mapped;
    logic [7:0] gelu_lut_out;
    
    // 實體常數查找表 ROM (256 x 8-bit)
    logic [7:0] gelu_rom [0:255];
    
    // 映射排程：依據固定點軟體模擬範圍截取適當的 8 位元作為 LUT 索引
    assign gelu_in_mapped = data_in[15:8]; 

    initial begin
        // 此處依據 Python 腳本導出的常數寫入實體硬體 ROM
        // gelu_rom[0] = 8'h00; ...
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            gelu_lut_out <= 8'd0;
        end else if (en) begin
            gelu_lut_out <= gelu_rom[gelu_in_mapped];
        end
    end
    
    // 將 8-bit 查表結果擴展回 INT32 格式，無縫對接後級的 Requant Unit
    assign data_out = {24'b0, gelu_lut_out};

endmodule