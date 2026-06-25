module BiasMem #(
    parameter INIT_FILE = "NONE"
)(
    input clk,
    input rst_n,

    input rd_en,
    input [16:0] rd_addr,
    output [31:0] rd_data,

    input wr_en,
    input [16:0] wr_addr,
    input [31:0] wr_data,
    input [3:0] wr_byte_en
);

`ifndef SYNTHESIS
reg [31:0] mem [0:1023];
reg [31:0] rd_data_r;
integer i;

assign rd_data = rd_data_r;

always @(posedge clk) begin
    if(!rst_n) begin
        rd_data_r <= 32'd0;
        for(i = 0; i < 1024; i = i + 1) begin
            mem[i] <= 32'd0;
        end
    end
    else begin
        if(rd_en) begin
            rd_data_r <= mem[rd_addr[9:0]];
        end

        if(wr_en) begin
            if(wr_byte_en[0]) mem[wr_addr[9:0]][7:0] <= wr_data[7:0];
            if(wr_byte_en[1]) mem[wr_addr[9:0]][15:8] <= wr_data[15:8];
            if(wr_byte_en[2]) mem[wr_addr[9:0]][23:16] <= wr_data[23:16];
            if(wr_byte_en[3]) mem[wr_addr[9:0]][31:24] <= wr_data[31:24];
        end
    end
end
`else
wire [31:0] unused_dobdo;

RAMB36E1 #(
    .RDADDR_COLLISION_HWCONFIG("DELAYED_WRITE"),
    .SIM_COLLISION_CHECK("ALL"),
    .DOA_REG(0),
    .DOB_REG(0),
    .EN_ECC_READ("FALSE"),
    .EN_ECC_WRITE("FALSE"),
    .INIT_A(36'h000000000),
    .INIT_B(36'h000000000),
    .INIT_FILE(INIT_FILE),
    .RAM_MODE("TDP"),
    .RAM_EXTENSION_A("NONE"),
    .RAM_EXTENSION_B("NONE"),
    .READ_WIDTH_A(36),
    .READ_WIDTH_B(36),
    .WRITE_WIDTH_A(36),
    .WRITE_WIDTH_B(36),
    .RSTREG_PRIORITY_A("RSTREG"),
    .RSTREG_PRIORITY_B("RSTREG"),
    .SRVAL_A(36'h000000000),
    .SRVAL_B(36'h000000000),
    .SIM_DEVICE("7SERIES"),
    .WRITE_MODE_A("READ_FIRST"),
    .WRITE_MODE_B("WRITE_FIRST")
) RAMB36E1_inst (
    .CASCADEOUTA(),
    .CASCADEOUTB(),
    .DBITERR(),
    .ECCPARITY(),
    .RDADDRECC(),
    .SBITERR(),
    .DOADO(rd_data),
    .DOPADOP(),
    .DOBDO(unused_dobdo),
    .DOPBDOP(),
    .CASCADEINA(1'b0),
    .CASCADEINB(1'b0),
    .INJECTDBITERR(1'b0),
    .INJECTSBITERR(1'b0),
    .ADDRARDADDR({1'b0, rd_addr[9:0], 5'b11111}),
    .CLKARDCLK(clk),
    .ENARDEN(rd_en),
    .REGCEAREGCE(1'b1),
    .RSTRAMARSTRAM(!rst_n),
    .RSTREGARSTREG(!rst_n),
    .WEA(4'b0000),
    .DIADI(32'd0),
    .DIPADIP(4'd0),
    .ADDRBWRADDR({1'b0, wr_addr[9:0], 5'b11111}),
    .CLKBWRCLK(clk),
    .ENBWREN(wr_en),
    .REGCEB(1'b1),
    .RSTRAMB(!rst_n),
    .RSTREGB(!rst_n),
    .WEBWE({4'b0000, wr_byte_en}),
    .DIBDI(wr_data),
    .DIPBDIP(4'd0)
);
`endif

endmodule
