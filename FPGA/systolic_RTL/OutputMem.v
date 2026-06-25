module OutputMem #(
    parameter ADDR_WIDTH = 8
)(
    input clk,
    input rst_n,

    // Port A: PL capture write port.
    input wr_en,
    input [ADDR_WIDTH-1:0] wr_addr,
    input [31:0] wr_data,
    input [3:0] wr_byte_en,

    // Port B: host/Python BRAM-like port.
    input host_en,
    input [ADDR_WIDTH-1:0] host_addr,
    input [31:0] host_wr_data,
    input [3:0] host_we,
    output [31:0] host_rd_data
);

wire [9:0] wr_addr_10 = {{(10-ADDR_WIDTH){1'b0}}, wr_addr};
wire [9:0] host_addr_10 = {{(10-ADDR_WIDTH){1'b0}}, host_addr};

`ifndef SYNTHESIS
reg [31:0] mem [0:1023];
reg [31:0] rd_data_r;
integer i;

assign host_rd_data = rd_data_r;

initial begin
    for(i = 0; i < 1024; i = i + 1) begin
        mem[i] = 32'd0;
    end
end

always @(posedge clk) begin
    if(!rst_n) begin
        rd_data_r <= 32'd0;
    end
    else if(host_en) begin
        rd_data_r <= mem[host_addr_10];
    end

    if(wr_en) begin
        if(wr_byte_en[0]) mem[wr_addr_10][7:0] <= wr_data[7:0];
        if(wr_byte_en[1]) mem[wr_addr_10][15:8] <= wr_data[15:8];
        if(wr_byte_en[2]) mem[wr_addr_10][23:16] <= wr_data[23:16];
        if(wr_byte_en[3]) mem[wr_addr_10][31:24] <= wr_data[31:24];
    end

    if(host_en) begin
        if(host_we[0]) mem[host_addr_10][7:0] <= host_wr_data[7:0];
        if(host_we[1]) mem[host_addr_10][15:8] <= host_wr_data[15:8];
        if(host_we[2]) mem[host_addr_10][23:16] <= host_wr_data[23:16];
        if(host_we[3]) mem[host_addr_10][31:24] <= host_wr_data[31:24];
    end
end
`else
wire [31:0] unused_doado;

RAMB36E1 #(
    .RDADDR_COLLISION_HWCONFIG("DELAYED_WRITE"),
    .SIM_COLLISION_CHECK("ALL"),
    .DOA_REG(0),
    .DOB_REG(0),
    .EN_ECC_READ("FALSE"),
    .EN_ECC_WRITE("FALSE"),
    .INIT_A(36'h000000000),
    .INIT_B(36'h000000000),
    .INIT_FILE("NONE"),
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
    .WRITE_MODE_A("WRITE_FIRST"),
    .WRITE_MODE_B("WRITE_FIRST")
) RAMB36E1_inst (
    .CASCADEOUTA(),
    .CASCADEOUTB(),
    .DBITERR(),
    .ECCPARITY(),
    .RDADDRECC(),
    .SBITERR(),
    .DOADO(unused_doado),
    .DOPADOP(),
    .DOBDO(host_rd_data),
    .DOPBDOP(),
    .CASCADEINA(1'b0),
    .CASCADEINB(1'b0),
    .INJECTDBITERR(1'b0),
    .INJECTSBITERR(1'b0),
    .ADDRARDADDR({1'b0, wr_addr_10, 5'b11111}),
    .CLKARDCLK(clk),
    .ENARDEN(wr_en),
    .REGCEAREGCE(1'b1),
    .RSTRAMARSTRAM(!rst_n),
    .RSTREGARSTREG(!rst_n),
    .WEA(wr_byte_en),
    .DIADI(wr_data),
    .DIPADIP(4'd0),
    .ADDRBWRADDR({1'b0, host_addr_10, 5'b11111}),
    .CLKBWRCLK(clk),
    .ENBWREN(host_en),
    .REGCEB(1'b1),
    .RSTRAMB(!rst_n),
    .RSTREGB(!rst_n),
    .WEBWE({4'b0000, host_we}),
    .DIBDI(host_wr_data),
    .DIPBDIP(4'd0)
);
`endif

endmodule
