module WeightMem #(
    parameter INIT_FILE = "NONE",
    parameter NUM_BANKS = 12
)(
    input clk,
    input rst_n,

    input rd_en,
    input [16:0] rd_addr,
    output reg [31:0] rd_data,

    input wr_en,
    input [16:0] wr_addr,
    input [31:0] wr_data,
    input [3:0] wr_byte_en
);

wire [6:0] rd_bank = rd_addr[16:10];
wire [6:0] wr_bank = wr_addr[16:10];
wire [9:0] rd_bank_addr = rd_addr[9:0];
wire [9:0] wr_bank_addr = wr_addr[9:0];
localparam [6:0] NUM_BANKS_L = NUM_BANKS;
wire rd_in_range = (rd_bank < NUM_BANKS_L);

reg [6:0] rd_bank_d;
reg rd_in_range_d;
wire [NUM_BANKS*32-1:0] bank_rd_data;

always @(posedge clk) begin
    if(!rst_n) begin
        rd_bank_d <= 7'd0;
        rd_in_range_d <= 1'b0;
    end
    else if(rd_en) begin
        rd_bank_d <= rd_bank;
        rd_in_range_d <= rd_in_range;
    end
end

always @(*) begin
    if(rd_in_range_d) begin
        rd_data = bank_rd_data[rd_bank_d*32 +: 32];
    end
    else begin
        rd_data = 32'd0;
    end
end

genvar bank;
generate
    for(bank = 0; bank < NUM_BANKS; bank = bank + 1) begin : g_weight_bank
        localparam [6:0] BANK_ID = bank;
        wire bank_rd_en = rd_en && (rd_bank == BANK_ID);
        wire bank_wr_en = wr_en && (wr_bank == BANK_ID);

`ifndef SYNTHESIS
        reg [31:0] mem [0:1023];
        reg [31:0] rd_data_r;
        integer i;

        assign bank_rd_data[bank*32 +: 32] = rd_data_r;

        always @(posedge clk) begin
            if(!rst_n) begin
                rd_data_r <= 32'd0;
                for(i = 0; i < 1024; i = i + 1) begin
                    mem[i] <= 32'd0;
                end
            end
            else begin
                if(bank_rd_en) begin
                    rd_data_r <= mem[rd_bank_addr];
                end

                if(bank_wr_en) begin
                    if(wr_byte_en[0]) mem[wr_bank_addr][7:0] <= wr_data[7:0];
                    if(wr_byte_en[1]) mem[wr_bank_addr][15:8] <= wr_data[15:8];
                    if(wr_byte_en[2]) mem[wr_bank_addr][23:16] <= wr_data[23:16];
                    if(wr_byte_en[3]) mem[wr_bank_addr][31:24] <= wr_data[31:24];
                end
            end
        end
`else
        wire [31:0] unused_dobdo;

        if(bank == 0) begin : g_init_bank
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
                .DOADO(bank_rd_data[bank*32 +: 32]),
                .DOPADOP(),
                .DOBDO(unused_dobdo),
                .DOPBDOP(),
                .CASCADEINA(1'b0),
                .CASCADEINB(1'b0),
                .INJECTDBITERR(1'b0),
                .INJECTSBITERR(1'b0),
                .ADDRARDADDR({1'b0, rd_bank_addr, 5'b11111}),
                .CLKARDCLK(clk),
                .ENARDEN(bank_rd_en),
                .REGCEAREGCE(1'b1),
                .RSTRAMARSTRAM(!rst_n),
                .RSTREGARSTREG(!rst_n),
                .WEA(4'b0000),
                .DIADI(32'd0),
                .DIPADIP(4'd0),
                .ADDRBWRADDR({1'b0, wr_bank_addr, 5'b11111}),
                .CLKBWRCLK(clk),
                .ENBWREN(bank_wr_en),
                .REGCEB(1'b1),
                .RSTRAMB(!rst_n),
                .RSTREGB(!rst_n),
                .WEBWE({4'b0000, wr_byte_en}),
                .DIBDI(wr_data),
                .DIPBDIP(4'd0)
            );
        end
        else begin : g_clear_bank
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
                .WRITE_MODE_A("READ_FIRST"),
                .WRITE_MODE_B("WRITE_FIRST")
            ) RAMB36E1_inst (
                .CASCADEOUTA(),
                .CASCADEOUTB(),
                .DBITERR(),
                .ECCPARITY(),
                .RDADDRECC(),
                .SBITERR(),
                .DOADO(bank_rd_data[bank*32 +: 32]),
                .DOPADOP(),
                .DOBDO(unused_dobdo),
                .DOPBDOP(),
                .CASCADEINA(1'b0),
                .CASCADEINB(1'b0),
                .INJECTDBITERR(1'b0),
                .INJECTSBITERR(1'b0),
                .ADDRARDADDR({1'b0, rd_bank_addr, 5'b11111}),
                .CLKARDCLK(clk),
                .ENARDEN(bank_rd_en),
                .REGCEAREGCE(1'b1),
                .RSTRAMARSTRAM(!rst_n),
                .RSTREGARSTREG(!rst_n),
                .WEA(4'b0000),
                .DIADI(32'd0),
                .DIPADIP(4'd0),
                .ADDRBWRADDR({1'b0, wr_bank_addr, 5'b11111}),
                .CLKBWRCLK(clk),
                .ENBWREN(bank_wr_en),
                .REGCEB(1'b1),
                .RSTRAMB(!rst_n),
                .RSTREGB(!rst_n),
                .WEBWE({4'b0000, wr_byte_en}),
                .DIBDI(wr_data),
                .DIPBDIP(4'd0)
            );
        end
`endif
    end
endgenerate

endmodule
