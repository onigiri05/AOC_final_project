`timescale 1ns/1ps

// 384 x 16 gamma buffer.
// The physical shape is fixed at 2048 x 18 so Vivado synthesis maps it to
// one RAMB36E1. Simulation keeps an equivalent one-cycle synchronous model.
module Gamma_Buffer_BRAM #(
    parameter int CHANNEL_NUM = 384,
    parameter int CHANNEL_AW  = 9,
    parameter int DATA_W      = 16,
    parameter int BRAM_AW     = 11
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                 wr_valid_i,
    input  logic [CHANNEL_AW-1:0] wr_addr_i,
    input  logic signed [DATA_W-1:0] wr_data_i,

    input  logic [CHANNEL_AW-1:0] rd_addr_i,
    output logic signed [DATA_W-1:0] rd_data_o
);

    localparam int BRAM_DEPTH = 1 << BRAM_AW;
    localparam int ADDR_PAD_W = BRAM_AW - CHANNEL_AW;
    localparam logic [CHANNEL_AW-1:0] CHANNEL_NUM_L = CHANNEL_NUM;

    logic wr_in_range;
    logic rd_in_range;
    logic wr_fire;
    logic [BRAM_AW-1:0] wr_addr_bram;
    logic [BRAM_AW-1:0] rd_addr_bram;

    assign wr_in_range = (wr_addr_i < CHANNEL_NUM_L);
    assign rd_in_range = (rd_addr_i < CHANNEL_NUM_L);
    assign wr_fire = wr_valid_i && wr_in_range;
    assign wr_addr_bram = wr_in_range ? {{ADDR_PAD_W{1'b0}}, wr_addr_i} : '0;
    assign rd_addr_bram = rd_in_range ? {{ADDR_PAD_W{1'b0}}, rd_addr_i} : '0;

`ifdef SYNTHESIS
    logic rd_in_range_q;
    wire [31:0] bram_rd_word;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_in_range_q <= 1'b0;
        else
            rd_in_range_q <= rd_in_range;
    end

    assign rd_data_o = rd_in_range_q ? bram_rd_word[DATA_W-1:0] : '0;

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
        .READ_WIDTH_A(18),
        .READ_WIDTH_B(18),
        .WRITE_WIDTH_A(18),
        .WRITE_WIDTH_B(18),
        .RSTREG_PRIORITY_A("RSTREG"),
        .RSTREG_PRIORITY_B("RSTREG"),
        .SRVAL_A(36'h000000000),
        .SRVAL_B(36'h000000000),
        .SIM_DEVICE("7SERIES"),
        .WRITE_MODE_A("WRITE_FIRST"),
        .WRITE_MODE_B("READ_FIRST")
    ) u_ramb36e1 (
        .CASCADEOUTA(),
        .CASCADEOUTB(),
        .DBITERR(),
        .ECCPARITY(),
        .RDADDRECC(),
        .SBITERR(),
        .DOADO(),
        .DOPADOP(),
        .DOBDO(bram_rd_word),
        .DOPBDOP(),

        .CASCADEINA(1'b0),
        .CASCADEINB(1'b0),
        .INJECTDBITERR(1'b0),
        .INJECTSBITERR(1'b0),

        .ADDRARDADDR({1'b0, wr_addr_bram, 4'b0000}),
        .CLKARDCLK(clk),
        .ENARDEN(wr_fire),
        .REGCEAREGCE(1'b1),
        .RSTRAMARSTRAM(1'b0),
        .RSTREGARSTREG(1'b0),
        .WEA(wr_fire ? 4'hF : 4'h0),
        .DIADI({{(32-DATA_W){1'b0}}, wr_data_i}),
        .DIPADIP(4'b0000),

        .ADDRBWRADDR({1'b0, rd_addr_bram, 4'b0000}),
        .CLKBWRCLK(clk),
        .ENBWREN(1'b1),
        .REGCEB(1'b1),
        .RSTRAMB(1'b0),
        .RSTREGB(1'b0),
        .WEBWE(8'h00),
        .DIBDI(32'h00000000),
        .DIPBDIP(4'b0000)
    );
`else
    (* ram_style = "block" *) logic [17:0] mem [0:BRAM_DEPTH-1];



    always_ff @(posedge clk) begin
        if (wr_fire)
            mem[wr_addr_bram] <= {{(18-DATA_W){1'b0}}, wr_data_i};
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            rd_data_o <= '0;
        else if (rd_in_range)
            rd_data_o <= mem[rd_addr_bram][DATA_W-1:0];
        else
            rd_data_o <= '0;
    end
`endif

endmodule
