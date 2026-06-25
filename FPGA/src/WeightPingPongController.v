module WeightPingPongController #(
    parameter ADDR_WIDTH = 17,
    parameter BANK_ADDR_BITS = 10,
    parameter TOTAL_BANKS = 12,
    parameter TILE_ADDR_BITS = 6,
    parameter INIT_ACTIVE_VALID = 1'b0,
    parameter INIT_ACTIVE_BUFFER = 1'b0
)(
    input clk,
    input rst_n,

    // Systolic side
    input systolic_w_rd_en,
    input [ADDR_WIDTH-1:0] systolic_w_addr,
    output systolic_w_valid,
    output [31:0] systolic_w_data,

    // WeightMem read side
    output weight_rd_en,
    output [ADDR_WIDTH-1:0] weight_rd_addr,
    input [31:0] weight_rd_data,

    // Loader side. Addresses are relative to one tile (0..63 for TILE_ADDR_BITS=6).
    input loader_valid,
    output loader_ready,
    input [ADDR_WIDTH-1:0] loader_addr,
    input [31:0] loader_data,
    input [3:0] loader_byte_en,
    input loader_done,

    // WeightMem write side
    output weight_wr_en,
    output [ADDR_WIDTH-1:0] weight_wr_addr,
    output [31:0] weight_wr_data,
    output [3:0] weight_wr_byte_en,

    // Status/debug
    output reg active_buffer,
    output write_buffer,
    output active_buffer_valid,
    output write_buffer_full,
    output reg swap_done,
    output systolic_addr_error,
    output loader_addr_error
);

localparam HALF_BANKS = TOTAL_BANKS / 2;
localparam [ADDR_WIDTH-1:0] HALF_DEPTH_WORDS = HALF_BANKS << BANK_ADDR_BITS;
localparam [ADDR_WIDTH-1:0] TILE_WORDS = 1 << TILE_ADDR_BITS;

reg write_buffer_ready;
reg active_valid;
reg loader_done_d;
reg [ADDR_WIDTH-TILE_ADDR_BITS-1:0] active_tile_id;

wire loader_done_pulse = loader_done & ~loader_done_d;
wire systolic_addr_in_range = (systolic_w_addr < HALF_DEPTH_WORDS);
wire loader_addr_in_range = (loader_addr < TILE_WORDS);
wire [TILE_ADDR_BITS-1:0] systolic_word_in_tile = systolic_w_addr[TILE_ADDR_BITS-1:0];
wire [TILE_ADDR_BITS-1:0] loader_word_in_tile = loader_addr[TILE_ADDR_BITS-1:0];
wire [ADDR_WIDTH-TILE_ADDR_BITS-1:0] requested_tile_id = systolic_w_addr[ADDR_WIDTH-1:TILE_ADDR_BITS];
wire requested_active_tile = active_valid & (requested_tile_id == active_tile_id);
wire need_tile_swap = write_buffer_ready & (!active_valid | !requested_active_tile);
wire [ADDR_WIDTH-1:0] active_offset = active_buffer ? HALF_DEPTH_WORDS : {ADDR_WIDTH{1'b0}};
wire [ADDR_WIDTH-1:0] write_offset = write_buffer ? HALF_DEPTH_WORDS : {ADDR_WIDTH{1'b0}};

assign write_buffer = ~active_buffer;
assign active_buffer_valid = active_valid;
assign write_buffer_full = write_buffer_ready;

assign systolic_addr_error = requested_active_tile & ~systolic_addr_in_range;
assign loader_addr_error = loader_valid & ~loader_addr_in_range;

assign systolic_w_valid = requested_active_tile & systolic_addr_in_range;
assign systolic_w_data = weight_rd_data;

// Logical address selects the tile id; physical BRAM address uses one tile slot in the active ping/pong half.
assign weight_rd_en = systolic_w_rd_en & systolic_w_valid;
assign weight_rd_addr = active_offset + {{(ADDR_WIDTH-TILE_ADDR_BITS){1'b0}}, systolic_word_in_tile};

assign loader_ready = ~write_buffer_ready & loader_addr_in_range;
assign weight_wr_en = loader_valid & loader_ready;
assign weight_wr_addr = write_offset + {{(ADDR_WIDTH-TILE_ADDR_BITS){1'b0}}, loader_word_in_tile};
assign weight_wr_data = loader_data;
assign weight_wr_byte_en = loader_byte_en;

always @(posedge clk) begin
    if(!rst_n) begin
        active_buffer <= INIT_ACTIVE_BUFFER;
        active_valid <= INIT_ACTIVE_VALID;
        active_tile_id <= {ADDR_WIDTH-TILE_ADDR_BITS{1'b0}};
        write_buffer_ready <= 1'b0;
        loader_done_d <= 1'b0;
        swap_done <= 1'b0;
    end
    else begin
        loader_done_d <= loader_done;
        swap_done <= 1'b0;

        if(loader_done_pulse) begin
            write_buffer_ready <= 1'b1;
        end

        if(need_tile_swap) begin
            active_buffer <= write_buffer;
            active_tile_id <= requested_tile_id;
            active_valid <= 1'b1;
            write_buffer_ready <= 1'b0;
            swap_done <= 1'b1;
        end
    end
end

endmodule
