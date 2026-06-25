module SystolicSystemCore #(
    parameter ADDR_WIDTH = 17,
    parameter OUTPUT_ADDR_WIDTH = 8,
    parameter OUTPUT_WORDS = 256
)(
    input clk,
    input rst_n,

    // Compute control/config. In a PS design, drive these from AXI-Lite
    // registers, not from package pins.
    input start,
    input [6:0] k_tile_cnt,
    input [ADDR_WIDTH-1:0] act_base_addr,
    input [ADDR_WIDTH-1:0] w_base_addr,
    input [ADDR_WIDTH-1:0] bias_base_addr,
    output module_ready,

    // Unified activation/bias/weight input loader register-write side.
    input input_wr_valid,
    output input_wr_ready,
    input [3:0] input_wr_addr,
    input [31:0] input_wr_data,
    input [3:0] input_wr_strb,
    output input_busy,
    output input_done,
    output input_error,
    output [1:0] input_active_target,
    output [ADDR_WIDTH:0] input_word_count,

    // Output memory host/Python BRAM-like port.
    input output_host_en,
    input [OUTPUT_ADDR_WIDTH-1:0] output_host_addr,
    input [31:0] output_host_wr_data,
    input [3:0] output_host_we,
    output [31:0] output_host_rd_data,

    // Status/debug.
    output output_capture_busy,
    output output_capture_done,
    output output_capture_overflow,
    output [OUTPUT_ADDR_WIDTH:0] output_capture_count,
    output weight_active_buffer_valid,
    output weight_write_buffer_full,
    output weight_swap_done,
    output weight_systolic_addr_error,
    output weight_loader_addr_error,

    output profile_systolic_load_active,
    output profile_systolic_op_active,
    output profile_systolic_weight_stall,
    output profile_weight_load_busy,
    output profile_weight_word_accept,
    output profile_act_bram_read,
    output profile_weight_bram_read,
    output profile_bias_bram_read,
    output profile_output_word_write
);

wire act_bram_rd_en;
wire w_bram_rd_en;
wire bias_bram_rd_en;
wire [ADDR_WIDTH-1:0] act_bram_addr;
wire [ADDR_WIDTH-1:0] w_bram_addr;
wire [ADDR_WIDTH-1:0] bias_bram_addr;
wire w_bram_valid;
wire [31:0] act_bram_data;
wire [31:0] w_bram_data;
wire [31:0] bias_bram_data;
wire opsum_valid;
wire [31:0] opsum;

wire act_wr_en;
wire [ADDR_WIDTH-1:0] act_wr_addr;
wire [31:0] act_wr_data;
wire [3:0] act_wr_byte_en;
wire bias_wr_en;
wire [ADDR_WIDTH-1:0] bias_wr_addr;
wire [31:0] bias_wr_data;
wire [3:0] bias_wr_byte_en;

wire weight_mem_rd_en;
wire [ADDR_WIDTH-1:0] weight_mem_rd_addr;
wire [31:0] weight_mem_rd_data;
wire weight_mem_wr_en;
wire [ADDR_WIDTH-1:0] weight_mem_wr_addr;
wire [31:0] weight_mem_wr_data;
wire [3:0] weight_mem_wr_byte_en;

wire weight_loader_valid;
wire weight_loader_ready;
wire [ADDR_WIDTH-1:0] weight_loader_addr;
wire [31:0] weight_loader_data;
wire [3:0] weight_loader_byte_en;
wire weight_loader_done;
wire weight_active_buffer;
wire weight_write_buffer;

wire output_mem_wr_en;
wire [OUTPUT_ADDR_WIDTH-1:0] output_mem_wr_addr;
wire [31:0] output_mem_wr_data;
wire [3:0] output_mem_wr_byte_en;

assign profile_weight_load_busy = input_busy & (input_active_target == 2'd2);
assign profile_weight_word_accept = weight_loader_valid & weight_loader_ready;
assign profile_act_bram_read = act_bram_rd_en;
assign profile_weight_bram_read = w_bram_rd_en;
assign profile_bias_bram_read = bias_bram_rd_en;
assign profile_output_word_write = output_mem_wr_en;

Systolic u_systolic (
    .clk(clk),
    .rst_n(rst_n),

    .en(start),
    .module_ready(module_ready),
    .act_base_addr(act_base_addr),
    .w_base_addr(w_base_addr),
    .bias_base_addr(bias_base_addr),
    .k_tile_cnt(k_tile_cnt),

    .act_bram_rd_en(act_bram_rd_en),
    .w_bram_rd_en(w_bram_rd_en),
    .bias_bram_rd_en(bias_bram_rd_en),
    .act_bram_addr(act_bram_addr),
    .w_bram_addr(w_bram_addr),
    .bias_bram_addr(bias_bram_addr),
    .w_bram_valid(w_bram_valid),
    .act_bram_data(act_bram_data),
    .w_bram_data(w_bram_data),
    .bias_bram_data(bias_bram_data),

    .opsum_valid(opsum_valid),
    .opsum(opsum),

    .profile_load_active(profile_systolic_load_active),
    .profile_op_active(profile_systolic_op_active),
    .profile_weight_stall(profile_systolic_weight_stall)
);

ActivationMem #(
    .INIT_FILE("NONE"),
    .NUM_BANKS(76)
) u_activation_mem (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(act_bram_rd_en),
    .rd_addr(act_bram_addr),
    .rd_data(act_bram_data),
    .wr_en(act_wr_en),
    .wr_addr(act_wr_addr),
    .wr_data(act_wr_data),
    .wr_byte_en(act_wr_byte_en)
);

BiasMem #(
    .INIT_FILE("NONE")
) u_bias_mem (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(bias_bram_rd_en),
    .rd_addr(bias_bram_addr),
    .rd_data(bias_bram_data),
    .wr_en(bias_wr_en),
    .wr_addr(bias_wr_addr),
    .wr_data(bias_wr_data),
    .wr_byte_en(bias_wr_byte_en)
);

InputLoadFSM #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .TILE_ADDR_BITS(6)
) u_input_loader_fsm (
    .clk(clk),
    .rst_n(rst_n),

    .host_wr_valid(input_wr_valid),
    .host_wr_ready(input_wr_ready),
    .host_wr_addr(input_wr_addr),
    .host_wr_data(input_wr_data),
    .host_wr_strb(input_wr_strb),

    .host_busy(input_busy),
    .host_done(input_done),
    .host_error(input_error),
    .host_active_target(input_active_target),
    .host_word_count(input_word_count),

    .act_wr_en(act_wr_en),
    .act_wr_addr(act_wr_addr),
    .act_wr_data(act_wr_data),
    .act_wr_byte_en(act_wr_byte_en),

    .bias_wr_en(bias_wr_en),
    .bias_wr_addr(bias_wr_addr),
    .bias_wr_data(bias_wr_data),
    .bias_wr_byte_en(bias_wr_byte_en),

    .weight_loader_valid(weight_loader_valid),
    .weight_loader_ready(weight_loader_ready),
    .weight_loader_addr(weight_loader_addr),
    .weight_loader_data(weight_loader_data),
    .weight_loader_byte_en(weight_loader_byte_en),
    .weight_loader_done(weight_loader_done)
);

WeightPingPongController #(
    .ADDR_WIDTH(ADDR_WIDTH),
    .BANK_ADDR_BITS(10),
    .TOTAL_BANKS(12),
    .TILE_ADDR_BITS(6)
) u_weight_pingpong (
    .clk(clk),
    .rst_n(rst_n),

    .systolic_w_rd_en(w_bram_rd_en),
    .systolic_w_addr(w_bram_addr),
    .systolic_w_valid(w_bram_valid),
    .systolic_w_data(w_bram_data),

    .weight_rd_en(weight_mem_rd_en),
    .weight_rd_addr(weight_mem_rd_addr),
    .weight_rd_data(weight_mem_rd_data),

    .loader_valid(weight_loader_valid),
    .loader_ready(weight_loader_ready),
    .loader_addr(weight_loader_addr),
    .loader_data(weight_loader_data),
    .loader_byte_en(weight_loader_byte_en),
    .loader_done(weight_loader_done),

    .weight_wr_en(weight_mem_wr_en),
    .weight_wr_addr(weight_mem_wr_addr),
    .weight_wr_data(weight_mem_wr_data),
    .weight_wr_byte_en(weight_mem_wr_byte_en),

    .active_buffer(weight_active_buffer),
    .write_buffer(weight_write_buffer),
    .active_buffer_valid(weight_active_buffer_valid),
    .write_buffer_full(weight_write_buffer_full),
    .swap_done(weight_swap_done),
    .systolic_addr_error(weight_systolic_addr_error),
    .loader_addr_error(weight_loader_addr_error)
);

WeightMem #(
    .INIT_FILE("NONE"),
    .NUM_BANKS(12)
) u_weight_mem (
    .clk(clk),
    .rst_n(rst_n),
    .rd_en(weight_mem_rd_en),
    .rd_addr(weight_mem_rd_addr),
    .rd_data(weight_mem_rd_data),
    .wr_en(weight_mem_wr_en),
    .wr_addr(weight_mem_wr_addr),
    .wr_data(weight_mem_wr_data),
    .wr_byte_en(weight_mem_wr_byte_en)
);

OutputCaptureFSM #(
    .ADDR_WIDTH(OUTPUT_ADDR_WIDTH),
    .OUTPUT_WORDS(OUTPUT_WORDS)
) u_output_capture (
    .clk(clk),
    .rst_n(rst_n),
    .start(start),
    .opsum_valid(opsum_valid),
    .opsum(opsum),

    .capture_busy(output_capture_busy),
    .capture_done(output_capture_done),
    .capture_overflow(output_capture_overflow),
    .capture_count(output_capture_count),

    .mem_wr_en(output_mem_wr_en),
    .mem_wr_addr(output_mem_wr_addr),
    .mem_wr_data(output_mem_wr_data),
    .mem_wr_byte_en(output_mem_wr_byte_en)
);

OutputMem #(
    .ADDR_WIDTH(OUTPUT_ADDR_WIDTH)
) u_output_mem (
    .clk(clk),
    .rst_n(rst_n),

    .wr_en(output_mem_wr_en),
    .wr_addr(output_mem_wr_addr),
    .wr_data(output_mem_wr_data),
    .wr_byte_en(output_mem_wr_byte_en),

    .host_en(output_host_en),
    .host_addr(output_host_addr),
    .host_wr_data(output_host_wr_data),
    .host_we(output_host_we),
    .host_rd_data(output_host_rd_data)
);

endmodule
