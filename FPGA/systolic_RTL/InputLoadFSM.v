module InputLoadFSM #(
    parameter ADDR_WIDTH = 17,
    parameter TILE_ADDR_BITS = 6
)(
    input clk,
    input rst_n,

    // Host/AXI-like write side.
    // 0x0 CTRL  : bit0=start, bits[2:1]=target (0=activation, 1=bias, 2=weight)
    // 0x4 BASE  : word base address for activation/bias
    // 0x8 COUNT : number of 32-bit words to stream
    // 0xc DATA  : payload words
    input host_wr_valid,
    output host_wr_ready,
    input [3:0] host_wr_addr,
    input [31:0] host_wr_data,
    input [3:0] host_wr_strb,

    output host_busy,
    output reg host_done,
    output reg host_error,
    output reg [1:0] host_active_target,
    output [ADDR_WIDTH:0] host_word_count,

    output reg act_wr_en,
    output reg [ADDR_WIDTH-1:0] act_wr_addr,
    output reg [31:0] act_wr_data,
    output reg [3:0] act_wr_byte_en,

    output reg bias_wr_en,
    output reg [ADDR_WIDTH-1:0] bias_wr_addr,
    output reg [31:0] bias_wr_data,
    output reg [3:0] bias_wr_byte_en,

    output reg weight_loader_valid,
    input weight_loader_ready,
    output reg [ADDR_WIDTH-1:0] weight_loader_addr,
    output reg [31:0] weight_loader_data,
    output reg [3:0] weight_loader_byte_en,
    output reg weight_loader_done
);

localparam [3:0] HOST_CTRL_ADDR = 4'h0;
localparam [3:0] HOST_BASE_ADDR = 4'h4;
localparam [3:0] HOST_COUNT_ADDR = 4'h8;
localparam [3:0] HOST_DATA_ADDR = 4'hc;

localparam [1:0] TARGET_ACT = 2'd0;
localparam [1:0] TARGET_BIAS = 2'd1;
localparam [1:0] TARGET_WEIGHT = 2'd2;

localparam [1:0] ST_IDLE = 2'd0;
localparam [1:0] ST_LOAD = 2'd1;
localparam [1:0] ST_DONE = 2'd2;

reg [1:0] state;
reg [ADDR_WIDTH-1:0] base_addr_reg;
reg [ADDR_WIDTH:0] count_reg;
reg [ADDR_WIDTH:0] active_count;
reg [ADDR_WIDTH:0] word_count;

wire ctrl_write = host_wr_valid & (host_wr_addr == HOST_CTRL_ADDR);
wire base_write = host_wr_valid & (host_wr_addr == HOST_BASE_ADDR);
wire count_write = host_wr_valid & (host_wr_addr == HOST_COUNT_ADDR);
wire data_write = host_wr_valid & (host_wr_addr == HOST_DATA_ADDR);
wire start_write = ctrl_write & host_wr_data[0];
wire [1:0] requested_target = host_wr_data[2:1];
wire valid_target = (requested_target == TARGET_ACT) |
                    (requested_target == TARGET_BIAS) |
                    (requested_target == TARGET_WEIGHT);
wire cfg_write = (state == ST_IDLE) & (base_write | count_write);
wire accept_ctrl = (state == ST_IDLE) & start_write & valid_target & (count_reg != 0);
wire target_ready = (host_active_target != TARGET_WEIGHT) | weight_loader_ready;
wire accept_data = (state == ST_LOAD) & data_write & target_ready;
wire last_word = (word_count == (active_count - 1'b1));

assign host_wr_ready = cfg_write | accept_ctrl | accept_data;
assign host_busy = (state != ST_IDLE);
assign host_word_count = word_count;

always @(posedge clk) begin
    if(!rst_n) begin
        state <= ST_IDLE;
        base_addr_reg <= {ADDR_WIDTH{1'b0}};
        count_reg <= {(ADDR_WIDTH+1){1'b0}};
        active_count <= {(ADDR_WIDTH+1){1'b0}};
        word_count <= {(ADDR_WIDTH+1){1'b0}};
        host_done <= 1'b0;
        host_error <= 1'b0;
        host_active_target <= TARGET_ACT;

        act_wr_en <= 1'b0;
        act_wr_addr <= {ADDR_WIDTH{1'b0}};
        act_wr_data <= 32'd0;
        act_wr_byte_en <= 4'd0;

        bias_wr_en <= 1'b0;
        bias_wr_addr <= {ADDR_WIDTH{1'b0}};
        bias_wr_data <= 32'd0;
        bias_wr_byte_en <= 4'd0;

        weight_loader_valid <= 1'b0;
        weight_loader_addr <= {ADDR_WIDTH{1'b0}};
        weight_loader_data <= 32'd0;
        weight_loader_byte_en <= 4'd0;
        weight_loader_done <= 1'b0;
    end
    else begin
        act_wr_en <= 1'b0;
        bias_wr_en <= 1'b0;
        weight_loader_valid <= 1'b0;
        weight_loader_done <= 1'b0;

        if(cfg_write) begin
            if(base_write) begin
                base_addr_reg <= host_wr_data[ADDR_WIDTH-1:0];
            end
            if(count_write) begin
                count_reg <= host_wr_data[ADDR_WIDTH:0];
            end
        end

        case(state)
            ST_IDLE: begin
                if(start_write & (!valid_target | (count_reg == 0))) begin
                    host_error <= 1'b1;
                end
                else if(accept_ctrl) begin
                    state <= ST_LOAD;
                    host_done <= 1'b0;
                    host_error <= 1'b0;
                    host_active_target <= requested_target;
                    active_count <= count_reg;
                    word_count <= {(ADDR_WIDTH+1){1'b0}};
                end
            end

            ST_LOAD: begin
                if(accept_data) begin
                    case(host_active_target)
                        TARGET_ACT: begin
                            act_wr_en <= 1'b1;
                            act_wr_addr <= base_addr_reg + word_count[ADDR_WIDTH-1:0];
                            act_wr_data <= host_wr_data;
                            act_wr_byte_en <= host_wr_strb;
                        end

                        TARGET_BIAS: begin
                            bias_wr_en <= 1'b1;
                            bias_wr_addr <= base_addr_reg + word_count[ADDR_WIDTH-1:0];
                            bias_wr_data <= host_wr_data;
                            bias_wr_byte_en <= host_wr_strb;
                        end

                        TARGET_WEIGHT: begin
                            weight_loader_valid <= 1'b1;
                            weight_loader_addr <= {{(ADDR_WIDTH-TILE_ADDR_BITS){1'b0}},
                                                   word_count[TILE_ADDR_BITS-1:0]};
                            weight_loader_data <= host_wr_data;
                            weight_loader_byte_en <= host_wr_strb;
                        end

                        default: begin
                            host_error <= 1'b1;
                        end
                    endcase

                    if(last_word) begin
                        state <= ST_DONE;
                    end
                    else begin
                        word_count <= word_count + 1'b1;
                    end
                end
            end

            ST_DONE: begin
                if(host_active_target == TARGET_WEIGHT) begin
                    weight_loader_done <= 1'b1;
                end
                host_done <= 1'b1;
                state <= ST_IDLE;
            end

            default: begin
                state <= ST_IDLE;
            end
        endcase
    end
end

endmodule
