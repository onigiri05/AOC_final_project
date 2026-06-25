module SystolicAxiLiteWrapper #(
    parameter AXI_ADDR_WIDTH = 12,
    parameter AXI_DATA_WIDTH = 32
)(
    input s_axi_aclk,
    input s_axi_aresetn,

    input [AXI_ADDR_WIDTH-1:0] s_axi_awaddr,
    input s_axi_awvalid,
    output reg s_axi_awready,

    input [AXI_DATA_WIDTH-1:0] s_axi_wdata,
    input [(AXI_DATA_WIDTH/8)-1:0] s_axi_wstrb,
    input s_axi_wvalid,
    output reg s_axi_wready,

    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input s_axi_bready,

    input [AXI_ADDR_WIDTH-1:0] s_axi_araddr,
    input s_axi_arvalid,
    output reg s_axi_arready,

    output reg [AXI_DATA_WIDTH-1:0] s_axi_rdata,
    output reg [1:0] s_axi_rresp,
    output reg s_axi_rvalid,
    input s_axi_rready
);

localparam [AXI_ADDR_WIDTH-1:0] REG_CONTROL = 12'h000;
localparam [AXI_ADDR_WIDTH-1:0] REG_K_TILE_CNT = 12'h004;
localparam [AXI_ADDR_WIDTH-1:0] REG_ACT_BASE = 12'h008;
localparam [AXI_ADDR_WIDTH-1:0] REG_W_BASE = 12'h00c;
localparam [AXI_ADDR_WIDTH-1:0] REG_BIAS_BASE = 12'h010;
localparam [AXI_ADDR_WIDTH-1:0] REG_STATUS = 12'h014;
localparam [AXI_ADDR_WIDTH-1:0] REG_INPUT_TARGET = 12'h018;
localparam [AXI_ADDR_WIDTH-1:0] REG_INPUT_COUNT = 12'h01c;
localparam [AXI_ADDR_WIDTH-1:0] REG_OUTPUT_COUNT = 12'h020;
localparam [AXI_ADDR_WIDTH-1:0] REG_TOTAL_CYCLES = 12'h024;
localparam [AXI_ADDR_WIDTH-1:0] REG_COMPUTE_BUSY_CYCLES = 12'h028;
localparam [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_LOAD_CYCLES = 12'h02c;
localparam [AXI_ADDR_WIDTH-1:0] REG_OVERLAP_CYCLES = 12'h030;
localparam [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_STALL_CYCLES = 12'h034;
localparam [AXI_ADDR_WIDTH-1:0] REG_SYSTOLIC_LOAD_CYCLES = 12'h038;
localparam [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_WORD_ACCEPTS = 12'h03c;
localparam [AXI_ADDR_WIDTH-1:0] REG_ACT_BRAM_READS = 12'h040;
localparam [AXI_ADDR_WIDTH-1:0] REG_WEIGHT_BRAM_READS = 12'h044;
localparam [AXI_ADDR_WIDTH-1:0] REG_BIAS_BRAM_READS = 12'h048;
localparam [AXI_ADDR_WIDTH-1:0] REG_OUTPUT_WORD_WRITES = 12'h04c;
localparam [AXI_ADDR_WIDTH-1:0] REG_INPUT_BRAM_ACCESS_CYCLES = 12'h050;
localparam [AXI_ADDR_WIDTH-1:0] REG_CORE_BRAM_ACCESS_CYCLES = 12'h054;

localparam [AXI_ADDR_WIDTH-1:0] INPUT_BASE = 12'h100;
localparam [AXI_ADDR_WIDTH-1:0] INPUT_LIMIT = 12'h10f;
localparam [AXI_ADDR_WIDTH-1:0] OUTPUT_BASE = 12'h400;
localparam [AXI_ADDR_WIDTH-1:0] OUTPUT_LIMIT = 12'h7ff;

localparam [1:0] WR_IDLE = 2'd0;
localparam [1:0] WR_WAIT_INPUT = 2'd1;
localparam [1:0] WR_RESP = 2'd2;
localparam [1:0] RD_IDLE = 2'd0;
localparam [1:0] RD_WAIT_OUTPUT = 2'd1;
localparam [1:0] RD_CAPTURE_OUTPUT = 2'd2;
localparam [1:0] RD_RESP = 2'd3;

reg [1:0] wr_state;
reg [1:0] rd_state;
reg aw_pending;
reg w_pending;
reg [AXI_ADDR_WIDTH-1:0] awaddr_reg;
reg [AXI_DATA_WIDTH-1:0] wdata_reg;
reg [(AXI_DATA_WIDTH/8)-1:0] wstrb_reg;

reg [6:0] k_tile_cnt_reg;
reg [16:0] act_base_addr_reg;
reg [16:0] w_base_addr_reg;
reg [16:0] bias_base_addr_reg;
reg start_pulse;

reg input_wr_valid;
wire input_wr_ready;
reg [3:0] input_wr_addr;
reg [31:0] input_wr_data;
reg [3:0] input_wr_strb;
wire input_busy;
wire input_done;
wire input_error;
wire [1:0] input_active_target;
wire [17:0] input_word_count;

reg output_host_en;
reg [7:0] output_host_addr;
reg [31:0] output_host_wr_data;
reg [3:0] output_host_we;
wire [31:0] output_host_rd_data;

wire module_ready;
wire output_capture_busy;
wire output_capture_done;
wire output_capture_overflow;
wire [8:0] output_capture_count;
wire weight_active_buffer_valid;
wire weight_write_buffer_full;
wire weight_swap_done;
wire weight_systolic_addr_error;
wire weight_loader_addr_error;
wire profile_systolic_load_active;
wire profile_systolic_op_active;
wire profile_systolic_weight_stall;
wire profile_weight_load_busy;
wire profile_weight_word_accept;
wire profile_act_bram_read;
wire profile_weight_bram_read;
wire profile_bias_bram_read;
wire profile_output_word_write;

reg profile_running;
reg [31:0] total_cycle_count;
reg [31:0] compute_busy_cycles;
reg [31:0] weight_load_cycles;
reg [31:0] overlap_cycles;
reg [31:0] weight_stall_cycles;
reg [31:0] systolic_load_cycles;
reg [31:0] weight_word_accepts;
reg [31:0] act_bram_reads;
reg [31:0] weight_bram_reads;
reg [31:0] bias_bram_reads;
reg [31:0] output_word_writes;
reg [31:0] input_bram_access_cycles;
reg [31:0] core_bram_access_cycles;

wire aw_fire = s_axi_awvalid & s_axi_awready;
wire w_fire = s_axi_wvalid & s_axi_wready;
wire ar_fire = s_axi_arvalid & s_axi_arready;
wire write_buffer_full = aw_pending & w_pending;
wire write_is_input = (awaddr_reg >= INPUT_BASE) & (awaddr_reg <= INPUT_LIMIT);
wire read_is_output = (s_axi_araddr >= OUTPUT_BASE) & (s_axi_araddr <= OUTPUT_LIMIT);
wire [3:0] input_addr_from_axi = awaddr_reg[3:0];
wire [7:0] output_addr_from_axi = s_axi_araddr[9:2];
wire profile_input_bram_access = profile_act_bram_read | profile_weight_bram_read | profile_bias_bram_read;
wire profile_core_bram_access = profile_input_bram_access | profile_output_word_write;

wire [31:0] status_word = {
    20'd0,
    weight_loader_addr_error,
    weight_systolic_addr_error,
    weight_swap_done,
    weight_write_buffer_full,
    weight_active_buffer_valid,
    output_capture_overflow,
    output_capture_done,
    output_capture_busy,
    input_error,
    input_done,
    input_busy,
    module_ready
};

function [31:0] read_register;
    input [AXI_ADDR_WIDTH-1:0] addr;
begin
    case(addr)
        REG_CONTROL: read_register = 32'd0;
        REG_K_TILE_CNT: read_register = {25'd0, k_tile_cnt_reg};
        REG_ACT_BASE: read_register = {15'd0, act_base_addr_reg};
        REG_W_BASE: read_register = {15'd0, w_base_addr_reg};
        REG_BIAS_BASE: read_register = {15'd0, bias_base_addr_reg};
        REG_STATUS: read_register = status_word;
        REG_INPUT_TARGET: read_register = {30'd0, input_active_target};
        REG_INPUT_COUNT: read_register = {14'd0, input_word_count};
        REG_OUTPUT_COUNT: read_register = {23'd0, output_capture_count};
        REG_TOTAL_CYCLES: read_register = total_cycle_count;
        REG_COMPUTE_BUSY_CYCLES: read_register = compute_busy_cycles;
        REG_WEIGHT_LOAD_CYCLES: read_register = weight_load_cycles;
        REG_OVERLAP_CYCLES: read_register = overlap_cycles;
        REG_WEIGHT_STALL_CYCLES: read_register = weight_stall_cycles;
        REG_SYSTOLIC_LOAD_CYCLES: read_register = systolic_load_cycles;
        REG_WEIGHT_WORD_ACCEPTS: read_register = weight_word_accepts;
        REG_ACT_BRAM_READS: read_register = act_bram_reads;
        REG_WEIGHT_BRAM_READS: read_register = weight_bram_reads;
        REG_BIAS_BRAM_READS: read_register = bias_bram_reads;
        REG_OUTPUT_WORD_WRITES: read_register = output_word_writes;
        REG_INPUT_BRAM_ACCESS_CYCLES: read_register = input_bram_access_cycles;
        REG_CORE_BRAM_ACCESS_CYCLES: read_register = core_bram_access_cycles;
        default: read_register = 32'd0;
    endcase
end
endfunction

always @(posedge s_axi_aclk) begin
    if(!s_axi_aresetn) begin
        s_axi_awready <= 1'b0;
        s_axi_wready <= 1'b0;
        s_axi_bresp <= 2'b00;
        s_axi_bvalid <= 1'b0;
        s_axi_arready <= 1'b0;
        s_axi_rdata <= 32'd0;
        s_axi_rresp <= 2'b00;
        s_axi_rvalid <= 1'b0;

        wr_state <= WR_IDLE;
        rd_state <= RD_IDLE;
        aw_pending <= 1'b0;
        w_pending <= 1'b0;
        awaddr_reg <= {AXI_ADDR_WIDTH{1'b0}};
        wdata_reg <= 32'd0;
        wstrb_reg <= 4'd0;

        k_tile_cnt_reg <= 7'd0;
        act_base_addr_reg <= 17'd0;
        w_base_addr_reg <= 17'd0;
        bias_base_addr_reg <= 17'd0;
        start_pulse <= 1'b0;

        input_wr_valid <= 1'b0;
        input_wr_addr <= 4'd0;
        input_wr_data <= 32'd0;
        input_wr_strb <= 4'd0;

        output_host_en <= 1'b0;
        output_host_addr <= 8'd0;
        output_host_wr_data <= 32'd0;
        output_host_we <= 4'd0;

        profile_running <= 1'b0;
        total_cycle_count <= 32'd0;
        compute_busy_cycles <= 32'd0;
        weight_load_cycles <= 32'd0;
        overlap_cycles <= 32'd0;
        weight_stall_cycles <= 32'd0;
        systolic_load_cycles <= 32'd0;
        weight_word_accepts <= 32'd0;
        act_bram_reads <= 32'd0;
        weight_bram_reads <= 32'd0;
        bias_bram_reads <= 32'd0;
        output_word_writes <= 32'd0;
        input_bram_access_cycles <= 32'd0;
        core_bram_access_cycles <= 32'd0;
    end
    else begin
        start_pulse <= 1'b0;
        input_wr_valid <= 1'b0;
        output_host_en <= 1'b0;
        output_host_we <= 4'd0;
        s_axi_awready <= (wr_state == WR_IDLE) & (!aw_pending);
        s_axi_wready <= (wr_state == WR_IDLE) & (!w_pending);
        s_axi_arready <= (rd_state == RD_IDLE) & (!s_axi_rvalid);

        if(aw_fire) begin
            awaddr_reg <= s_axi_awaddr;
            aw_pending <= 1'b1;
        end

        if(w_fire) begin
            wdata_reg <= s_axi_wdata;
            wstrb_reg <= s_axi_wstrb;
            w_pending <= 1'b1;
        end

        if(s_axi_bvalid & s_axi_bready) begin
            s_axi_bvalid <= 1'b0;
        end

        if(s_axi_rvalid & s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end

        if(start_pulse) begin
            profile_running <= 1'b1;
            total_cycle_count <= 32'd0;
            compute_busy_cycles <= 32'd0;
            weight_load_cycles <= 32'd0;
            overlap_cycles <= 32'd0;
            weight_stall_cycles <= 32'd0;
            systolic_load_cycles <= 32'd0;
            weight_word_accepts <= 32'd0;
            act_bram_reads <= 32'd0;
            weight_bram_reads <= 32'd0;
            bias_bram_reads <= 32'd0;
            output_word_writes <= 32'd0;
            input_bram_access_cycles <= 32'd0;
            core_bram_access_cycles <= 32'd0;
        end
        else if(profile_running) begin
            if(profile_output_word_write) begin
                output_word_writes <= output_word_writes + 1'b1;
            end

            if(profile_core_bram_access) begin
                core_bram_access_cycles <= core_bram_access_cycles + 1'b1;
            end

            if(output_capture_done) begin
                profile_running <= 1'b0;
            end
            else begin
                total_cycle_count <= total_cycle_count + 1'b1;

                if(profile_systolic_op_active) begin
                    compute_busy_cycles <= compute_busy_cycles + 1'b1;
                end

                if(profile_systolic_load_active) begin
                    systolic_load_cycles <= systolic_load_cycles + 1'b1;
                end

                if(profile_weight_load_busy) begin
                    weight_load_cycles <= weight_load_cycles + 1'b1;
                end

                if(profile_systolic_op_active & profile_weight_load_busy) begin
                    overlap_cycles <= overlap_cycles + 1'b1;
                end

                if(profile_systolic_weight_stall) begin
                    weight_stall_cycles <= weight_stall_cycles + 1'b1;
                end

                if(profile_weight_word_accept) begin
                    weight_word_accepts <= weight_word_accepts + 1'b1;
                end

                if(profile_act_bram_read) begin
                    act_bram_reads <= act_bram_reads + 1'b1;
                end

                if(profile_weight_bram_read) begin
                    weight_bram_reads <= weight_bram_reads + 1'b1;
                end

                if(profile_bias_bram_read) begin
                    bias_bram_reads <= bias_bram_reads + 1'b1;
                end

                if(profile_input_bram_access) begin
                    input_bram_access_cycles <= input_bram_access_cycles + 1'b1;
                end
            end
        end

        case(wr_state)
            WR_IDLE: begin
                if(write_buffer_full) begin
                    if(write_is_input) begin
                        input_wr_addr <= input_addr_from_axi;
                        input_wr_data <= wdata_reg;
                        input_wr_strb <= wstrb_reg;
                        input_wr_valid <= 1'b1;
                        wr_state <= WR_WAIT_INPUT;
                    end
                    else begin
                        case(awaddr_reg)
                            REG_CONTROL: begin
                                if(wdata_reg[0]) begin
                                    start_pulse <= 1'b1;
                                end
                            end
                            REG_K_TILE_CNT: k_tile_cnt_reg <= wdata_reg[6:0];
                            REG_ACT_BASE: act_base_addr_reg <= wdata_reg[16:0];
                            REG_W_BASE: w_base_addr_reg <= wdata_reg[16:0];
                            REG_BIAS_BASE: bias_base_addr_reg <= wdata_reg[16:0];
                            default: begin
                            end
                        endcase

                        aw_pending <= 1'b0;
                        w_pending <= 1'b0;
                        s_axi_bresp <= 2'b00;
                        s_axi_bvalid <= 1'b1;
                        wr_state <= WR_RESP;
                    end
                end
            end

            WR_WAIT_INPUT: begin
                input_wr_addr <= input_addr_from_axi;
                input_wr_data <= wdata_reg;
                input_wr_strb <= wstrb_reg;

                if(input_wr_ready) begin
                    input_wr_valid <= 1'b0;
                    aw_pending <= 1'b0;
                    w_pending <= 1'b0;
                    s_axi_bresp <= 2'b00;
                    s_axi_bvalid <= 1'b1;
                    wr_state <= WR_RESP;
                end
                else begin
                    input_wr_valid <= 1'b1;
                end
            end

            WR_RESP: begin
                if(!s_axi_bvalid) begin
                    wr_state <= WR_IDLE;
                end
            end

            default: begin
                wr_state <= WR_IDLE;
            end
        endcase

        case(rd_state)
            RD_IDLE: begin
                if(ar_fire) begin
                    if(read_is_output) begin
                        output_host_en <= 1'b1;
                        output_host_addr <= output_addr_from_axi;
                        output_host_wr_data <= 32'd0;
                        output_host_we <= 4'd0;
                        rd_state <= RD_WAIT_OUTPUT;
                    end
                    else begin
                        s_axi_rdata <= read_register(s_axi_araddr);
                        s_axi_rresp <= 2'b00;
                        s_axi_rvalid <= 1'b1;
                        rd_state <= RD_RESP;
                    end
                end
            end

            RD_WAIT_OUTPUT: begin
                rd_state <= RD_CAPTURE_OUTPUT;
            end

            RD_CAPTURE_OUTPUT: begin
                s_axi_rdata <= output_host_rd_data;
                s_axi_rresp <= 2'b00;
                s_axi_rvalid <= 1'b1;
                rd_state <= RD_RESP;
            end

            RD_RESP: begin
                if(!s_axi_rvalid) begin
                    rd_state <= RD_IDLE;
                end
            end

            default: begin
                rd_state <= RD_IDLE;
            end
        endcase
    end
end

SystolicSystemCore u_core (
    .clk(s_axi_aclk),
    .rst_n(s_axi_aresetn),

    .start(start_pulse),
    .k_tile_cnt(k_tile_cnt_reg),
    .act_base_addr(act_base_addr_reg),
    .w_base_addr(w_base_addr_reg),
    .bias_base_addr(bias_base_addr_reg),
    .module_ready(module_ready),

    .input_wr_valid(input_wr_valid),
    .input_wr_ready(input_wr_ready),
    .input_wr_addr(input_wr_addr),
    .input_wr_data(input_wr_data),
    .input_wr_strb(input_wr_strb),
    .input_busy(input_busy),
    .input_done(input_done),
    .input_error(input_error),
    .input_active_target(input_active_target),
    .input_word_count(input_word_count),

    .output_host_en(output_host_en),
    .output_host_addr(output_host_addr),
    .output_host_wr_data(output_host_wr_data),
    .output_host_we(output_host_we),
    .output_host_rd_data(output_host_rd_data),

    .output_capture_busy(output_capture_busy),
    .output_capture_done(output_capture_done),
    .output_capture_overflow(output_capture_overflow),
    .output_capture_count(output_capture_count),
    .weight_active_buffer_valid(weight_active_buffer_valid),
    .weight_write_buffer_full(weight_write_buffer_full),
    .weight_swap_done(weight_swap_done),
    .weight_systolic_addr_error(weight_systolic_addr_error),
    .weight_loader_addr_error(weight_loader_addr_error),

    .profile_systolic_load_active(profile_systolic_load_active),
    .profile_systolic_op_active(profile_systolic_op_active),
    .profile_systolic_weight_stall(profile_systolic_weight_stall),
    .profile_weight_load_busy(profile_weight_load_busy),
    .profile_weight_word_accept(profile_weight_word_accept),
    .profile_act_bram_read(profile_act_bram_read),
    .profile_weight_bram_read(profile_weight_bram_read),
    .profile_bias_bram_read(profile_bias_bram_read),
    .profile_output_word_write(profile_output_word_write)
);

endmodule
