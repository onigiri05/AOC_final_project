`timescale 1ns/1ps

// ============================================================
// Module: ViT_InputLoadFSM
//
// AXI/register-write side loader for the ViT system core.
// Python/PS writes CTRL/BASE/COUNT/DATA through AXI-Lite; this FSM turns
// those writes into internal BRAM write strobes. This mirrors the role of
// systolic_array/src/InputLoadFSM.v, but supports more ViT memory targets.
//
// Host register offsets:
//   0x0 CTRL  : bit0=start, bits[7:4]=target
//   0x4 BASE  : word base address or requested tile id
//   0x8 COUNT : number of 32-bit DATA writes
//   0xc DATA  : payload words
// ============================================================
module ViT_InputLoadFSM #(
    parameter int ADDR_WIDTH   = 20,
    parameter int TARGET_WIDTH = 4,
    parameter int TARGET_MAX   = 8
)(
    input  logic clk,
    input  logic rst_n,

    input  logic                    host_wr_valid,
    output logic                    host_wr_ready,
    input  logic [3:0]              host_wr_addr,
    input  logic [31:0]             host_wr_data,
    input  logic [3:0]              host_wr_strb,

    output logic                    host_busy,
    output logic                    host_done,
    output logic                    host_error,
    output logic [TARGET_WIDTH-1:0] host_active_target,
    output logic [ADDR_WIDTH:0]     host_word_count,

    output logic                    load_start_pulse,
    output logic [TARGET_WIDTH-1:0] load_start_target,
    output logic [ADDR_WIDTH-1:0]   load_start_base,

    output logic                    load_wr_en,
    output logic [TARGET_WIDTH-1:0] load_wr_target,
    output logic [ADDR_WIDTH-1:0]   load_wr_addr,
    output logic [31:0]             load_wr_data,
    output logic [3:0]              load_wr_strb,

    output logic                    load_done_pulse,
    output logic [TARGET_WIDTH-1:0] load_done_target,
    output logic [ADDR_WIDTH-1:0]   load_done_base,
    output logic [ADDR_WIDTH:0]     load_done_count
);

    localparam logic [3:0] HOST_CTRL_ADDR  = 4'h0;
    localparam logic [3:0] HOST_BASE_ADDR  = 4'h4;
    localparam logic [3:0] HOST_COUNT_ADDR = 4'h8;
    localparam logic [3:0] HOST_DATA_ADDR  = 4'hc;

    typedef enum logic [1:0] {
        ST_IDLE,
        ST_LOAD,
        ST_DONE
    } state_t;

    state_t state_q;

    logic [ADDR_WIDTH-1:0] base_addr_q;
    logic [ADDR_WIDTH:0]   count_q;
    logic [ADDR_WIDTH:0]   active_count_q;
    logic [ADDR_WIDTH:0]   word_count_q;

    logic ctrl_write;
    logic base_write;
    logic count_write;
    logic data_write;
    logic start_write;
    logic valid_target;
    logic cfg_write;
    logic accept_ctrl;
    logic accept_data;
    logic last_word;
    logic [TARGET_WIDTH-1:0] requested_target;

    assign ctrl_write = host_wr_valid && (host_wr_addr == HOST_CTRL_ADDR);
    assign base_write = host_wr_valid && (host_wr_addr == HOST_BASE_ADDR);
    assign count_write = host_wr_valid && (host_wr_addr == HOST_COUNT_ADDR);
    assign data_write = host_wr_valid && (host_wr_addr == HOST_DATA_ADDR);
    assign start_write = ctrl_write && host_wr_data[0];
    assign requested_target = host_wr_data[7:4];
    assign valid_target = (requested_target <= TARGET_MAX[TARGET_WIDTH-1:0]);
    assign cfg_write = (state_q == ST_IDLE) && (base_write || count_write);
    assign accept_ctrl = (state_q == ST_IDLE) && start_write && valid_target && (count_q != '0);
    assign accept_data = (state_q == ST_LOAD) && data_write;
    assign last_word = (word_count_q == (active_count_q - 1'b1));

    assign host_wr_ready = cfg_write || accept_ctrl || accept_data;
    assign host_busy = (state_q != ST_IDLE);
    assign host_word_count = word_count_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q <= ST_IDLE;
            base_addr_q <= '0;
            count_q <= '0;
            active_count_q <= '0;
            word_count_q <= '0;
            host_done <= 1'b0;
            host_error <= 1'b0;
            host_active_target <= '0;

            load_start_pulse <= 1'b0;
            load_start_target <= '0;
            load_start_base <= '0;
            load_wr_en <= 1'b0;
            load_wr_target <= '0;
            load_wr_addr <= '0;
            load_wr_data <= 32'd0;
            load_wr_strb <= 4'd0;
            load_done_pulse <= 1'b0;
            load_done_target <= '0;
            load_done_base <= '0;
            load_done_count <= '0;
        end
        else begin
            load_start_pulse <= 1'b0;
            load_wr_en <= 1'b0;
            load_done_pulse <= 1'b0;

            if (cfg_write) begin
                if (base_write) begin
                    base_addr_q <= host_wr_data[ADDR_WIDTH-1:0];
                end
                if (count_write) begin
                    count_q <= host_wr_data[ADDR_WIDTH:0];
                end
            end

            case (state_q)
                ST_IDLE: begin
                    if (start_write && (!valid_target || (count_q == '0))) begin
                        host_error <= 1'b1;
                    end
                    else if (accept_ctrl) begin
                        state_q <= ST_LOAD;
                        host_done <= 1'b0;
                        host_error <= 1'b0;
                        host_active_target <= requested_target;
                        active_count_q <= count_q;
                        word_count_q <= '0;

                        load_start_pulse <= 1'b1;
                        load_start_target <= requested_target;
                        load_start_base <= base_addr_q;
                    end
                end

                ST_LOAD: begin
                    if (accept_data) begin
                        load_wr_en <= 1'b1;
                        load_wr_target <= host_active_target;
                        load_wr_addr <= base_addr_q + word_count_q[ADDR_WIDTH-1:0];
                        load_wr_data <= host_wr_data;
                        load_wr_strb <= host_wr_strb;

                        if (last_word) begin
                            state_q <= ST_DONE;
                        end
                        else begin
                            word_count_q <= word_count_q + 1'b1;
                        end
                    end
                end

                ST_DONE: begin
                    host_done <= 1'b1;
                    load_done_pulse <= 1'b1;
                    load_done_target <= host_active_target;
                    load_done_base <= base_addr_q;
                    load_done_count <= active_count_q;
                    state_q <= ST_IDLE;
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
