// Softmax unit for one attention score row.
//
// Resource-oriented interface:
// - Scores are loaded one by one through score_load_*.
// - The unit stores scaled scores internally, then runs max/exp/sum/normalize.
// - Normalized attention values are streamed out one by one through
//   attention_valid/index/data, so the caller can write A buffer directly.
//
// This avoids a 208-entry score row register in the caller and avoids a full
// attention_row output register/mux in this unit.
module Softmax_Unit #(
    parameter int COLS = 208,
    parameter int IDX_W = (COLS <= 2) ? 1 : $clog2(COLS),
    parameter EXP_LUT_HEX = "exp_lut_10bit_Q1_15_range12.hex"
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,

    input  logic score_load_valid,
    input  logic [IDX_W-1:0] score_load_index,
    input  logic signed [31:0] score_load_data,

    input  logic [5:0] q_shift,
    input  logic [5:0] k_shift,
    input  logic [15:0] valid_cols,

    output logic attention_valid,
    output logic [IDX_W-1:0] attention_index,
    output logic signed [7:0] attention_data,

    output logic done
);

    localparam int SCORE_SHIFT = 3;
    localparam int LUT_DEPTH = 1024;

    typedef enum logic [3:0] {
        ST_IDLE,
        ST_FIND_MAX,
        ST_EXP_PRODUCT,
        ST_EXP_SHIFT,
        ST_EXP_READ,
        ST_NORM_PREP,
        ST_NORM_DIV,
        ST_DONE
    } state_t;

    state_t state;
    logic [IDX_W-1:0] index;

    (* ram_style = "distributed" *) logic signed [31:0] scaled_score [0:COLS-1];
    (* ram_style = "distributed" *) logic [15:0] exp_value [0:COLS-1];

    logic signed [31:0] max_score;
    logic [31:0] exp_sum;
    logic has_valid;

    logic signed [31:0] lut_diff_comb;
    logic [32:0] lut_diff_magnitude_comb;
    logic [42:0] lut_product_q;
    logic [7:0]  lut_shift_q;
    logic [42:0] lut_index_unclamped_comb;
    logic [9:0]  lut_idx_q;

    logic [47:0] div_num_q;
    logic [31:0] div_den_q;
    logic [32:0] div_rem_q;
    logic [47:0] div_quot_q;
    logic [5:0]  div_bit_q;
    logic [32:0] div_rem_shift;
    logic [32:0] div_rem_next;
    logic [47:0] div_quot_shift;
    logic [47:0] div_quot_next;
    logic        div_take;

    logic index_valid;

    (* rom_style = "block" *)
    logic [15:0] exp_lut_rom [0:LUT_DEPTH-1];

    initial begin
        $readmemh(EXP_LUT_HEX, exp_lut_rom);
    end

    assign index_valid = ({8'd0, index} < valid_cols) && (valid_cols != 16'd0);

    assign lut_diff_comb = scaled_score[index] - max_score;

    always_comb begin
        if (lut_diff_comb >= 32'sd0) begin
            lut_diff_magnitude_comb = 33'd0;
        end
        else begin
            lut_diff_magnitude_comb = -$signed({lut_diff_comb[31], lut_diff_comb});
        end
    end

    assign lut_index_unclamped_comb =
        (lut_shift_q >= 8'd43) ? 43'd0 : (lut_product_q >> lut_shift_q);

    assign div_rem_shift  = {div_rem_q[31:0], div_num_q[47]};
    assign div_quot_shift = {div_quot_q[46:0], 1'b0};
    assign div_take       = div_rem_shift >= {1'b0, div_den_q};
    assign div_rem_next   = div_take ? (div_rem_shift - {1'b0, div_den_q}) :
                                       div_rem_shift;
    assign div_quot_next  = div_quot_shift | (div_take ? 48'd1 : 48'd0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            index <= '0;
            has_valid <= 1'b0;
            done <= 1'b0;
            attention_valid <= 1'b0;
            attention_index <= '0;
            attention_data <= 8'sd0;
        end
        else begin
            done <= 1'b0;
            attention_valid <= 1'b0;

            if (score_load_valid) begin
                scaled_score[score_load_index] <= score_load_data >>> SCORE_SHIFT;
            end

            case (state)
                ST_IDLE: begin
                    if (start) begin
                        index <= '0;
                        max_score <= -32'sd2147483647;
                        exp_sum <= 32'd0;
                        has_valid <= 1'b0;
                        lut_product_q <= 43'd0;
                        lut_shift_q <= 8'd0;
                        lut_idx_q <= 10'd0;
                        div_num_q <= 48'd0;
                        div_den_q <= 32'd0;
                        div_rem_q <= 33'd0;
                        div_quot_q <= 48'd0;
                        div_bit_q <= 6'd0;
                        state <= ST_FIND_MAX;
                    end
                end

                ST_FIND_MAX: begin
                    if (index_valid) begin
                        has_valid <= 1'b1;
                        if (!has_valid || (scaled_score[index] > max_score)) begin
                            max_score <= scaled_score[index];
                        end
                    end

                    if (index == (COLS - 1)) begin
                        index <= '0;
                        exp_sum <= 32'd0;
                        state <= ST_EXP_PRODUCT;
                    end
                    else begin
                        index <= index + 1'b1;
                    end
                end

                ST_EXP_PRODUCT: begin
                    if (index_valid && has_valid) begin
                        lut_product_q <= lut_diff_magnitude_comb * 10'd341;
                        lut_shift_q <= {2'b00, q_shift} + {2'b00, k_shift} + 8'd2;
                        state <= ST_EXP_SHIFT;
                    end
                    else begin
                        exp_value[index] <= 16'd0;
                        if (index == (COLS - 1)) begin
                            index <= '0;
                            state <= ST_NORM_PREP;
                        end
                        else begin
                            index <= index + 1'b1;
                        end
                    end
                end

                ST_EXP_SHIFT: begin
                    if (lut_index_unclamped_comb >= 43'd1023) begin
                        lut_idx_q <= 10'd1023;
                    end
                    else begin
                        lut_idx_q <= lut_index_unclamped_comb[9:0];
                    end
                    state <= ST_EXP_READ;
                end

                ST_EXP_READ: begin
                    exp_value[index] <= exp_lut_rom[lut_idx_q];
                    exp_sum <= exp_sum + exp_lut_rom[lut_idx_q];

                    if (index == (COLS - 1)) begin
                        index <= '0;
                        state <= ST_NORM_PREP;
                    end
                    else begin
                        index <= index + 1'b1;
                        state <= ST_EXP_PRODUCT;
                    end
                end

                ST_NORM_PREP: begin
                    if (index_valid && has_valid && (exp_value[index] != 16'd0) && (exp_sum != 32'd0)) begin
                        div_num_q  <= ({32'd0, exp_value[index]} << 7) + {16'd0, (exp_sum >> 1)};
                        div_den_q  <= exp_sum;
                        div_rem_q  <= 33'd0;
                        div_quot_q <= 48'd0;
                        div_bit_q  <= 6'd0;
                        state      <= ST_NORM_DIV;
                    end
                    else begin
                        attention_valid <= 1'b1;
                        attention_index <= index;
                        attention_data <= 8'sd0;

                        if (index == (COLS - 1)) begin
                            index <= '0;
                            state <= ST_DONE;
                        end
                        else begin
                            index <= index + 1'b1;
                        end
                    end
                end

                ST_NORM_DIV: begin
                    div_num_q <= {div_num_q[46:0], 1'b0};
                    div_rem_q <= div_rem_next;
                    div_quot_q <= div_quot_next;

                    if (div_bit_q == 6'd47) begin
                        attention_valid <= 1'b1;
                        attention_index <= index;
                        if (div_quot_next > 48'd127) begin
                            attention_data <= 8'sd127;
                        end
                        else begin
                            attention_data <= $signed(div_quot_next[7:0]);
                        end

                        if (index == (COLS - 1)) begin
                            index <= '0;
                            state <= ST_DONE;
                        end
                        else begin
                            index <= index + 1'b1;
                            state <= ST_NORM_PREP;
                        end
                    end
                    else begin
                        div_bit_q <= div_bit_q + 6'd1;
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                    state <= ST_IDLE;
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
