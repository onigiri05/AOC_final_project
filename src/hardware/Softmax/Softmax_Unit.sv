// Softmax unit for MHSA
// Input score row: signed INT32, result of sum(Q_INT8 * K_INT8)
// q_scale = 2^(-q_shift)
// k_scale = 2^(-k_shift)
// Scaled score = score_row / 2^(q_shift + k_shift + 3)
// Output attention row: signed INT8 Q0.7

module Softmax_Unit (
    input  logic clk,
    input  logic rst_n,
    input  logic start, // Start softmax signal

    input  logic signed [31:0] score_row [0:207], // Q_INT8 × K_INT8
    input  logic [5:0] q_shift, // q_scale = 2^(-q_shift)
    input  logic [5:0] k_shift, // k_scale = 2^(-k_shift)
    input  logic [207:0] mask, // 1: valid token, 0: padded token, output = 0

    output logic signed [7:0] attention_row [0:207], // Signed INT8 Q0.7

    output logic done
);

    logic [2:0] state;
    logic [7:0] index; // Current row position, range 0~207
    logic [7:0] total_shift; // Total right shift = q_shift + k_shift + 3

    logic signed [31:0] scaled_score [0:207];
    logic signed [31:0] max_score;
    logic [15:0] exp_value [0:207]; // exp(scaled_score-max_score), UINT16 Q0.15
    logic [31:0] exp_sum;

    logic has_valid; // Indicates whether the row contains at least one valid token
    integer i;

    always_comb begin
        total_shift = {2'b00, q_shift} + {2'b00, k_shift} + 8'd3;
    end

    // Exponential approximation LUT
    // Input: signed integer scaled_score - max_score, always <= 0.
    // Output: UINT16 Q0.15
    function automatic logic [15:0] exp_lut (
        input logic signed [31:0] x
    );
        begin
            if (x <= -32'sd8)
                exp_lut = 16'd0;       // x <= -8 // 截在 -8 因為 exp^(-8) 已經很小了

            else if (x == -32'sd7)
                exp_lut = 16'd30;      // exp(-7) * 32768

            else if (x == -32'sd6)
                exp_lut = 16'd81;      // exp(-6) * 32768

            else if (x == -32'sd5)
                exp_lut = 16'd221;     // exp(-5) * 32768

            else if (x == -32'sd4)
                exp_lut = 16'd600;     // exp(-4) * 32768

            else if (x == -32'sd3)
                exp_lut = 16'd1631;    // exp(-3) * 32768

            else if (x == -32'sd2)
                exp_lut = 16'd4435;    // exp(-2) * 32768

            else if (x == -32'sd1)
                exp_lut = 16'd12055;   // exp(-1) * 32768

            else
                exp_lut = 16'd32767;   // exp(0)
        end
    endfunction

    // Convert (exp_value / exp_sum) to signed INT8 Q0.7
    // attention_int8 = round(exp_value / exp_sum * 128)
    function automatic logic signed [7:0] normalize_q07 (
        input logic [15:0] value,
        input logic [31:0] sum
    );

        logic [47:0] result;

        begin
            if ((value == 0) || (sum == 0)) begin
                normalize_q07 = 8'sd0;
            end

            else begin
                // Add sum/2 to implement round-to-nearest
                result = value * 48'd128;
                result = result + (sum >> 1);
                result = result / sum;

                if (result > 48'd127)
                    normalize_q07 = 8'sd127;
                else
                    normalize_q07 = $signed(result[7:0]);
            end
        end
    endfunction


    // ============================================================
    // State 0: Wait for start
    // State 1: Apply q_shift, k_shift, and division by 8
    // State 2: Find the maximum unmasked scaled score
    // State 3: Calculate exp(score-max) and accumulate exp_sum
    // State 4: Normalize the complete attention row to Q0.7
    // State 5: Assert done
    // ============================================================
    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin
            state      <= 3'd0;
            index      <= 8'd0;
            max_score  <= 32'sd0;
            exp_sum    <= 32'd0;
            has_valid  <= 1'b0;
            done       <= 1'b0;

            for (i = 0; i < 208; i = i + 1) begin
                scaled_score[i]  <= 32'sd0;
                exp_value[i]     <= 16'd0;
                attention_row[i] <= 8'sd0;
            end
        end

        else begin
            done <= 1'b0;

            case (state)

                3'd0: begin
                    if (start) begin
                        index      <= 8'd0;
                        max_score  <= -32'sd2147483647;
                        exp_sum    <= 32'd0;
                        has_valid  <= 1'b0;

                        for (i = 0; i < 208; i = i + 1) begin
                            scaled_score[i]  <= 32'sd0;
                            exp_value[i]     <= 16'd0;
                            attention_row[i] <= 8'sd0;
                        end

                        state <= 3'd1;
                    end
                end


                // score total shift
                3'd1: begin
                    if (total_shift >= 8'd32)
                        scaled_score[index] <= score_row[index][31] ? -32'sd1 : 32'sd0;
                    else
                        scaled_score[index] <= score_row[index] >>> total_shift;

                    if (index == 8'd207) begin
                        index <= 8'd0;
                        state <= 3'd2;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // Find the maximum scaled score where mask=1
                3'd2: begin
                    if (mask[index]) begin
                        has_valid <= 1'b1;

                        if (!has_valid || (scaled_score[index] > max_score)) // 同時考慮還沒遇到有效 score 以及遇到情況
                            max_score <= scaled_score[index];
                    end

                    if (index == 8'd207) begin
                        index   <= 8'd0;
                        exp_sum <= 32'd0;
                        state   <= 3'd3;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // Calculate exp(scaled_score-max_score and accumulate row denominator
                // if mask=0: exp_value = 0
                3'd3: begin
                    if (mask[index] && has_valid) begin
                        exp_value[index] <= exp_lut(
                            scaled_score[index] - max_score
                        );

                        exp_sum <= exp_sum + exp_lut(
                            scaled_score[index] - max_score
                        );
                    end

                    else begin
                        exp_value[index] <= 16'd0;
                    end

                    if (index == 8'd207) begin
                        index <= 8'd0;
                        state <= 3'd4;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // Normalize all 208 positions
                // if mask=1: attention = exp_value / exp_sum
                // if mask=0: attention = 0
                3'd4: begin
                    if (mask[index] && has_valid) begin
                        attention_row[index] <= normalize_q07(
                            exp_value[index],
                            exp_sum
                        );
                    end

                    else begin
                        attention_row[index] <= 8'sd0;
                    end

                    if (index == 8'd207) begin
                        index <= 8'd0;
                        state <= 3'd5;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // Complete attention_row[0:207] is ready
                3'd5: begin
                    done  <= 1'b1;
                    state <= 3'd0;
                end


                default: begin
                    state <= 3'd0;
                end

            endcase
        end
    end

endmodule
