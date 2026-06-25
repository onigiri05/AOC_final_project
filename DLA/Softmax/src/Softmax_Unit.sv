// Softmax unit for MHSA
// Input score row: signed INT32, result of sum(Q_INT8 * K_INT8)
// q_scale = 2^(-q_shift)
// k_scale = 2^(-k_shift)
//
// streaming_attention_pow2lut12 dataflow:
// 1. scores_shifted = score_row >>> 3, where >>>3 implements 1/sqrt(64)
// 2. Find max_scores_shifted over valid tokens
// 3. int_diff = scores_shifted - max_scores_shifted, always <= 0
// 4. lut_idx = clamp(floor(-int_diff * combined_scale), 0, 1023)
//    combined_scale = 2^(-(q_shift+k_shift)) * (1023/12) = 341 / 2^(q_shift+k_shift+2).
// 5. exp LUT output: unsigned UQ1.15, 16 bits
// 6. Output attention row: signed INT8 Q0.7

module Softmax_Unit (
    input  logic clk,
    input  logic rst_n,
    input  logic start, // Start softmax signal

    input  logic signed [31:0] score_row [0:207], // Q_INT8 ?? K_INT8
    input  logic [5:0] q_shift, // q_scale = 2^(-q_shift), pow2lut12 expects 4
    input  logic [5:0] k_shift, // k_scale = 2^(-k_shift), pow2lut12 expects 4
    input  logic [207:0] mask, // 1: valid token, 0: padded token, output = 0

    output logic signed [7:0] attention_row [0:207], // Signed INT8 Q0.7

    output logic done
);

    localparam int SCORE_SHIFT = 3;

    // LUT: 1024 entries, address 0~1023, input real range [-12, 0].
    localparam int LUT_DEPTH = 1024;

    logic [2:0] state;
    logic [7:0] index; // Current row position, range 0~207

    logic signed [31:0] scaled_score [0:207];
    logic signed [31:0] max_score;
    logic [15:0] exp_value [0:207]; // exp(real score difference), unsigned UQ1.15
    logic [31:0] exp_sum;

    logic has_valid; // Indicates whether the row contains at least one valid token
    integer i;

    // ------------------------------------------------------------------------
    // 1024-entry exponential ROM
    //
    // LUT construction:
    //   lut[i] = round(exp(-i / 85.25) * 2^15)
    //
    // Expected file:
    //   exp_lut_10bit_Q1_15_range12.hex
    //
    // The ROM output is unsigned UQ1.15:
    //   address 0    -> exp(0)   -> 32768 = 16'h8000
    //   address 1023 -> exp(-12) -> approximately 0
    // ------------------------------------------------------------------------
    (* rom_style = "distributed" *)
    logic [15:0] exp_lut_rom [0:LUT_DEPTH-1];

    /*
    initial begin
        $readmemh(
            "exp_lut_10bit_Q1_15_range12.hex",
            exp_lut_rom
        );
    end
    */
    // ------------------------------------------------------------------------
    // Convert shifted-score difference to 10-bit LUT index.
    // int_diff = scores_shifted - max_scores_shifted <= 0
    // combined_scale = 2^(-(q_shift+k_shift)) * (1023/12) = 341 / 2^(q_shift+k_shift+2).
    // lut_idx = floor((-int_diff) * 341 / / 2^(q_shift+k_shift+2)) = ((-int_diff) * 341) >> q_shift+k_shift+2
    // Finally clamp to [0, 1023].
    // ------------------------------------------------------------------------
    function automatic logic [9:0] calculate_lut_index (
        input logic signed [31:0] int_diff,
        input logic        [5:0] q_shift_in,
        input logic        [5:0] k_shift_in
    );
        logic [32:0] diff_magnitude;
        logic [42:0] index_product;
        logic [42:0] index_unclamped;
        logic [7:0]  index_shift;

        begin
            if (int_diff >= 32'sd0) begin
                diff_magnitude = 33'd0;
            end

            else begin
                // Sign-extend before negation to safely handle negative values.
                diff_magnitude = -$signed({int_diff[31], int_diff});
            end

            // combined_scale
            index_product = diff_magnitude * 10'd341;
            index_shift   = {2'b00, q_shift_in}
                          + {2'b00, k_shift_in}
                          + 8'd2;

            // A shift greater than or equal to the product width produces zero.
            if (index_shift >= 8'd43)
                index_unclamped = 43'd0;
            else
                index_unclamped = index_product >> index_shift;

            if (index_unclamped >= 43'd1023)
                calculate_lut_index = 10'd1023;
            else
                calculate_lut_index = index_unclamped[9:0];
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
    // State 1: Apply division by 8 only: score_row >>> 3
    // State 2: Find the maximum unmasked shifted score
    // State 3: Calculate LUT index, read exp LUT, and accumulate exp_sum
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


                // score shift: only divide by sqrt(64)=8
                // scores_shifted = score_row >>> 3
                3'd1: begin
                    scaled_score[index] <= score_row[index] >>> SCORE_SHIFT;

                    if (index == 8'd207) begin
                        index <= 8'd0;
                        state <= 3'd2;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // Find the maximum shifted score where mask=1
                3'd2: begin
                    if (mask[index]) begin
                        has_valid <= 1'b1;

                        if (!has_valid || (scaled_score[index] > max_score)) // ??��?��?�慮??��?��?�到??��?? score 以�?��?�到?��大�?? score
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

                // Calculate:
                //   int_diff = scaled_score - max_score
                //   lut_idx  = clamp(floor(-int_diff * combined_scale), 0, 1023)
                //   combined_scale = 341 / 2^(q_shift+k_shift+2)
                // Then read UQ1.15 exp LUT and accumulate row denominator.
                // if mask=0: exp_value = 0
                3'd3: begin
                    if (mask[index] && has_valid) begin
                        exp_value[index] <= exp_lut_rom[
                            calculate_lut_index(
                                scaled_score[index] - max_score,
                                q_shift,
                                k_shift
                            )
                        ];

                        exp_sum <= exp_sum + exp_lut_rom[
                            calculate_lut_index(
                                scaled_score[index] - max_score,
                                q_shift,
                                k_shift
                            )
                        ];
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
                // exp_value remains unsigned UQ1.15, while output remains
                // signed INT8 Q0.7 as required by the original interface.
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
