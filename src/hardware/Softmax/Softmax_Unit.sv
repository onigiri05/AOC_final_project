// Softmax unit for MHSA
// Input score row: signed INT32, result of sum(Q_INT8 * K_INT8)
// q_scale = 2^(-q_shift)
// k_scale = 2^(-k_shift)
// Processing order: find row max in the raw INT32 score domain,
//                   subtract row max, then apply q/k scale and division by 8
// LUT input score difference: signed Q5.7, clamped to [-12, 0]
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

    localparam int SCORE_FRAC_BITS = 7;
    localparam logic signed [11:0] SCORE_MIN_Q57 = -12'sd1536; // -12.0 × 2^7

    logic [2:0] state;
    logic [7:0] index; // Current row position, range 0~207
    logic [7:0] total_shift; // Total real-score right shift = q_shift + k_shift + 3

    logic signed [11:0] scaled_score [0:207]; // (score-row_max) in signed Q5.7
    logic signed [31:0] max_score; // Raw INT32 row maximum before scaling
    logic signed [32:0] score_delta; // 33-bit difference avoids subtraction overflow
    logic signed [32:0] scaled_delta_wide;
    logic [15:0] exp_value [0:207]; // exp(scaled_score), UINT16 UQ1.15
    logic [31:0] exp_sum;

    logic has_valid; // Indicates whether the row contains at least one valid token
    integer i;

    always_comb begin
        total_shift = {2'b00, q_shift} + {2'b00, k_shift} + 8'd3;

        // First subtract the raw INT32 row maximum, then convert the
        // difference into signed Q5.7.  Multiplication by 2^7 preserves
        // seven fractional bits for the exponential LUT input.
        score_delta = $signed({score_row[index][31], score_row[index]})
                    - $signed({max_score[31], max_score});

        if (total_shift >= SCORE_FRAC_BITS)
            scaled_delta_wide = score_delta >>> (total_shift - SCORE_FRAC_BITS);
        else
            scaled_delta_wide = score_delta <<< (SCORE_FRAC_BITS - total_shift);
    end

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
    // State 1: Find the maximum unmasked raw INT32 score
    // State 2: Subtract row max, then apply q_shift, k_shift,
    //          division by 8, and convert to signed Q5.7
    // State 3: Calculate exp(score-row_max) and accumulate exp_sum
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
                scaled_score[i]  <= 12'sd0;
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
                            scaled_score[i]  <= 12'sd0;
                            exp_value[i]     <= 16'd0;
                            attention_row[i] <= 8'sd0;
                        end

                        state <= 3'd1;
                    end
                end


                // Find the maximum raw INT32 score where mask=1
                3'd1: begin
                    if (mask[index]) begin
                        has_valid <= 1'b1;

                        if (!has_valid || (score_row[index] > max_score)) // 同時考慮還沒遇到有效 score 以及遇到情況
                            max_score <= score_row[index];
                    end

                    if (index == 8'd207) begin
                        index <= 8'd0;
                        state <= 3'd2;
                    end

                    else begin
                        index <= index + 8'd1;
                    end
                end

                // First subtract raw row max, then scale the difference to Q5.7
                // Q5.7 code = (score_row-max_score) * 2^7 / 2^(q_shift+k_shift+3)
                3'd2: begin
                    if (mask[index] && has_valid) begin
                        if (scaled_delta_wide <= -33'sd1536) // -1536 = -12 * 128
                            scaled_score[index] <= SCORE_MIN_Q57;
                        else if (scaled_delta_wide >= 33'sd0)
                            scaled_score[index] <= 12'sd0; // 0 = 0 * 128
                        else
                            scaled_score[index] <= scaled_delta_wide[11:0];
                    end

                    else begin
                        scaled_score[index] <= 12'sd0;
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

                // Calculate exp(scaled_score) and accumulate row denominator
                // scaled_score already represents (score-row_max) in Q5.7
                // if mask=0: exp_value = 0
                3'd3: begin
                    if (mask[index] && has_valid) begin
                        exp_value[index] <= exp_lut_q57(
                            scaled_score[index]
                        );

                        exp_sum <= exp_sum + exp_lut_q57(
                            scaled_score[index]
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


    // Exponential approximation LUT
    // Input : signed Q5.7 score difference, always in [-12, 0]
    // Address step = 2^-7 = 1/128
    // Output: UINT16 UQ1.15, round(exp(x) * 32768)
    //         exp(0) is represented exactly as 32768.
    // Values below -12 are clamped to -12 before this function is called.
    function automatic logic [15:0] exp_lut_q57 (
        input logic signed [11:0] x_q57
    );
        logic [11:0] lut_addr;
        begin
            if (x_q57 >= 12'sd0)
                lut_addr = 12'd0;
            else if (x_q57 <= -12'sd1536)
                lut_addr = 12'd1536;
            else
                lut_addr = $unsigned(-x_q57);

            case (lut_addr)
                12'd0: exp_lut_q57 = 16'd32768; // 2^(15) = 32768
                12'd1: exp_lut_q57 = 16'd32513;
                12'd2: exp_lut_q57 = 16'd32260;
                12'd3: exp_lut_q57 = 16'd32009;
                12'd4: exp_lut_q57 = 16'd31760;
                12'd5: exp_lut_q57 = 16'd31513;
                12'd6: exp_lut_q57 = 16'd31267;
                12'd7: exp_lut_q57 = 16'd31024;
                12'd8: exp_lut_q57 = 16'd30783;
                12'd9: exp_lut_q57 = 16'd30543;
                12'd10: exp_lut_q57 = 16'd30305;
                12'd11: exp_lut_q57 = 16'd30070;
                12'd12: exp_lut_q57 = 16'd29836;
                12'd13: exp_lut_q57 = 16'd29603;
                12'd14: exp_lut_q57 = 16'd29373;
                12'd15: exp_lut_q57 = 16'd29144;
                12'd16: exp_lut_q57 = 16'd28918;
                12'd17: exp_lut_q57 = 16'd28693;
                12'd18: exp_lut_q57 = 16'd28469;
                12'd19: exp_lut_q57 = 16'd28248;
                12'd20: exp_lut_q57 = 16'd28028;
                12'd21: exp_lut_q57 = 16'd27810;
                12'd22: exp_lut_q57 = 16'd27593;
                12'd23: exp_lut_q57 = 16'd27379;
                12'd24: exp_lut_q57 = 16'd27166;
                12'd25: exp_lut_q57 = 16'd26954;
                12'd26: exp_lut_q57 = 16'd26744;
                12'd27: exp_lut_q57 = 16'd26536;
                12'd28: exp_lut_q57 = 16'd26330;
                12'd29: exp_lut_q57 = 16'd26125;
                12'd30: exp_lut_q57 = 16'd25922;
                12'd31: exp_lut_q57 = 16'd25720;
                12'd32: exp_lut_q57 = 16'd25520;
                12'd33: exp_lut_q57 = 16'd25321;
                12'd34: exp_lut_q57 = 16'd25124;
                12'd35: exp_lut_q57 = 16'd24929;
                12'd36: exp_lut_q57 = 16'd24735;
                12'd37: exp_lut_q57 = 16'd24542;
                12'd38: exp_lut_q57 = 16'd24351;
                12'd39: exp_lut_q57 = 16'd24162;
                12'd40: exp_lut_q57 = 16'd23974;
                12'd41: exp_lut_q57 = 16'd23787;
                12'd42: exp_lut_q57 = 16'd23602;
                12'd43: exp_lut_q57 = 16'd23418;
                12'd44: exp_lut_q57 = 16'd23236;
                12'd45: exp_lut_q57 = 16'd23055;
                12'd46: exp_lut_q57 = 16'd22876;
                12'd47: exp_lut_q57 = 16'd22698;
                12'd48: exp_lut_q57 = 16'd22521;
                12'd49: exp_lut_q57 = 16'd22346;
                12'd50: exp_lut_q57 = 16'd22172;
                12'd51: exp_lut_q57 = 16'd21999;
                12'd52: exp_lut_q57 = 16'd21828;
                12'd53: exp_lut_q57 = 16'd21658;
                12'd54: exp_lut_q57 = 16'd21490;
                12'd55: exp_lut_q57 = 16'd21323;
                12'd56: exp_lut_q57 = 16'd21157;
                12'd57: exp_lut_q57 = 16'd20992;
                12'd58: exp_lut_q57 = 16'd20829;
                12'd59: exp_lut_q57 = 16'd20667;
                12'd60: exp_lut_q57 = 16'd20506;
                12'd61: exp_lut_q57 = 16'd20346;
                12'd62: exp_lut_q57 = 16'd20188;
                12'd63: exp_lut_q57 = 16'd20031;
                12'd64: exp_lut_q57 = 16'd19875;
                12'd65: exp_lut_q57 = 16'd19720;
                12'd66: exp_lut_q57 = 16'd19567;
                12'd67: exp_lut_q57 = 16'd19414;
                12'd68: exp_lut_q57 = 16'd19263;
                12'd69: exp_lut_q57 = 16'd19113;
                12'd70: exp_lut_q57 = 16'd18965;
                12'd71: exp_lut_q57 = 16'd18817;
                12'd72: exp_lut_q57 = 16'd18671;
                12'd73: exp_lut_q57 = 16'd18525;
                12'd74: exp_lut_q57 = 16'd18381;
                12'd75: exp_lut_q57 = 16'd18238;
                12'd76: exp_lut_q57 = 16'd18096;
                12'd77: exp_lut_q57 = 16'd17955;
                12'd78: exp_lut_q57 = 16'd17816;
                12'd79: exp_lut_q57 = 16'd17677;
                12'd80: exp_lut_q57 = 16'd17539;
                12'd81: exp_lut_q57 = 16'd17403;
                12'd82: exp_lut_q57 = 16'd17268;
                12'd83: exp_lut_q57 = 16'd17133;
                12'd84: exp_lut_q57 = 16'd17000;
                12'd85: exp_lut_q57 = 16'd16868;
                12'd86: exp_lut_q57 = 16'd16736;
                12'd87: exp_lut_q57 = 16'd16606;
                12'd88: exp_lut_q57 = 16'd16477;
                12'd89: exp_lut_q57 = 16'd16349;
                12'd90: exp_lut_q57 = 16'd16221;
                12'd91: exp_lut_q57 = 16'd16095;
                12'd92: exp_lut_q57 = 16'd15970;
                12'd93: exp_lut_q57 = 16'd15846;
                12'd94: exp_lut_q57 = 16'd15722;
                12'd95: exp_lut_q57 = 16'd15600;
                12'd96: exp_lut_q57 = 16'd15479;
                12'd97: exp_lut_q57 = 16'd15358;
                12'd98: exp_lut_q57 = 16'd15239;
                12'd99: exp_lut_q57 = 16'd15120;
                12'd100: exp_lut_q57 = 16'd15002;
                12'd101: exp_lut_q57 = 16'd14886;
                12'd102: exp_lut_q57 = 16'd14770;
                12'd103: exp_lut_q57 = 16'd14655;
                12'd104: exp_lut_q57 = 16'd14541;
                12'd105: exp_lut_q57 = 16'd14428;
                12'd106: exp_lut_q57 = 16'd14315;
                12'd107: exp_lut_q57 = 16'd14204;
                12'd108: exp_lut_q57 = 16'd14093;
                12'd109: exp_lut_q57 = 16'd13984;
                12'd110: exp_lut_q57 = 16'd13875;
                12'd111: exp_lut_q57 = 16'd13767;
                12'd112: exp_lut_q57 = 16'd13660;
                12'd113: exp_lut_q57 = 16'd13553;
                12'd114: exp_lut_q57 = 16'd13448;
                12'd115: exp_lut_q57 = 16'd13343;
                12'd116: exp_lut_q57 = 16'd13239;
                12'd117: exp_lut_q57 = 16'd13136;
                12'd118: exp_lut_q57 = 16'd13034;
                12'd119: exp_lut_q57 = 16'd12933;
                12'd120: exp_lut_q57 = 16'd12832;
                12'd121: exp_lut_q57 = 16'd12732;
                12'd122: exp_lut_q57 = 16'd12633;
                12'd123: exp_lut_q57 = 16'd12535;
                12'd124: exp_lut_q57 = 16'd12437;
                12'd125: exp_lut_q57 = 16'd12341;
                12'd126: exp_lut_q57 = 16'd12245;
                12'd127: exp_lut_q57 = 16'd12149;
                12'd128: exp_lut_q57 = 16'd12055;
                12'd129: exp_lut_q57 = 16'd11961;
                12'd130: exp_lut_q57 = 16'd11868;
                12'd131: exp_lut_q57 = 16'd11775;
                12'd132: exp_lut_q57 = 16'd11684;
                12'd133: exp_lut_q57 = 16'd11593;
                12'd134: exp_lut_q57 = 16'd11503;
                12'd135: exp_lut_q57 = 16'd11413;
                12'd136: exp_lut_q57 = 16'd11324;
                12'd137: exp_lut_q57 = 16'd11236;
                12'd138: exp_lut_q57 = 16'd11149;
                12'd139: exp_lut_q57 = 16'd11062;
                12'd140: exp_lut_q57 = 16'd10976;
                12'd141: exp_lut_q57 = 16'd10890;
                12'd142: exp_lut_q57 = 16'd10806;
                12'd143: exp_lut_q57 = 16'd10722;
                12'd144: exp_lut_q57 = 16'd10638;
                12'd145: exp_lut_q57 = 16'd10555;
                12'd146: exp_lut_q57 = 16'd10473;
                12'd147: exp_lut_q57 = 16'd10392;
                12'd148: exp_lut_q57 = 16'd10311;
                12'd149: exp_lut_q57 = 16'd10231;
                12'd150: exp_lut_q57 = 16'd10151;
                12'd151: exp_lut_q57 = 16'd10072;
                12'd152: exp_lut_q57 = 16'd9994;
                12'd153: exp_lut_q57 = 16'd9916;
                12'd154: exp_lut_q57 = 16'd9839;
                12'd155: exp_lut_q57 = 16'd9762;
                12'd156: exp_lut_q57 = 16'd9686;
                12'd157: exp_lut_q57 = 16'd9611;
                12'd158: exp_lut_q57 = 16'd9536;
                12'd159: exp_lut_q57 = 16'd9462;
                12'd160: exp_lut_q57 = 16'd9388;
                12'd161: exp_lut_q57 = 16'd9315;
                12'd162: exp_lut_q57 = 16'd9243;
                12'd163: exp_lut_q57 = 16'd9171;
                12'd164: exp_lut_q57 = 16'd9099;
                12'd165: exp_lut_q57 = 16'd9029;
                12'd166: exp_lut_q57 = 16'd8958;
                12'd167: exp_lut_q57 = 16'd8889;
                12'd168: exp_lut_q57 = 16'd8819;
                12'd169: exp_lut_q57 = 16'd8751;
                12'd170: exp_lut_q57 = 16'd8683;
                12'd171: exp_lut_q57 = 16'd8615;
                12'd172: exp_lut_q57 = 16'd8548;
                12'd173: exp_lut_q57 = 16'd8482;
                12'd174: exp_lut_q57 = 16'd8416;
                12'd175: exp_lut_q57 = 16'd8350;
                12'd176: exp_lut_q57 = 16'd8285;
                12'd177: exp_lut_q57 = 16'd8221;
                12'd178: exp_lut_q57 = 16'd8157;
                12'd179: exp_lut_q57 = 16'd8093;
                12'd180: exp_lut_q57 = 16'd8030;
                12'd181: exp_lut_q57 = 16'd7968;
                12'd182: exp_lut_q57 = 16'd7906;
                12'd183: exp_lut_q57 = 16'd7844;
                12'd184: exp_lut_q57 = 16'd7783;
                12'd185: exp_lut_q57 = 16'd7723;
                12'd186: exp_lut_q57 = 16'd7662;
                12'd187: exp_lut_q57 = 16'd7603;
                12'd188: exp_lut_q57 = 16'd7544;
                12'd189: exp_lut_q57 = 16'd7485;
                12'd190: exp_lut_q57 = 16'd7427;
                12'd191: exp_lut_q57 = 16'd7369;
                12'd192: exp_lut_q57 = 16'd7312;
                12'd193: exp_lut_q57 = 16'd7255;
                12'd194: exp_lut_q57 = 16'd7198;
                12'd195: exp_lut_q57 = 16'd7142;
                12'd196: exp_lut_q57 = 16'd7087;
                12'd197: exp_lut_q57 = 16'd7031;
                12'd198: exp_lut_q57 = 16'd6977;
                12'd199: exp_lut_q57 = 16'd6922;
                12'd200: exp_lut_q57 = 16'd6869;
                12'd201: exp_lut_q57 = 16'd6815;
                12'd202: exp_lut_q57 = 16'd6762;
                12'd203: exp_lut_q57 = 16'd6709;
                12'd204: exp_lut_q57 = 16'd6657;
                12'd205: exp_lut_q57 = 16'd6605;
                12'd206: exp_lut_q57 = 16'd6554;
                12'd207: exp_lut_q57 = 16'd6503;
                12'd208: exp_lut_q57 = 16'd6452;
                12'd209: exp_lut_q57 = 16'd6402;
                12'd210: exp_lut_q57 = 16'd6352;
                12'd211: exp_lut_q57 = 16'd6303;
                12'd212: exp_lut_q57 = 16'd6254;
                12'd213: exp_lut_q57 = 16'd6205;
                12'd214: exp_lut_q57 = 16'd6157;
                12'd215: exp_lut_q57 = 16'd6109;
                12'd216: exp_lut_q57 = 16'd6061;
                12'd217: exp_lut_q57 = 16'd6014;
                12'd218: exp_lut_q57 = 16'd5967;
                12'd219: exp_lut_q57 = 16'd5921;
                12'd220: exp_lut_q57 = 16'd5875;
                12'd221: exp_lut_q57 = 16'd5829;
                12'd222: exp_lut_q57 = 16'd5784;
                12'd223: exp_lut_q57 = 16'd5739;
                12'd224: exp_lut_q57 = 16'd5694;
                12'd225: exp_lut_q57 = 16'd5650;
                12'd226: exp_lut_q57 = 16'd5606;
                12'd227: exp_lut_q57 = 16'd5562;
                12'd228: exp_lut_q57 = 16'd5519;
                12'd229: exp_lut_q57 = 16'd5476;
                12'd230: exp_lut_q57 = 16'd5433;
                12'd231: exp_lut_q57 = 16'd5391;
                12'd232: exp_lut_q57 = 16'd5349;
                12'd233: exp_lut_q57 = 16'd5308;
                12'd234: exp_lut_q57 = 16'd5266;
                12'd235: exp_lut_q57 = 16'd5225;
                12'd236: exp_lut_q57 = 16'd5185;
                12'd237: exp_lut_q57 = 16'd5144;
                12'd238: exp_lut_q57 = 16'd5104;
                12'd239: exp_lut_q57 = 16'd5065;
                12'd240: exp_lut_q57 = 16'd5025;
                12'd241: exp_lut_q57 = 16'd4986;
                12'd242: exp_lut_q57 = 16'd4947;
                12'd243: exp_lut_q57 = 16'd4909;
                12'd244: exp_lut_q57 = 16'd4871;
                12'd245: exp_lut_q57 = 16'd4833;
                12'd246: exp_lut_q57 = 16'd4795;
                12'd247: exp_lut_q57 = 16'd4758;
                12'd248: exp_lut_q57 = 16'd4721;
                12'd249: exp_lut_q57 = 16'd4684;
                12'd250: exp_lut_q57 = 16'd4647;
                12'd251: exp_lut_q57 = 16'd4611;
                12'd252: exp_lut_q57 = 16'd4575;
                12'd253: exp_lut_q57 = 16'd4540;
                12'd254: exp_lut_q57 = 16'd4505;
                12'd255: exp_lut_q57 = 16'd4469;
                12'd256: exp_lut_q57 = 16'd4435;
                12'd257: exp_lut_q57 = 16'd4400;
                12'd258: exp_lut_q57 = 16'd4366;
                12'd259: exp_lut_q57 = 16'd4332;
                12'd260: exp_lut_q57 = 16'd4298;
                12'd261: exp_lut_q57 = 16'd4265;
                12'd262: exp_lut_q57 = 16'd4232;
                12'd263: exp_lut_q57 = 16'd4199;
                12'd264: exp_lut_q57 = 16'd4166;
                12'd265: exp_lut_q57 = 16'd4134;
                12'd266: exp_lut_q57 = 16'd4101;
                12'd267: exp_lut_q57 = 16'd4069;
                12'd268: exp_lut_q57 = 16'd4038;
                12'd269: exp_lut_q57 = 16'd4006;
                12'd270: exp_lut_q57 = 16'd3975;
                12'd271: exp_lut_q57 = 16'd3944;
                12'd272: exp_lut_q57 = 16'd3914;
                12'd273: exp_lut_q57 = 16'd3883;
                12'd274: exp_lut_q57 = 16'd3853;
                12'd275: exp_lut_q57 = 16'd3823;
                12'd276: exp_lut_q57 = 16'd3793;
                12'd277: exp_lut_q57 = 16'd3764;
                12'd278: exp_lut_q57 = 16'd3734;
                12'd279: exp_lut_q57 = 16'd3705;
                12'd280: exp_lut_q57 = 16'd3676;
                12'd281: exp_lut_q57 = 16'd3648;
                12'd282: exp_lut_q57 = 16'd3619;
                12'd283: exp_lut_q57 = 16'd3591;
                12'd284: exp_lut_q57 = 16'd3563;
                12'd285: exp_lut_q57 = 16'd3536;
                12'd286: exp_lut_q57 = 16'd3508;
                12'd287: exp_lut_q57 = 16'd3481;
                12'd288: exp_lut_q57 = 16'd3454;
                12'd289: exp_lut_q57 = 16'd3427;
                12'd290: exp_lut_q57 = 16'd3400;
                12'd291: exp_lut_q57 = 16'd3374;
                12'd292: exp_lut_q57 = 16'd3347;
                12'd293: exp_lut_q57 = 16'd3321;
                12'd294: exp_lut_q57 = 16'd3296;
                12'd295: exp_lut_q57 = 16'd3270;
                12'd296: exp_lut_q57 = 16'd3244;
                12'd297: exp_lut_q57 = 16'd3219;
                12'd298: exp_lut_q57 = 16'd3194;
                12'd299: exp_lut_q57 = 16'd3169;
                12'd300: exp_lut_q57 = 16'd3145;
                12'd301: exp_lut_q57 = 16'd3120;
                12'd302: exp_lut_q57 = 16'd3096;
                12'd303: exp_lut_q57 = 16'd3072;
                12'd304: exp_lut_q57 = 16'd3048;
                12'd305: exp_lut_q57 = 16'd3024;
                12'd306: exp_lut_q57 = 16'd3001;
                12'd307: exp_lut_q57 = 16'd2977;
                12'd308: exp_lut_q57 = 16'd2954;
                12'd309: exp_lut_q57 = 16'd2931;
                12'd310: exp_lut_q57 = 16'd2908;
                12'd311: exp_lut_q57 = 16'd2886;
                12'd312: exp_lut_q57 = 16'd2863;
                12'd313: exp_lut_q57 = 16'd2841;
                12'd314: exp_lut_q57 = 16'd2819;
                12'd315: exp_lut_q57 = 16'd2797;
                12'd316: exp_lut_q57 = 16'd2775;
                12'd317: exp_lut_q57 = 16'd2754;
                12'd318: exp_lut_q57 = 16'd2732;
                12'd319: exp_lut_q57 = 16'd2711;
                12'd320: exp_lut_q57 = 16'd2690;
                12'd321: exp_lut_q57 = 16'd2669;
                12'd322: exp_lut_q57 = 16'd2648;
                12'd323: exp_lut_q57 = 16'd2627;
                12'd324: exp_lut_q57 = 16'd2607;
                12'd325: exp_lut_q57 = 16'd2587;
                12'd326: exp_lut_q57 = 16'd2567;
                12'd327: exp_lut_q57 = 16'd2547;
                12'd328: exp_lut_q57 = 16'd2527;
                12'd329: exp_lut_q57 = 16'd2507;
                12'd330: exp_lut_q57 = 16'd2488;
                12'd331: exp_lut_q57 = 16'd2468;
                12'd332: exp_lut_q57 = 16'd2449;
                12'd333: exp_lut_q57 = 16'd2430;
                12'd334: exp_lut_q57 = 16'd2411;
                12'd335: exp_lut_q57 = 16'd2392;
                12'd336: exp_lut_q57 = 16'd2374;
                12'd337: exp_lut_q57 = 16'd2355;
                12'd338: exp_lut_q57 = 16'd2337;
                12'd339: exp_lut_q57 = 16'd2319;
                12'd340: exp_lut_q57 = 16'd2301;
                12'd341: exp_lut_q57 = 16'd2283;
                12'd342: exp_lut_q57 = 16'd2265;
                12'd343: exp_lut_q57 = 16'd2247;
                12'd344: exp_lut_q57 = 16'd2230;
                12'd345: exp_lut_q57 = 16'd2213;
                12'd346: exp_lut_q57 = 16'd2195;
                12'd347: exp_lut_q57 = 16'd2178;
                12'd348: exp_lut_q57 = 16'd2161;
                12'd349: exp_lut_q57 = 16'd2144;
                12'd350: exp_lut_q57 = 16'd2128;
                12'd351: exp_lut_q57 = 16'd2111;
                12'd352: exp_lut_q57 = 16'd2095;
                12'd353: exp_lut_q57 = 16'd2078;
                12'd354: exp_lut_q57 = 16'd2062;
                12'd355: exp_lut_q57 = 16'd2046;
                12'd356: exp_lut_q57 = 16'd2030;
                12'd357: exp_lut_q57 = 16'd2015;
                12'd358: exp_lut_q57 = 16'd1999;
                12'd359: exp_lut_q57 = 16'd1983;
                12'd360: exp_lut_q57 = 16'd1968;
                12'd361: exp_lut_q57 = 16'd1953;
                12'd362: exp_lut_q57 = 16'd1937;
                12'd363: exp_lut_q57 = 16'd1922;
                12'd364: exp_lut_q57 = 16'd1907;
                12'd365: exp_lut_q57 = 16'd1892;
                12'd366: exp_lut_q57 = 16'd1878;
                12'd367: exp_lut_q57 = 16'd1863;
                12'd368: exp_lut_q57 = 16'd1849;
                12'd369: exp_lut_q57 = 16'd1834;
                12'd370: exp_lut_q57 = 16'd1820;
                12'd371: exp_lut_q57 = 16'd1806;
                12'd372: exp_lut_q57 = 16'd1792;
                12'd373: exp_lut_q57 = 16'd1778;
                12'd374: exp_lut_q57 = 16'd1764;
                12'd375: exp_lut_q57 = 16'd1750;
                12'd376: exp_lut_q57 = 16'd1737;
                12'd377: exp_lut_q57 = 16'd1723;
                12'd378: exp_lut_q57 = 16'd1710;
                12'd379: exp_lut_q57 = 16'd1696;
                12'd380: exp_lut_q57 = 16'd1683;
                12'd381: exp_lut_q57 = 16'd1670;
                12'd382: exp_lut_q57 = 16'd1657;
                12'd383: exp_lut_q57 = 16'd1644;
                12'd384: exp_lut_q57 = 16'd1631;
                12'd385: exp_lut_q57 = 16'd1619;
                12'd386: exp_lut_q57 = 16'd1606;
                12'd387: exp_lut_q57 = 16'd1594;
                12'd388: exp_lut_q57 = 16'd1581;
                12'd389: exp_lut_q57 = 16'd1569;
                12'd390: exp_lut_q57 = 16'd1557;
                12'd391: exp_lut_q57 = 16'd1545;
                12'd392: exp_lut_q57 = 16'd1533;
                12'd393: exp_lut_q57 = 16'd1521;
                12'd394: exp_lut_q57 = 16'd1509;
                12'd395: exp_lut_q57 = 16'd1497;
                12'd396: exp_lut_q57 = 16'd1485;
                12'd397: exp_lut_q57 = 16'd1474;
                12'd398: exp_lut_q57 = 16'd1462;
                12'd399: exp_lut_q57 = 16'd1451;
                12'd400: exp_lut_q57 = 16'd1440;
                12'd401: exp_lut_q57 = 16'd1429;
                12'd402: exp_lut_q57 = 16'd1417;
                12'd403: exp_lut_q57 = 16'd1406;
                12'd404: exp_lut_q57 = 16'd1395;
                12'd405: exp_lut_q57 = 16'd1385;
                12'd406: exp_lut_q57 = 16'd1374;
                12'd407: exp_lut_q57 = 16'd1363;
                12'd408: exp_lut_q57 = 16'd1352;
                12'd409: exp_lut_q57 = 16'd1342;
                12'd410: exp_lut_q57 = 16'd1332;
                12'd411: exp_lut_q57 = 16'd1321;
                12'd412: exp_lut_q57 = 16'd1311;
                12'd413: exp_lut_q57 = 16'd1301;
                12'd414: exp_lut_q57 = 16'd1291;
                12'd415: exp_lut_q57 = 16'd1281;
                12'd416: exp_lut_q57 = 16'd1271;
                12'd417: exp_lut_q57 = 16'd1261;
                12'd418: exp_lut_q57 = 16'd1251;
                12'd419: exp_lut_q57 = 16'd1241;
                12'd420: exp_lut_q57 = 16'd1231;
                12'd421: exp_lut_q57 = 16'd1222;
                12'd422: exp_lut_q57 = 16'd1212;
                12'd423: exp_lut_q57 = 16'd1203;
                12'd424: exp_lut_q57 = 16'd1194;
                12'd425: exp_lut_q57 = 16'd1184;
                12'd426: exp_lut_q57 = 16'd1175;
                12'd427: exp_lut_q57 = 16'd1166;
                12'd428: exp_lut_q57 = 16'd1157;
                12'd429: exp_lut_q57 = 16'd1148;
                12'd430: exp_lut_q57 = 16'd1139;
                12'd431: exp_lut_q57 = 16'd1130;
                12'd432: exp_lut_q57 = 16'd1121;
                12'd433: exp_lut_q57 = 16'd1113;
                12'd434: exp_lut_q57 = 16'd1104;
                12'd435: exp_lut_q57 = 16'd1095;
                12'd436: exp_lut_q57 = 16'd1087;
                12'd437: exp_lut_q57 = 16'd1078;
                12'd438: exp_lut_q57 = 16'd1070;
                12'd439: exp_lut_q57 = 16'd1062;
                12'd440: exp_lut_q57 = 16'd1053;
                12'd441: exp_lut_q57 = 16'd1045;
                12'd442: exp_lut_q57 = 16'd1037;
                12'd443: exp_lut_q57 = 16'd1029;
                12'd444: exp_lut_q57 = 16'd1021;
                12'd445: exp_lut_q57 = 16'd1013;
                12'd446: exp_lut_q57 = 16'd1005;
                12'd447: exp_lut_q57 = 16'd997;
                12'd448: exp_lut_q57 = 16'd990;
                12'd449: exp_lut_q57 = 16'd982;
                12'd450: exp_lut_q57 = 16'd974;
                12'd451: exp_lut_q57 = 16'd967;
                12'd452: exp_lut_q57 = 16'd959;
                12'd453: exp_lut_q57 = 16'd952;
                12'd454: exp_lut_q57 = 16'd944;
                12'd455: exp_lut_q57 = 16'd937;
                12'd456: exp_lut_q57 = 16'd930;
                12'd457: exp_lut_q57 = 16'd922;
                12'd458: exp_lut_q57 = 16'd915;
                12'd459: exp_lut_q57 = 16'd908;
                12'd460: exp_lut_q57 = 16'd901;
                12'd461: exp_lut_q57 = 16'd894;
                12'd462: exp_lut_q57 = 16'd887;
                12'd463: exp_lut_q57 = 16'd880;
                12'd464: exp_lut_q57 = 16'd873;
                12'd465: exp_lut_q57 = 16'd866;
                12'd466: exp_lut_q57 = 16'd860;
                12'd467: exp_lut_q57 = 16'd853;
                12'd468: exp_lut_q57 = 16'd846;
                12'd469: exp_lut_q57 = 16'd840;
                12'd470: exp_lut_q57 = 16'd833;
                12'd471: exp_lut_q57 = 16'd827;
                12'd472: exp_lut_q57 = 16'd820;
                12'd473: exp_lut_q57 = 16'd814;
                12'd474: exp_lut_q57 = 16'd808;
                12'd475: exp_lut_q57 = 16'd801;
                12'd476: exp_lut_q57 = 16'd795;
                12'd477: exp_lut_q57 = 16'd789;
                12'd478: exp_lut_q57 = 16'd783;
                12'd479: exp_lut_q57 = 16'd777;
                12'd480: exp_lut_q57 = 16'd771;
                12'd481: exp_lut_q57 = 16'd765;
                12'd482: exp_lut_q57 = 16'd759;
                12'd483: exp_lut_q57 = 16'd753;
                12'd484: exp_lut_q57 = 16'd747;
                12'd485: exp_lut_q57 = 16'd741;
                12'd486: exp_lut_q57 = 16'd735;
                12'd487: exp_lut_q57 = 16'd730;
                12'd488: exp_lut_q57 = 16'd724;
                12'd489: exp_lut_q57 = 16'd718;
                12'd490: exp_lut_q57 = 16'd713;
                12'd491: exp_lut_q57 = 16'd707;
                12'd492: exp_lut_q57 = 16'd702;
                12'd493: exp_lut_q57 = 16'd696;
                12'd494: exp_lut_q57 = 16'd691;
                12'd495: exp_lut_q57 = 16'd685;
                12'd496: exp_lut_q57 = 16'd680;
                12'd497: exp_lut_q57 = 16'd675;
                12'd498: exp_lut_q57 = 16'd670;
                12'd499: exp_lut_q57 = 16'd664;
                12'd500: exp_lut_q57 = 16'd659;
                12'd501: exp_lut_q57 = 16'd654;
                12'd502: exp_lut_q57 = 16'd649;
                12'd503: exp_lut_q57 = 16'd644;
                12'd504: exp_lut_q57 = 16'd639;
                12'd505: exp_lut_q57 = 16'd634;
                12'd506: exp_lut_q57 = 16'd629;
                12'd507: exp_lut_q57 = 16'd624;
                12'd508: exp_lut_q57 = 16'd619;
                12'd509: exp_lut_q57 = 16'd614;
                12'd510: exp_lut_q57 = 16'd610;
                12'd511: exp_lut_q57 = 16'd605;
                12'd512: exp_lut_q57 = 16'd600;
                12'd513: exp_lut_q57 = 16'd595;
                12'd514: exp_lut_q57 = 16'd591;
                12'd515: exp_lut_q57 = 16'd586;
                12'd516: exp_lut_q57 = 16'd582;
                12'd517: exp_lut_q57 = 16'd577;
                12'd518: exp_lut_q57 = 16'd573;
                12'd519: exp_lut_q57 = 16'd568;
                12'd520: exp_lut_q57 = 16'd564;
                12'd521: exp_lut_q57 = 16'd559;
                12'd522: exp_lut_q57 = 16'd555;
                12'd523: exp_lut_q57 = 16'd551;
                12'd524: exp_lut_q57 = 16'd546;
                12'd525: exp_lut_q57 = 16'd542;
                12'd526: exp_lut_q57 = 16'd538;
                12'd527: exp_lut_q57 = 16'd534;
                12'd528: exp_lut_q57 = 16'd530;
                12'd529: exp_lut_q57 = 16'd526;
                12'd530: exp_lut_q57 = 16'd521;
                12'd531: exp_lut_q57 = 16'd517;
                12'd532: exp_lut_q57 = 16'd513;
                12'd533: exp_lut_q57 = 16'd509;
                12'd534: exp_lut_q57 = 16'd505;
                12'd535: exp_lut_q57 = 16'd501;
                12'd536: exp_lut_q57 = 16'd498;
                12'd537: exp_lut_q57 = 16'd494;
                12'd538: exp_lut_q57 = 16'd490;
                12'd539: exp_lut_q57 = 16'd486;
                12'd540: exp_lut_q57 = 16'd482;
                12'd541: exp_lut_q57 = 16'd478;
                12'd542: exp_lut_q57 = 16'd475;
                12'd543: exp_lut_q57 = 16'd471;
                12'd544: exp_lut_q57 = 16'd467;
                12'd545: exp_lut_q57 = 16'd464;
                12'd546: exp_lut_q57 = 16'd460;
                12'd547: exp_lut_q57 = 16'd457;
                12'd548: exp_lut_q57 = 16'd453;
                12'd549: exp_lut_q57 = 16'd450;
                12'd550: exp_lut_q57 = 16'd446;
                12'd551: exp_lut_q57 = 16'd443;
                12'd552: exp_lut_q57 = 16'd439;
                12'd553: exp_lut_q57 = 16'd436;
                12'd554: exp_lut_q57 = 16'd432;
                12'd555: exp_lut_q57 = 16'd429;
                12'd556: exp_lut_q57 = 16'd426;
                12'd557: exp_lut_q57 = 16'd422;
                12'd558: exp_lut_q57 = 16'd419;
                12'd559: exp_lut_q57 = 16'd416;
                12'd560: exp_lut_q57 = 16'd412;
                12'd561: exp_lut_q57 = 16'd409;
                12'd562: exp_lut_q57 = 16'd406;
                12'd563: exp_lut_q57 = 16'd403;
                12'd564: exp_lut_q57 = 16'd400;
                12'd565: exp_lut_q57 = 16'd397;
                12'd566: exp_lut_q57 = 16'd394;
                12'd567: exp_lut_q57 = 16'd391;
                12'd568: exp_lut_q57 = 16'd387;
                12'd569: exp_lut_q57 = 16'd384;
                12'd570: exp_lut_q57 = 16'd381;
                12'd571: exp_lut_q57 = 16'd379;
                12'd572: exp_lut_q57 = 16'd376;
                12'd573: exp_lut_q57 = 16'd373;
                12'd574: exp_lut_q57 = 16'd370;
                12'd575: exp_lut_q57 = 16'd367;
                12'd576: exp_lut_q57 = 16'd364;
                12'd577: exp_lut_q57 = 16'd361;
                12'd578: exp_lut_q57 = 16'd358;
                12'd579: exp_lut_q57 = 16'd356;
                12'd580: exp_lut_q57 = 16'd353;
                12'd581: exp_lut_q57 = 16'd350;
                12'd582: exp_lut_q57 = 16'd347;
                12'd583: exp_lut_q57 = 16'd345;
                12'd584: exp_lut_q57 = 16'd342;
                12'd585: exp_lut_q57 = 16'd339;
                12'd586: exp_lut_q57 = 16'd337;
                12'd587: exp_lut_q57 = 16'd334;
                12'd588: exp_lut_q57 = 16'd331;
                12'd589: exp_lut_q57 = 16'd329;
                12'd590: exp_lut_q57 = 16'd326;
                12'd591: exp_lut_q57 = 16'd324;
                12'd592: exp_lut_q57 = 16'd321;
                12'd593: exp_lut_q57 = 16'd319;
                12'd594: exp_lut_q57 = 16'd316;
                12'd595: exp_lut_q57 = 16'd314;
                12'd596: exp_lut_q57 = 16'd311;
                12'd597: exp_lut_q57 = 16'd309;
                12'd598: exp_lut_q57 = 16'd307;
                12'd599: exp_lut_q57 = 16'd304;
                12'd600: exp_lut_q57 = 16'd302;
                12'd601: exp_lut_q57 = 16'd299;
                12'd602: exp_lut_q57 = 16'd297;
                12'd603: exp_lut_q57 = 16'd295;
                12'd604: exp_lut_q57 = 16'd292;
                12'd605: exp_lut_q57 = 16'd290;
                12'd606: exp_lut_q57 = 16'd288;
                12'd607: exp_lut_q57 = 16'd286;
                12'd608: exp_lut_q57 = 16'd283;
                12'd609: exp_lut_q57 = 16'd281;
                12'd610: exp_lut_q57 = 16'd279;
                12'd611: exp_lut_q57 = 16'd277;
                12'd612: exp_lut_q57 = 16'd275;
                12'd613: exp_lut_q57 = 16'd273;
                12'd614: exp_lut_q57 = 16'd271;
                12'd615: exp_lut_q57 = 16'd268;
                12'd616: exp_lut_q57 = 16'd266;
                12'd617: exp_lut_q57 = 16'd264;
                12'd618: exp_lut_q57 = 16'd262;
                12'd619: exp_lut_q57 = 16'd260;
                12'd620: exp_lut_q57 = 16'd258;
                12'd621: exp_lut_q57 = 16'd256;
                12'd622: exp_lut_q57 = 16'd254;
                12'd623: exp_lut_q57 = 16'd252;
                12'd624: exp_lut_q57 = 16'd250;
                12'd625: exp_lut_q57 = 16'd248;
                12'd626: exp_lut_q57 = 16'd246;
                12'd627: exp_lut_q57 = 16'd244;
                12'd628: exp_lut_q57 = 16'd242;
                12'd629: exp_lut_q57 = 16'd241;
                12'd630: exp_lut_q57 = 16'd239;
                12'd631: exp_lut_q57 = 16'd237;
                12'd632: exp_lut_q57 = 16'd235;
                12'd633: exp_lut_q57 = 16'd233;
                12'd634: exp_lut_q57 = 16'd231;
                12'd635: exp_lut_q57 = 16'd230;
                12'd636: exp_lut_q57 = 16'd228;
                12'd637: exp_lut_q57 = 16'd226;
                12'd638: exp_lut_q57 = 16'd224;
                12'd639: exp_lut_q57 = 16'd223;
                12'd640: exp_lut_q57 = 16'd221;
                12'd641: exp_lut_q57 = 16'd219;
                12'd642: exp_lut_q57 = 16'd217;
                12'd643: exp_lut_q57 = 16'd216;
                12'd644: exp_lut_q57 = 16'd214;
                12'd645: exp_lut_q57 = 16'd212;
                12'd646: exp_lut_q57 = 16'd211;
                12'd647: exp_lut_q57 = 16'd209;
                12'd648: exp_lut_q57 = 16'd207;
                12'd649: exp_lut_q57 = 16'd206;
                12'd650: exp_lut_q57 = 16'd204;
                12'd651: exp_lut_q57 = 16'd203;
                12'd652: exp_lut_q57 = 16'd201;
                12'd653: exp_lut_q57 = 16'd199;
                12'd654: exp_lut_q57 = 16'd198;
                12'd655: exp_lut_q57 = 16'd196;
                12'd656: exp_lut_q57 = 16'd195;
                12'd657: exp_lut_q57 = 16'd193;
                12'd658: exp_lut_q57 = 16'd192;
                12'd659: exp_lut_q57 = 16'd190;
                12'd660: exp_lut_q57 = 16'd189;
                12'd661: exp_lut_q57 = 16'd187;
                12'd662: exp_lut_q57 = 16'd186;
                12'd663: exp_lut_q57 = 16'd184;
                12'd664: exp_lut_q57 = 16'd183;
                12'd665: exp_lut_q57 = 16'd182;
                12'd666: exp_lut_q57 = 16'd180;
                12'd667: exp_lut_q57 = 16'd179;
                12'd668: exp_lut_q57 = 16'd177;
                12'd669: exp_lut_q57 = 16'd176;
                12'd670: exp_lut_q57 = 16'd175;
                12'd671: exp_lut_q57 = 16'd173;
                12'd672: exp_lut_q57 = 16'd172;
                12'd673: exp_lut_q57 = 16'd171;
                12'd674: exp_lut_q57 = 16'd169;
                12'd675: exp_lut_q57 = 16'd168;
                12'd676: exp_lut_q57 = 16'd167;
                12'd677: exp_lut_q57 = 16'd165;
                12'd678: exp_lut_q57 = 16'd164;
                12'd679: exp_lut_q57 = 16'd163;
                12'd680: exp_lut_q57 = 16'd162;
                12'd681: exp_lut_q57 = 16'd160;
                12'd682: exp_lut_q57 = 16'd159;
                12'd683: exp_lut_q57 = 16'd158;
                12'd684: exp_lut_q57 = 16'd157;
                12'd685: exp_lut_q57 = 16'd155;
                12'd686: exp_lut_q57 = 16'd154;
                12'd687: exp_lut_q57 = 16'd153;
                12'd688: exp_lut_q57 = 16'd152;
                12'd689: exp_lut_q57 = 16'd151;
                12'd690: exp_lut_q57 = 16'd149;
                12'd691: exp_lut_q57 = 16'd148;
                12'd692: exp_lut_q57 = 16'd147;
                12'd693: exp_lut_q57 = 16'd146;
                12'd694: exp_lut_q57 = 16'd145;
                12'd695: exp_lut_q57 = 16'd144;
                12'd696: exp_lut_q57 = 16'd143;
                12'd697: exp_lut_q57 = 16'd141;
                12'd698: exp_lut_q57 = 16'd140;
                12'd699: exp_lut_q57 = 16'd139;
                12'd700: exp_lut_q57 = 16'd138;
                12'd701: exp_lut_q57 = 16'd137;
                12'd702: exp_lut_q57 = 16'd136;
                12'd703: exp_lut_q57 = 16'd135;
                12'd704: exp_lut_q57 = 16'd134;
                12'd705: exp_lut_q57 = 16'd133;
                12'd706: exp_lut_q57 = 16'd132;
                12'd707: exp_lut_q57 = 16'd131;
                12'd708: exp_lut_q57 = 16'd130;
                12'd709: exp_lut_q57 = 16'd129;
                12'd710: exp_lut_q57 = 16'd128;
                12'd711: exp_lut_q57 = 16'd127;
                12'd712: exp_lut_q57 = 16'd126;
                12'd713: exp_lut_q57 = 16'd125;
                12'd714: exp_lut_q57 = 16'd124;
                12'd715: exp_lut_q57 = 16'd123;
                12'd716: exp_lut_q57 = 16'd122;
                12'd717: exp_lut_q57 = 16'd121;
                12'd718: exp_lut_q57 = 16'd120;
                12'd719: exp_lut_q57 = 16'd119;
                12'd720: exp_lut_q57 = 16'd118;
                12'd721: exp_lut_q57 = 16'd117;
                12'd722: exp_lut_q57 = 16'd116;
                12'd723: exp_lut_q57 = 16'd115;
                12'd724: exp_lut_q57 = 16'd115;
                12'd725: exp_lut_q57 = 16'd114;
                12'd726: exp_lut_q57 = 16'd113;
                12'd727: exp_lut_q57 = 16'd112;
                12'd728: exp_lut_q57 = 16'd111;
                12'd729: exp_lut_q57 = 16'd110;
                12'd730: exp_lut_q57 = 16'd109;
                12'd731: exp_lut_q57 = 16'd108;
                12'd732: exp_lut_q57 = 16'd108;
                12'd733: exp_lut_q57 = 16'd107;
                12'd734: exp_lut_q57 = 16'd106;
                12'd735: exp_lut_q57 = 16'd105;
                12'd736: exp_lut_q57 = 16'd104;
                12'd737: exp_lut_q57 = 16'd103;
                12'd738: exp_lut_q57 = 16'd103;
                12'd739: exp_lut_q57 = 16'd102;
                12'd740: exp_lut_q57 = 16'd101;
                12'd741: exp_lut_q57 = 16'd100;
                12'd742: exp_lut_q57 = 16'd100;
                12'd743: exp_lut_q57 = 16'd99;
                12'd744: exp_lut_q57 = 16'd98;
                12'd745: exp_lut_q57 = 16'd97;
                12'd746: exp_lut_q57 = 16'd96;
                12'd747: exp_lut_q57 = 16'd96;
                12'd748: exp_lut_q57 = 16'd95;
                12'd749: exp_lut_q57 = 16'd94;
                12'd750: exp_lut_q57 = 16'd93;
                12'd751: exp_lut_q57 = 16'd93;
                12'd752: exp_lut_q57 = 16'd92;
                12'd753: exp_lut_q57 = 16'd91;
                12'd754: exp_lut_q57 = 16'd91;
                12'd755: exp_lut_q57 = 16'd90;
                12'd756: exp_lut_q57 = 16'd89;
                12'd757: exp_lut_q57 = 16'd89;
                12'd758: exp_lut_q57 = 16'd88;
                12'd759: exp_lut_q57 = 16'd87;
                12'd760: exp_lut_q57 = 16'd86;
                12'd761: exp_lut_q57 = 16'd86;
                12'd762: exp_lut_q57 = 16'd85;
                12'd763: exp_lut_q57 = 16'd84;
                12'd764: exp_lut_q57 = 16'd84;
                12'd765: exp_lut_q57 = 16'd83;
                12'd766: exp_lut_q57 = 16'd83;
                12'd767: exp_lut_q57 = 16'd82;
                12'd768: exp_lut_q57 = 16'd81;
                12'd769: exp_lut_q57 = 16'd81;
                12'd770: exp_lut_q57 = 16'd80;
                12'd771: exp_lut_q57 = 16'd79;
                12'd772: exp_lut_q57 = 16'd79;
                12'd773: exp_lut_q57 = 16'd78;
                12'd774: exp_lut_q57 = 16'd78;
                12'd775: exp_lut_q57 = 16'd77;
                12'd776: exp_lut_q57 = 16'd76;
                12'd777: exp_lut_q57 = 16'd76;
                12'd778: exp_lut_q57 = 16'd75;
                12'd779: exp_lut_q57 = 16'd75;
                12'd780: exp_lut_q57 = 16'd74;
                12'd781: exp_lut_q57 = 16'd73;
                12'd782: exp_lut_q57 = 16'd73;
                12'd783: exp_lut_q57 = 16'd72;
                12'd784: exp_lut_q57 = 16'd72;
                12'd785: exp_lut_q57 = 16'd71;
                12'd786: exp_lut_q57 = 16'd71;
                12'd787: exp_lut_q57 = 16'd70;
                12'd788: exp_lut_q57 = 16'd69;
                12'd789: exp_lut_q57 = 16'd69;
                12'd790: exp_lut_q57 = 16'd68;
                12'd791: exp_lut_q57 = 16'd68;
                12'd792: exp_lut_q57 = 16'd67;
                12'd793: exp_lut_q57 = 16'd67;
                12'd794: exp_lut_q57 = 16'd66;
                12'd795: exp_lut_q57 = 16'd66;
                12'd796: exp_lut_q57 = 16'd65;
                12'd797: exp_lut_q57 = 16'd65;
                12'd798: exp_lut_q57 = 16'd64;
                12'd799: exp_lut_q57 = 16'd64;
                12'd800: exp_lut_q57 = 16'd63;
                12'd801: exp_lut_q57 = 16'd63;
                12'd802: exp_lut_q57 = 16'd62;
                12'd803: exp_lut_q57 = 16'd62;
                12'd804: exp_lut_q57 = 16'd61;
                12'd805: exp_lut_q57 = 16'd61;
                12'd806: exp_lut_q57 = 16'd60;
                12'd807: exp_lut_q57 = 16'd60;
                12'd808: exp_lut_q57 = 16'd59;
                12'd809: exp_lut_q57 = 16'd59;
                12'd810: exp_lut_q57 = 16'd59;
                12'd811: exp_lut_q57 = 16'd58;
                12'd812: exp_lut_q57 = 16'd58;
                12'd813: exp_lut_q57 = 16'd57;
                12'd814: exp_lut_q57 = 16'd57;
                12'd815: exp_lut_q57 = 16'd56;
                12'd816: exp_lut_q57 = 16'd56;
                12'd817: exp_lut_q57 = 16'd55;
                12'd818: exp_lut_q57 = 16'd55;
                12'd819: exp_lut_q57 = 16'd55;
                12'd820: exp_lut_q57 = 16'd54;
                12'd821: exp_lut_q57 = 16'd54;
                12'd822: exp_lut_q57 = 16'd53;
                12'd823: exp_lut_q57 = 16'd53;
                12'd824: exp_lut_q57 = 16'd52;
                12'd825: exp_lut_q57 = 16'd52;
                12'd826: exp_lut_q57 = 16'd52;
                12'd827: exp_lut_q57 = 16'd51;
                12'd828: exp_lut_q57 = 16'd51;
                12'd829: exp_lut_q57 = 16'd50;
                12'd830: exp_lut_q57 = 16'd50;
                12'd831: exp_lut_q57 = 16'd50;
                12'd832: exp_lut_q57 = 16'd49;
                12'd833: exp_lut_q57 = 16'd49;
                12'd834: exp_lut_q57 = 16'd49;
                12'd835: exp_lut_q57 = 16'd48;
                12'd836: exp_lut_q57 = 16'd48;
                12'd837: exp_lut_q57 = 16'd47;
                12'd838: exp_lut_q57 = 16'd47;
                12'd839: exp_lut_q57 = 16'd47;
                12'd840: exp_lut_q57 = 16'd46;
                12'd841: exp_lut_q57 = 16'd46;
                12'd842: exp_lut_q57 = 16'd46;
                12'd843: exp_lut_q57 = 16'd45;
                12'd844: exp_lut_q57 = 16'd45;
                12'd845: exp_lut_q57 = 16'd45;
                12'd846: exp_lut_q57 = 16'd44;
                12'd847: exp_lut_q57 = 16'd44;
                12'd848: exp_lut_q57 = 16'd43;
                12'd849: exp_lut_q57 = 16'd43;
                12'd850: exp_lut_q57 = 16'd43;
                12'd851: exp_lut_q57 = 16'd42;
                12'd852: exp_lut_q57 = 16'd42;
                12'd853: exp_lut_q57 = 16'd42;
                12'd854: exp_lut_q57 = 16'd41;
                12'd855: exp_lut_q57 = 16'd41;
                12'd856: exp_lut_q57 = 16'd41;
                12'd857: exp_lut_q57 = 16'd41;
                12'd858: exp_lut_q57 = 16'd40;
                12'd859: exp_lut_q57 = 16'd40;
                12'd860: exp_lut_q57 = 16'd40;
                12'd861: exp_lut_q57 = 16'd39;
                12'd862: exp_lut_q57 = 16'd39;
                12'd863: exp_lut_q57 = 16'd39;
                12'd864: exp_lut_q57 = 16'd38;
                12'd865: exp_lut_q57 = 16'd38;
                12'd866: exp_lut_q57 = 16'd38;
                12'd867: exp_lut_q57 = 16'd37;
                12'd868: exp_lut_q57 = 16'd37;
                12'd869: exp_lut_q57 = 16'd37;
                12'd870: exp_lut_q57 = 16'd37;
                12'd871: exp_lut_q57 = 16'd36;
                12'd872: exp_lut_q57 = 16'd36;
                12'd873: exp_lut_q57 = 16'd36;
                12'd874: exp_lut_q57 = 16'd35;
                12'd875: exp_lut_q57 = 16'd35;
                12'd876: exp_lut_q57 = 16'd35;
                12'd877: exp_lut_q57 = 16'd35;
                12'd878: exp_lut_q57 = 16'd34;
                12'd879: exp_lut_q57 = 16'd34;
                12'd880: exp_lut_q57 = 16'd34;
                12'd881: exp_lut_q57 = 16'd34;
                12'd882: exp_lut_q57 = 16'd33;
                12'd883: exp_lut_q57 = 16'd33;
                12'd884: exp_lut_q57 = 16'd33;
                12'd885: exp_lut_q57 = 16'd33;
                12'd886: exp_lut_q57 = 16'd32;
                12'd887: exp_lut_q57 = 16'd32;
                12'd888: exp_lut_q57 = 16'd32;
                12'd889: exp_lut_q57 = 16'd32;
                12'd890: exp_lut_q57 = 16'd31;
                12'd891: exp_lut_q57 = 16'd31;
                12'd892: exp_lut_q57 = 16'd31;
                12'd893: exp_lut_q57 = 16'd31;
                12'd894: exp_lut_q57 = 16'd30;
                12'd895: exp_lut_q57 = 16'd30;
                12'd896: exp_lut_q57 = 16'd30;
                12'd897: exp_lut_q57 = 16'd30;
                12'd898: exp_lut_q57 = 16'd29;
                12'd899: exp_lut_q57 = 16'd29;
                12'd900: exp_lut_q57 = 16'd29;
                12'd901: exp_lut_q57 = 16'd29;
                12'd902: exp_lut_q57 = 16'd29;
                12'd903: exp_lut_q57 = 16'd28;
                12'd904: exp_lut_q57 = 16'd28;
                12'd905: exp_lut_q57 = 16'd28;
                12'd906: exp_lut_q57 = 16'd28;
                12'd907: exp_lut_q57 = 16'd27;
                12'd908: exp_lut_q57 = 16'd27;
                12'd909: exp_lut_q57 = 16'd27;
                12'd910: exp_lut_q57 = 16'd27;
                12'd911: exp_lut_q57 = 16'd27;
                12'd912: exp_lut_q57 = 16'd26;
                12'd913: exp_lut_q57 = 16'd26;
                12'd914: exp_lut_q57 = 16'd26;
                12'd915: exp_lut_q57 = 16'd26;
                12'd916: exp_lut_q57 = 16'd26;
                12'd917: exp_lut_q57 = 16'd25;
                12'd918: exp_lut_q57 = 16'd25;
                12'd919: exp_lut_q57 = 16'd25;
                12'd920: exp_lut_q57 = 16'd25;
                12'd921: exp_lut_q57 = 16'd25;
                12'd922: exp_lut_q57 = 16'd24;
                12'd923: exp_lut_q57 = 16'd24;
                12'd924: exp_lut_q57 = 16'd24;
                12'd925: exp_lut_q57 = 16'd24;
                12'd926: exp_lut_q57 = 16'd24;
                12'd927: exp_lut_q57 = 16'd23;
                12'd928: exp_lut_q57 = 16'd23;
                12'd929: exp_lut_q57 = 16'd23;
                12'd930: exp_lut_q57 = 16'd23;
                12'd931: exp_lut_q57 = 16'd23;
                12'd932: exp_lut_q57 = 16'd23;
                12'd933: exp_lut_q57 = 16'd22;
                12'd934: exp_lut_q57 = 16'd22;
                12'd935: exp_lut_q57 = 16'd22;
                12'd936: exp_lut_q57 = 16'd22;
                12'd937: exp_lut_q57 = 16'd22;
                12'd938: exp_lut_q57 = 16'd22;
                12'd939: exp_lut_q57 = 16'd21;
                12'd940: exp_lut_q57 = 16'd21;
                12'd941: exp_lut_q57 = 16'd21;
                12'd942: exp_lut_q57 = 16'd21;
                12'd943: exp_lut_q57 = 16'd21;
                12'd944: exp_lut_q57 = 16'd21;
                12'd945: exp_lut_q57 = 16'd20;
                12'd946: exp_lut_q57 = 16'd20;
                12'd947: exp_lut_q57 = 16'd20;
                12'd948: exp_lut_q57 = 16'd20;
                12'd949: exp_lut_q57 = 16'd20;
                12'd950: exp_lut_q57 = 16'd20;
                12'd951: exp_lut_q57 = 16'd19;
                12'd952: exp_lut_q57 = 16'd19;
                12'd953: exp_lut_q57 = 16'd19;
                12'd954: exp_lut_q57 = 16'd19;
                12'd955: exp_lut_q57 = 16'd19;
                12'd956: exp_lut_q57 = 16'd19;
                12'd957: exp_lut_q57 = 16'd19;
                12'd958: exp_lut_q57 = 16'd18;
                12'd959: exp_lut_q57 = 16'd18;
                12'd960: exp_lut_q57 = 16'd18;
                12'd961: exp_lut_q57 = 16'd18;
                12'd962: exp_lut_q57 = 16'd18;
                12'd963: exp_lut_q57 = 16'd18;
                12'd964: exp_lut_q57 = 16'd18;
                12'd965: exp_lut_q57 = 16'd17;
                12'd966: exp_lut_q57 = 16'd17;
                12'd967: exp_lut_q57 = 16'd17;
                12'd968: exp_lut_q57 = 16'd17;
                12'd969: exp_lut_q57 = 16'd17;
                12'd970: exp_lut_q57 = 16'd17;
                12'd971: exp_lut_q57 = 16'd17;
                12'd972: exp_lut_q57 = 16'd17;
                12'd973: exp_lut_q57 = 16'd16;
                12'd974: exp_lut_q57 = 16'd16;
                12'd975: exp_lut_q57 = 16'd16;
                12'd976: exp_lut_q57 = 16'd16;
                12'd977: exp_lut_q57 = 16'd16;
                12'd978: exp_lut_q57 = 16'd16;
                12'd979: exp_lut_q57 = 16'd16;
                12'd980: exp_lut_q57 = 16'd16;
                12'd981: exp_lut_q57 = 16'd15;
                12'd982: exp_lut_q57 = 16'd15;
                12'd983: exp_lut_q57 = 16'd15;
                12'd984: exp_lut_q57 = 16'd15;
                12'd985: exp_lut_q57 = 16'd15;
                12'd986: exp_lut_q57 = 16'd15;
                12'd987: exp_lut_q57 = 16'd15;
                12'd988: exp_lut_q57 = 16'd15;
                12'd989: exp_lut_q57 = 16'd14;
                12'd990: exp_lut_q57 = 16'd14;
                12'd991: exp_lut_q57 = 16'd14;
                12'd992: exp_lut_q57 = 16'd14;
                12'd993: exp_lut_q57 = 16'd14;
                12'd994: exp_lut_q57 = 16'd14;
                12'd995: exp_lut_q57 = 16'd14;
                12'd996: exp_lut_q57 = 16'd14;
                12'd997: exp_lut_q57 = 16'd14;
                12'd998: exp_lut_q57 = 16'd13;
                12'd999: exp_lut_q57 = 16'd13;
                12'd1000: exp_lut_q57 = 16'd13;
                12'd1001: exp_lut_q57 = 16'd13;
                12'd1002: exp_lut_q57 = 16'd13;
                12'd1003: exp_lut_q57 = 16'd13;
                12'd1004: exp_lut_q57 = 16'd13;
                12'd1005: exp_lut_q57 = 16'd13;
                12'd1006: exp_lut_q57 = 16'd13;
                12'd1007: exp_lut_q57 = 16'd13;
                12'd1008: exp_lut_q57 = 16'd12;
                12'd1009: exp_lut_q57 = 16'd12;
                12'd1010: exp_lut_q57 = 16'd12;
                12'd1011: exp_lut_q57 = 16'd12;
                12'd1012: exp_lut_q57 = 16'd12;
                12'd1013: exp_lut_q57 = 16'd12;
                12'd1014: exp_lut_q57 = 16'd12;
                12'd1015: exp_lut_q57 = 16'd12;
                12'd1016: exp_lut_q57 = 16'd12;
                12'd1017: exp_lut_q57 = 16'd12;
                12'd1018: exp_lut_q57 = 16'd12;
                12'd1019: exp_lut_q57 = 16'd11;
                12'd1020: exp_lut_q57 = 16'd11;
                12'd1021: exp_lut_q57 = 16'd11;
                12'd1022: exp_lut_q57 = 16'd11;
                12'd1023: exp_lut_q57 = 16'd11;
                12'd1024: exp_lut_q57 = 16'd11;
                12'd1025: exp_lut_q57 = 16'd11;
                12'd1026: exp_lut_q57 = 16'd11;
                12'd1027: exp_lut_q57 = 16'd11;
                12'd1028: exp_lut_q57 = 16'd11;
                12'd1029: exp_lut_q57 = 16'd11;
                12'd1030: exp_lut_q57 = 16'd10;
                12'd1031: exp_lut_q57 = 16'd10;
                12'd1032: exp_lut_q57 = 16'd10;
                12'd1033: exp_lut_q57 = 16'd10;
                12'd1034: exp_lut_q57 = 16'd10;
                12'd1035: exp_lut_q57 = 16'd10;
                12'd1036: exp_lut_q57 = 16'd10;
                12'd1037: exp_lut_q57 = 16'd10;
                12'd1038: exp_lut_q57 = 16'd10;
                12'd1039: exp_lut_q57 = 16'd10;
                12'd1040: exp_lut_q57 = 16'd10;
                12'd1041: exp_lut_q57 = 16'd10;
                12'd1042: exp_lut_q57 = 16'd10;
                12'd1043: exp_lut_q57 = 16'd9;
                12'd1044: exp_lut_q57 = 16'd9;
                12'd1045: exp_lut_q57 = 16'd9;
                12'd1046: exp_lut_q57 = 16'd9;
                12'd1047: exp_lut_q57 = 16'd9;
                12'd1048: exp_lut_q57 = 16'd9;
                12'd1049: exp_lut_q57 = 16'd9;
                12'd1050: exp_lut_q57 = 16'd9;
                12'd1051: exp_lut_q57 = 16'd9;
                12'd1052: exp_lut_q57 = 16'd9;
                12'd1053: exp_lut_q57 = 16'd9;
                12'd1054: exp_lut_q57 = 16'd9;
                12'd1055: exp_lut_q57 = 16'd9;
                12'd1056: exp_lut_q57 = 16'd9;
                12'd1057: exp_lut_q57 = 16'd8;
                12'd1058: exp_lut_q57 = 16'd8;
                12'd1059: exp_lut_q57 = 16'd8;
                12'd1060: exp_lut_q57 = 16'd8;
                12'd1061: exp_lut_q57 = 16'd8;
                12'd1062: exp_lut_q57 = 16'd8;
                12'd1063: exp_lut_q57 = 16'd8;
                12'd1064: exp_lut_q57 = 16'd8;
                12'd1065: exp_lut_q57 = 16'd8;
                12'd1066: exp_lut_q57 = 16'd8;
                12'd1067: exp_lut_q57 = 16'd8;
                12'd1068: exp_lut_q57 = 16'd8;
                12'd1069: exp_lut_q57 = 16'd8;
                12'd1070: exp_lut_q57 = 16'd8;
                12'd1071: exp_lut_q57 = 16'd8;
                12'd1072: exp_lut_q57 = 16'd8;
                12'd1073: exp_lut_q57 = 16'd7;
                12'd1074: exp_lut_q57 = 16'd7;
                12'd1075: exp_lut_q57 = 16'd7;
                12'd1076: exp_lut_q57 = 16'd7;
                12'd1077: exp_lut_q57 = 16'd7;
                12'd1078: exp_lut_q57 = 16'd7;
                12'd1079: exp_lut_q57 = 16'd7;
                12'd1080: exp_lut_q57 = 16'd7;
                12'd1081: exp_lut_q57 = 16'd7;
                12'd1082: exp_lut_q57 = 16'd7;
                12'd1083: exp_lut_q57 = 16'd7;
                12'd1084: exp_lut_q57 = 16'd7;
                12'd1085: exp_lut_q57 = 16'd7;
                12'd1086: exp_lut_q57 = 16'd7;
                12'd1087: exp_lut_q57 = 16'd7;
                12'd1088: exp_lut_q57 = 16'd7;
                12'd1089: exp_lut_q57 = 16'd7;
                12'd1090: exp_lut_q57 = 16'd7;
                12'd1091: exp_lut_q57 = 16'd7;
                12'd1092: exp_lut_q57 = 16'd6;
                12'd1093: exp_lut_q57 = 16'd6;
                12'd1094: exp_lut_q57 = 16'd6;
                12'd1095: exp_lut_q57 = 16'd6;
                12'd1096: exp_lut_q57 = 16'd6;
                12'd1097: exp_lut_q57 = 16'd6;
                12'd1098: exp_lut_q57 = 16'd6;
                12'd1099: exp_lut_q57 = 16'd6;
                12'd1100: exp_lut_q57 = 16'd6;
                12'd1101: exp_lut_q57 = 16'd6;
                12'd1102: exp_lut_q57 = 16'd6;
                12'd1103: exp_lut_q57 = 16'd6;
                12'd1104: exp_lut_q57 = 16'd6;
                12'd1105: exp_lut_q57 = 16'd6;
                12'd1106: exp_lut_q57 = 16'd6;
                12'd1107: exp_lut_q57 = 16'd6;
                12'd1108: exp_lut_q57 = 16'd6;
                12'd1109: exp_lut_q57 = 16'd6;
                12'd1110: exp_lut_q57 = 16'd6;
                12'd1111: exp_lut_q57 = 16'd6;
                12'd1112: exp_lut_q57 = 16'd6;
                12'd1113: exp_lut_q57 = 16'd5;
                12'd1114: exp_lut_q57 = 16'd5;
                12'd1115: exp_lut_q57 = 16'd5;
                12'd1116: exp_lut_q57 = 16'd5;
                12'd1117: exp_lut_q57 = 16'd5;
                12'd1118: exp_lut_q57 = 16'd5;
                12'd1119: exp_lut_q57 = 16'd5;
                12'd1120: exp_lut_q57 = 16'd5;
                12'd1121: exp_lut_q57 = 16'd5;
                12'd1122: exp_lut_q57 = 16'd5;
                12'd1123: exp_lut_q57 = 16'd5;
                12'd1124: exp_lut_q57 = 16'd5;
                12'd1125: exp_lut_q57 = 16'd5;
                12'd1126: exp_lut_q57 = 16'd5;
                12'd1127: exp_lut_q57 = 16'd5;
                12'd1128: exp_lut_q57 = 16'd5;
                12'd1129: exp_lut_q57 = 16'd5;
                12'd1130: exp_lut_q57 = 16'd5;
                12'd1131: exp_lut_q57 = 16'd5;
                12'd1132: exp_lut_q57 = 16'd5;
                12'd1133: exp_lut_q57 = 16'd5;
                12'd1134: exp_lut_q57 = 16'd5;
                12'd1135: exp_lut_q57 = 16'd5;
                12'd1136: exp_lut_q57 = 16'd5;
                12'd1137: exp_lut_q57 = 16'd5;
                12'd1138: exp_lut_q57 = 16'd5;
                12'd1139: exp_lut_q57 = 16'd4;
                12'd1140: exp_lut_q57 = 16'd4;
                12'd1141: exp_lut_q57 = 16'd4;
                12'd1142: exp_lut_q57 = 16'd4;
                12'd1143: exp_lut_q57 = 16'd4;
                12'd1144: exp_lut_q57 = 16'd4;
                12'd1145: exp_lut_q57 = 16'd4;
                12'd1146: exp_lut_q57 = 16'd4;
                12'd1147: exp_lut_q57 = 16'd4;
                12'd1148: exp_lut_q57 = 16'd4;
                12'd1149: exp_lut_q57 = 16'd4;
                12'd1150: exp_lut_q57 = 16'd4;
                12'd1151: exp_lut_q57 = 16'd4;
                12'd1152: exp_lut_q57 = 16'd4;
                12'd1153: exp_lut_q57 = 16'd4;
                12'd1154: exp_lut_q57 = 16'd4;
                12'd1155: exp_lut_q57 = 16'd4;
                12'd1156: exp_lut_q57 = 16'd4;
                12'd1157: exp_lut_q57 = 16'd4;
                12'd1158: exp_lut_q57 = 16'd4;
                12'd1159: exp_lut_q57 = 16'd4;
                12'd1160: exp_lut_q57 = 16'd4;
                12'd1161: exp_lut_q57 = 16'd4;
                12'd1162: exp_lut_q57 = 16'd4;
                12'd1163: exp_lut_q57 = 16'd4;
                12'd1164: exp_lut_q57 = 16'd4;
                12'd1165: exp_lut_q57 = 16'd4;
                12'd1166: exp_lut_q57 = 16'd4;
                12'd1167: exp_lut_q57 = 16'd4;
                12'd1168: exp_lut_q57 = 16'd4;
                12'd1169: exp_lut_q57 = 16'd4;
                12'd1170: exp_lut_q57 = 16'd4;
                12'd1171: exp_lut_q57 = 16'd3;
                12'd1172: exp_lut_q57 = 16'd3;
                12'd1173: exp_lut_q57 = 16'd3;
                12'd1174: exp_lut_q57 = 16'd3;
                12'd1175: exp_lut_q57 = 16'd3;
                12'd1176: exp_lut_q57 = 16'd3;
                12'd1177: exp_lut_q57 = 16'd3;
                12'd1178: exp_lut_q57 = 16'd3;
                12'd1179: exp_lut_q57 = 16'd3;
                12'd1180: exp_lut_q57 = 16'd3;
                12'd1181: exp_lut_q57 = 16'd3;
                12'd1182: exp_lut_q57 = 16'd3;
                12'd1183: exp_lut_q57 = 16'd3;
                12'd1184: exp_lut_q57 = 16'd3;
                12'd1185: exp_lut_q57 = 16'd3;
                12'd1186: exp_lut_q57 = 16'd3;
                12'd1187: exp_lut_q57 = 16'd3;
                12'd1188: exp_lut_q57 = 16'd3;
                12'd1189: exp_lut_q57 = 16'd3;
                12'd1190: exp_lut_q57 = 16'd3;
                12'd1191: exp_lut_q57 = 16'd3;
                12'd1192: exp_lut_q57 = 16'd3;
                12'd1193: exp_lut_q57 = 16'd3;
                12'd1194: exp_lut_q57 = 16'd3;
                12'd1195: exp_lut_q57 = 16'd3;
                12'd1196: exp_lut_q57 = 16'd3;
                12'd1197: exp_lut_q57 = 16'd3;
                12'd1198: exp_lut_q57 = 16'd3;
                12'd1199: exp_lut_q57 = 16'd3;
                12'd1200: exp_lut_q57 = 16'd3;
                12'd1201: exp_lut_q57 = 16'd3;
                12'd1202: exp_lut_q57 = 16'd3;
                12'd1203: exp_lut_q57 = 16'd3;
                12'd1204: exp_lut_q57 = 16'd3;
                12'd1205: exp_lut_q57 = 16'd3;
                12'd1206: exp_lut_q57 = 16'd3;
                12'd1207: exp_lut_q57 = 16'd3;
                12'd1208: exp_lut_q57 = 16'd3;
                12'd1209: exp_lut_q57 = 16'd3;
                12'd1210: exp_lut_q57 = 16'd3;
                12'd1211: exp_lut_q57 = 16'd3;
                12'd1212: exp_lut_q57 = 16'd3;
                12'd1213: exp_lut_q57 = 16'd3;
                12'd1214: exp_lut_q57 = 16'd2;
                12'd1215: exp_lut_q57 = 16'd2;
                12'd1216: exp_lut_q57 = 16'd2;
                12'd1217: exp_lut_q57 = 16'd2;
                12'd1218: exp_lut_q57 = 16'd2;
                12'd1219: exp_lut_q57 = 16'd2;
                12'd1220: exp_lut_q57 = 16'd2;
                12'd1221: exp_lut_q57 = 16'd2;
                12'd1222: exp_lut_q57 = 16'd2;
                12'd1223: exp_lut_q57 = 16'd2;
                12'd1224: exp_lut_q57 = 16'd2;
                12'd1225: exp_lut_q57 = 16'd2;
                12'd1226: exp_lut_q57 = 16'd2;
                12'd1227: exp_lut_q57 = 16'd2;
                12'd1228: exp_lut_q57 = 16'd2;
                12'd1229: exp_lut_q57 = 16'd2;
                12'd1230: exp_lut_q57 = 16'd2;
                12'd1231: exp_lut_q57 = 16'd2;
                12'd1232: exp_lut_q57 = 16'd2;
                12'd1233: exp_lut_q57 = 16'd2;
                12'd1234: exp_lut_q57 = 16'd2;
                12'd1235: exp_lut_q57 = 16'd2;
                12'd1236: exp_lut_q57 = 16'd2;
                12'd1237: exp_lut_q57 = 16'd2;
                12'd1238: exp_lut_q57 = 16'd2;
                12'd1239: exp_lut_q57 = 16'd2;
                12'd1240: exp_lut_q57 = 16'd2;
                12'd1241: exp_lut_q57 = 16'd2;
                12'd1242: exp_lut_q57 = 16'd2;
                12'd1243: exp_lut_q57 = 16'd2;
                12'd1244: exp_lut_q57 = 16'd2;
                12'd1245: exp_lut_q57 = 16'd2;
                12'd1246: exp_lut_q57 = 16'd2;
                12'd1247: exp_lut_q57 = 16'd2;
                12'd1248: exp_lut_q57 = 16'd2;
                12'd1249: exp_lut_q57 = 16'd2;
                12'd1250: exp_lut_q57 = 16'd2;
                12'd1251: exp_lut_q57 = 16'd2;
                12'd1252: exp_lut_q57 = 16'd2;
                12'd1253: exp_lut_q57 = 16'd2;
                12'd1254: exp_lut_q57 = 16'd2;
                12'd1255: exp_lut_q57 = 16'd2;
                12'd1256: exp_lut_q57 = 16'd2;
                12'd1257: exp_lut_q57 = 16'd2;
                12'd1258: exp_lut_q57 = 16'd2;
                12'd1259: exp_lut_q57 = 16'd2;
                12'd1260: exp_lut_q57 = 16'd2;
                12'd1261: exp_lut_q57 = 16'd2;
                12'd1262: exp_lut_q57 = 16'd2;
                12'd1263: exp_lut_q57 = 16'd2;
                12'd1264: exp_lut_q57 = 16'd2;
                12'd1265: exp_lut_q57 = 16'd2;
                12'd1266: exp_lut_q57 = 16'd2;
                12'd1267: exp_lut_q57 = 16'd2;
                12'd1268: exp_lut_q57 = 16'd2;
                12'd1269: exp_lut_q57 = 16'd2;
                12'd1270: exp_lut_q57 = 16'd2;
                12'd1271: exp_lut_q57 = 16'd2;
                12'd1272: exp_lut_q57 = 16'd2;
                12'd1273: exp_lut_q57 = 16'd2;
                12'd1274: exp_lut_q57 = 16'd2;
                12'd1275: exp_lut_q57 = 16'd2;
                12'd1276: exp_lut_q57 = 16'd2;
                12'd1277: exp_lut_q57 = 16'd2;
                12'd1278: exp_lut_q57 = 16'd2;
                12'd1279: exp_lut_q57 = 16'd1;
                12'd1280: exp_lut_q57 = 16'd1;
                12'd1281: exp_lut_q57 = 16'd1;
                12'd1282: exp_lut_q57 = 16'd1;
                12'd1283: exp_lut_q57 = 16'd1;
                12'd1284: exp_lut_q57 = 16'd1;
                12'd1285: exp_lut_q57 = 16'd1;
                12'd1286: exp_lut_q57 = 16'd1;
                12'd1287: exp_lut_q57 = 16'd1;
                12'd1288: exp_lut_q57 = 16'd1;
                12'd1289: exp_lut_q57 = 16'd1;
                12'd1290: exp_lut_q57 = 16'd1;
                12'd1291: exp_lut_q57 = 16'd1;
                12'd1292: exp_lut_q57 = 16'd1;
                12'd1293: exp_lut_q57 = 16'd1;
                12'd1294: exp_lut_q57 = 16'd1;
                12'd1295: exp_lut_q57 = 16'd1;
                12'd1296: exp_lut_q57 = 16'd1;
                12'd1297: exp_lut_q57 = 16'd1;
                12'd1298: exp_lut_q57 = 16'd1;
                12'd1299: exp_lut_q57 = 16'd1;
                12'd1300: exp_lut_q57 = 16'd1;
                12'd1301: exp_lut_q57 = 16'd1;
                12'd1302: exp_lut_q57 = 16'd1;
                12'd1303: exp_lut_q57 = 16'd1;
                12'd1304: exp_lut_q57 = 16'd1;
                12'd1305: exp_lut_q57 = 16'd1;
                12'd1306: exp_lut_q57 = 16'd1;
                12'd1307: exp_lut_q57 = 16'd1;
                12'd1308: exp_lut_q57 = 16'd1;
                12'd1309: exp_lut_q57 = 16'd1;
                12'd1310: exp_lut_q57 = 16'd1;
                12'd1311: exp_lut_q57 = 16'd1;
                12'd1312: exp_lut_q57 = 16'd1;
                12'd1313: exp_lut_q57 = 16'd1;
                12'd1314: exp_lut_q57 = 16'd1;
                12'd1315: exp_lut_q57 = 16'd1;
                12'd1316: exp_lut_q57 = 16'd1;
                12'd1317: exp_lut_q57 = 16'd1;
                12'd1318: exp_lut_q57 = 16'd1;
                12'd1319: exp_lut_q57 = 16'd1;
                12'd1320: exp_lut_q57 = 16'd1;
                12'd1321: exp_lut_q57 = 16'd1;
                12'd1322: exp_lut_q57 = 16'd1;
                12'd1323: exp_lut_q57 = 16'd1;
                12'd1324: exp_lut_q57 = 16'd1;
                12'd1325: exp_lut_q57 = 16'd1;
                12'd1326: exp_lut_q57 = 16'd1;
                12'd1327: exp_lut_q57 = 16'd1;
                12'd1328: exp_lut_q57 = 16'd1;
                12'd1329: exp_lut_q57 = 16'd1;
                12'd1330: exp_lut_q57 = 16'd1;
                12'd1331: exp_lut_q57 = 16'd1;
                12'd1332: exp_lut_q57 = 16'd1;
                12'd1333: exp_lut_q57 = 16'd1;
                12'd1334: exp_lut_q57 = 16'd1;
                12'd1335: exp_lut_q57 = 16'd1;
                12'd1336: exp_lut_q57 = 16'd1;
                12'd1337: exp_lut_q57 = 16'd1;
                12'd1338: exp_lut_q57 = 16'd1;
                12'd1339: exp_lut_q57 = 16'd1;
                12'd1340: exp_lut_q57 = 16'd1;
                12'd1341: exp_lut_q57 = 16'd1;
                12'd1342: exp_lut_q57 = 16'd1;
                12'd1343: exp_lut_q57 = 16'd1;
                12'd1344: exp_lut_q57 = 16'd1;
                12'd1345: exp_lut_q57 = 16'd1;
                12'd1346: exp_lut_q57 = 16'd1;
                12'd1347: exp_lut_q57 = 16'd1;
                12'd1348: exp_lut_q57 = 16'd1;
                12'd1349: exp_lut_q57 = 16'd1;
                12'd1350: exp_lut_q57 = 16'd1;
                12'd1351: exp_lut_q57 = 16'd1;
                12'd1352: exp_lut_q57 = 16'd1;
                12'd1353: exp_lut_q57 = 16'd1;
                12'd1354: exp_lut_q57 = 16'd1;
                12'd1355: exp_lut_q57 = 16'd1;
                12'd1356: exp_lut_q57 = 16'd1;
                12'd1357: exp_lut_q57 = 16'd1;
                12'd1358: exp_lut_q57 = 16'd1;
                12'd1359: exp_lut_q57 = 16'd1;
                12'd1360: exp_lut_q57 = 16'd1;
                12'd1361: exp_lut_q57 = 16'd1;
                12'd1362: exp_lut_q57 = 16'd1;
                12'd1363: exp_lut_q57 = 16'd1;
                12'd1364: exp_lut_q57 = 16'd1;
                12'd1365: exp_lut_q57 = 16'd1;
                12'd1366: exp_lut_q57 = 16'd1;
                12'd1367: exp_lut_q57 = 16'd1;
                12'd1368: exp_lut_q57 = 16'd1;
                12'd1369: exp_lut_q57 = 16'd1;
                12'd1370: exp_lut_q57 = 16'd1;
                12'd1371: exp_lut_q57 = 16'd1;
                12'd1372: exp_lut_q57 = 16'd1;
                12'd1373: exp_lut_q57 = 16'd1;
                12'd1374: exp_lut_q57 = 16'd1;
                12'd1375: exp_lut_q57 = 16'd1;
                12'd1376: exp_lut_q57 = 16'd1;
                12'd1377: exp_lut_q57 = 16'd1;
                12'd1378: exp_lut_q57 = 16'd1;
                12'd1379: exp_lut_q57 = 16'd1;
                12'd1380: exp_lut_q57 = 16'd1;
                12'd1381: exp_lut_q57 = 16'd1;
                12'd1382: exp_lut_q57 = 16'd1;
                12'd1383: exp_lut_q57 = 16'd1;
                12'd1384: exp_lut_q57 = 16'd1;
                12'd1385: exp_lut_q57 = 16'd1;
                12'd1386: exp_lut_q57 = 16'd1;
                12'd1387: exp_lut_q57 = 16'd1;
                12'd1388: exp_lut_q57 = 16'd1;
                12'd1389: exp_lut_q57 = 16'd1;
                12'd1390: exp_lut_q57 = 16'd1;
                12'd1391: exp_lut_q57 = 16'd1;
                12'd1392: exp_lut_q57 = 16'd1;
                12'd1393: exp_lut_q57 = 16'd1;
                12'd1394: exp_lut_q57 = 16'd1;
                12'd1395: exp_lut_q57 = 16'd1;
                12'd1396: exp_lut_q57 = 16'd1;
                12'd1397: exp_lut_q57 = 16'd1;
                12'd1398: exp_lut_q57 = 16'd1;
                12'd1399: exp_lut_q57 = 16'd1;
                12'd1400: exp_lut_q57 = 16'd1;
                12'd1401: exp_lut_q57 = 16'd1;
                12'd1402: exp_lut_q57 = 16'd1;
                12'd1403: exp_lut_q57 = 16'd1;
                12'd1404: exp_lut_q57 = 16'd1;
                12'd1405: exp_lut_q57 = 16'd1;
                12'd1406: exp_lut_q57 = 16'd1;
                12'd1407: exp_lut_q57 = 16'd1;
                12'd1408: exp_lut_q57 = 16'd1;
                12'd1409: exp_lut_q57 = 16'd1;
                12'd1410: exp_lut_q57 = 16'd1;
                12'd1411: exp_lut_q57 = 16'd1;
                12'd1412: exp_lut_q57 = 16'd1;
                12'd1413: exp_lut_q57 = 16'd1;
                12'd1414: exp_lut_q57 = 16'd1;
                12'd1415: exp_lut_q57 = 16'd1;
                12'd1416: exp_lut_q57 = 16'd1;
                12'd1417: exp_lut_q57 = 16'd1;
                12'd1418: exp_lut_q57 = 16'd1;
                12'd1419: exp_lut_q57 = 16'd1;
                12'd1420: exp_lut_q57 = 16'd0;
                12'd1421: exp_lut_q57 = 16'd0;
                12'd1422: exp_lut_q57 = 16'd0;
                12'd1423: exp_lut_q57 = 16'd0;
                12'd1424: exp_lut_q57 = 16'd0;
                12'd1425: exp_lut_q57 = 16'd0;
                12'd1426: exp_lut_q57 = 16'd0;
                12'd1427: exp_lut_q57 = 16'd0;
                12'd1428: exp_lut_q57 = 16'd0;
                12'd1429: exp_lut_q57 = 16'd0;
                12'd1430: exp_lut_q57 = 16'd0;
                12'd1431: exp_lut_q57 = 16'd0;
                12'd1432: exp_lut_q57 = 16'd0;
                12'd1433: exp_lut_q57 = 16'd0;
                12'd1434: exp_lut_q57 = 16'd0;
                12'd1435: exp_lut_q57 = 16'd0;
                12'd1436: exp_lut_q57 = 16'd0;
                12'd1437: exp_lut_q57 = 16'd0;
                12'd1438: exp_lut_q57 = 16'd0;
                12'd1439: exp_lut_q57 = 16'd0;
                12'd1440: exp_lut_q57 = 16'd0;
                12'd1441: exp_lut_q57 = 16'd0;
                12'd1442: exp_lut_q57 = 16'd0;
                12'd1443: exp_lut_q57 = 16'd0;
                12'd1444: exp_lut_q57 = 16'd0;
                12'd1445: exp_lut_q57 = 16'd0;
                12'd1446: exp_lut_q57 = 16'd0;
                12'd1447: exp_lut_q57 = 16'd0;
                12'd1448: exp_lut_q57 = 16'd0;
                12'd1449: exp_lut_q57 = 16'd0;
                12'd1450: exp_lut_q57 = 16'd0;
                12'd1451: exp_lut_q57 = 16'd0;
                12'd1452: exp_lut_q57 = 16'd0;
                12'd1453: exp_lut_q57 = 16'd0;
                12'd1454: exp_lut_q57 = 16'd0;
                12'd1455: exp_lut_q57 = 16'd0;
                12'd1456: exp_lut_q57 = 16'd0;
                12'd1457: exp_lut_q57 = 16'd0;
                12'd1458: exp_lut_q57 = 16'd0;
                12'd1459: exp_lut_q57 = 16'd0;
                12'd1460: exp_lut_q57 = 16'd0;
                12'd1461: exp_lut_q57 = 16'd0;
                12'd1462: exp_lut_q57 = 16'd0;
                12'd1463: exp_lut_q57 = 16'd0;
                12'd1464: exp_lut_q57 = 16'd0;
                12'd1465: exp_lut_q57 = 16'd0;
                12'd1466: exp_lut_q57 = 16'd0;
                12'd1467: exp_lut_q57 = 16'd0;
                12'd1468: exp_lut_q57 = 16'd0;
                12'd1469: exp_lut_q57 = 16'd0;
                12'd1470: exp_lut_q57 = 16'd0;
                12'd1471: exp_lut_q57 = 16'd0;
                12'd1472: exp_lut_q57 = 16'd0;
                12'd1473: exp_lut_q57 = 16'd0;
                12'd1474: exp_lut_q57 = 16'd0;
                12'd1475: exp_lut_q57 = 16'd0;
                12'd1476: exp_lut_q57 = 16'd0;
                12'd1477: exp_lut_q57 = 16'd0;
                12'd1478: exp_lut_q57 = 16'd0;
                12'd1479: exp_lut_q57 = 16'd0;
                12'd1480: exp_lut_q57 = 16'd0;
                12'd1481: exp_lut_q57 = 16'd0;
                12'd1482: exp_lut_q57 = 16'd0;
                12'd1483: exp_lut_q57 = 16'd0;
                12'd1484: exp_lut_q57 = 16'd0;
                12'd1485: exp_lut_q57 = 16'd0;
                12'd1486: exp_lut_q57 = 16'd0;
                12'd1487: exp_lut_q57 = 16'd0;
                12'd1488: exp_lut_q57 = 16'd0;
                12'd1489: exp_lut_q57 = 16'd0;
                12'd1490: exp_lut_q57 = 16'd0;
                12'd1491: exp_lut_q57 = 16'd0;
                12'd1492: exp_lut_q57 = 16'd0;
                12'd1493: exp_lut_q57 = 16'd0;
                12'd1494: exp_lut_q57 = 16'd0;
                12'd1495: exp_lut_q57 = 16'd0;
                12'd1496: exp_lut_q57 = 16'd0;
                12'd1497: exp_lut_q57 = 16'd0;
                12'd1498: exp_lut_q57 = 16'd0;
                12'd1499: exp_lut_q57 = 16'd0;
                12'd1500: exp_lut_q57 = 16'd0;
                12'd1501: exp_lut_q57 = 16'd0;
                12'd1502: exp_lut_q57 = 16'd0;
                12'd1503: exp_lut_q57 = 16'd0;
                12'd1504: exp_lut_q57 = 16'd0;
                12'd1505: exp_lut_q57 = 16'd0;
                12'd1506: exp_lut_q57 = 16'd0;
                12'd1507: exp_lut_q57 = 16'd0;
                12'd1508: exp_lut_q57 = 16'd0;
                12'd1509: exp_lut_q57 = 16'd0;
                12'd1510: exp_lut_q57 = 16'd0;
                12'd1511: exp_lut_q57 = 16'd0;
                12'd1512: exp_lut_q57 = 16'd0;
                12'd1513: exp_lut_q57 = 16'd0;
                12'd1514: exp_lut_q57 = 16'd0;
                12'd1515: exp_lut_q57 = 16'd0;
                12'd1516: exp_lut_q57 = 16'd0;
                12'd1517: exp_lut_q57 = 16'd0;
                12'd1518: exp_lut_q57 = 16'd0;
                12'd1519: exp_lut_q57 = 16'd0;
                12'd1520: exp_lut_q57 = 16'd0;
                12'd1521: exp_lut_q57 = 16'd0;
                12'd1522: exp_lut_q57 = 16'd0;
                12'd1523: exp_lut_q57 = 16'd0;
                12'd1524: exp_lut_q57 = 16'd0;
                12'd1525: exp_lut_q57 = 16'd0;
                12'd1526: exp_lut_q57 = 16'd0;
                12'd1527: exp_lut_q57 = 16'd0;
                12'd1528: exp_lut_q57 = 16'd0;
                12'd1529: exp_lut_q57 = 16'd0;
                12'd1530: exp_lut_q57 = 16'd0;
                12'd1531: exp_lut_q57 = 16'd0;
                12'd1532: exp_lut_q57 = 16'd0;
                12'd1533: exp_lut_q57 = 16'd0;
                12'd1534: exp_lut_q57 = 16'd0;
                12'd1535: exp_lut_q57 = 16'd0;
                12'd1536: exp_lut_q57 = 16'd0;
                default: exp_lut_q57 = 16'd0;
            endcase
        end
    endfunction

endmodule
