`timescale 1ns/1ps

module tb_Softmax_Unit;

    localparam int ROW_SIZE = 208;
    localparam int CLK_PERIOD_NS = 10;

    logic clk;
    logic rst_n;
    logic start;

    logic signed [31:0] score_row [0:207];
    logic [5:0] q_shift;
    logic [5:0] k_shift;
    logic [207:0] mask;

    logic signed [7:0] attention_row [0:207];
    logic done;

    integer i;
    integer error_count;

    Softmax_Unit dut (
        .clk           (clk),
        .rst_n         (rst_n),
        .start         (start),
        .score_row     (score_row),
        .q_shift       (q_shift),
        .k_shift       (k_shift),
        .mask          (mask),
        .attention_row (attention_row),
        .done          (done)
    );

    // 100 MHz clock
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD_NS/2) clk = ~clk;
    end

    task automatic clear_inputs;
        begin
            start   = 1'b0;
            q_shift = 6'd4;
            k_shift = 6'd4;
            mask    = '0;

            for (i = 0; i < ROW_SIZE; i = i + 1)
                score_row[i] = 32'sd0;
        end
    endtask

    task automatic reset_dut;
        begin
            rst_n = 1'b0;
            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic launch_and_wait;
        integer cycle_count;
        begin
            // start must be a one-cycle pulse
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;

            cycle_count = 0;
            while ((done !== 1'b1) && (cycle_count < 1500)) begin
                @(posedge clk);
                cycle_count = cycle_count + 1;
            end

            if (done !== 1'b1) begin
                $error("Timeout: done was not asserted within 1500 cycles");
                error_count = error_count + 1;
            end
            else begin
                $display("  done asserted after %0d cycles", cycle_count);
                // Sample registered outputs after the done edge.
                #1;
            end
        end
    endtask

    task automatic check_value(
        input integer position,
        input integer expected
    );
        integer actual;
        begin
            actual = $signed(attention_row[position]);

            if (actual !== expected) begin
                $error(
                    "Mismatch at position %0d: expected %0d, got %0d",
                    position, expected, actual
                );
                error_count = error_count + 1;
            end
        end
    endtask

    task automatic check_all_zero;
        begin
            for (i = 0; i < ROW_SIZE; i = i + 1) begin
                if ($signed(attention_row[i]) !== 0) begin
                    $error(
                        "Expected zero at position %0d, got %0d",
                        i, $signed(attention_row[i])
                    );
                    error_count = error_count + 1;
                end
            end
        end
    endtask

    initial begin
        error_count = 0;
        clear_inputs();
        reset_dut();

        // ============================================================
        // Test 1: Four equal valid scores
        //
        // Softmax([a,a,a,a]) = [0.25,0.25,0.25,0.25]
        // Signed Q0.7 output: round(0.25 * 128) = 32
        // ============================================================
        $display("\nTEST 1: four equal valid scores");

        clear_inputs();
        mask[0] = 1'b1;
        mask[1] = 1'b1;
        mask[2] = 1'b1;
        mask[3] = 1'b1;

        score_row[0] = 32'sd10000;
        score_row[1] = 32'sd10000;
        score_row[2] = 32'sd10000;
        score_row[3] = 32'sd10000;

        launch_and_wait();

        check_value(0, 32);
        check_value(1, 32);
        check_value(2, 32);
        check_value(3, 32);
        check_value(4, 0);

        // Internal datapath checks: all four scaled differences are zero.
        if ($signed(dut.scaled_score[0]) !== 0 ||
            $signed(dut.scaled_score[1]) !== 0 ||
            $signed(dut.scaled_score[2]) !== 0 ||
            $signed(dut.scaled_score[3]) !== 0) begin
            $error("TEST 1: scaled_score should be zero for equal scores");
            error_count = error_count + 1;
        end

        // ============================================================
        // Test 2: Known score differences 0, -1, -2, -3
        //
        // q_shift = 4, k_shift = 4:
        // real score = raw accumulator / 2^(4+4+3) = raw / 2048
        //
        // Raw inputs [0,-2048,-4096,-6144] therefore represent
        // scaled score differences [0,-1,-2,-3].
        //
        // exp LUT values are approximately:
        // [32768,12055,4435,1632]
        //
        // Expected signed Q0.7 probabilities:
        // [82,30,11,4]
        // ============================================================
        $display("\nTEST 2: known differences 0, -1, -2, -3");

        clear_inputs();
        mask[0] = 1'b1;
        mask[1] = 1'b1;
        mask[2] = 1'b1;
        mask[3] = 1'b1;

        score_row[0] =  32'sd0;
        score_row[1] = -32'sd2048;
        score_row[2] = -32'sd4096;
        score_row[3] = -32'sd6144;

        launch_and_wait();

        check_value(0, 82);
        check_value(1, 30);
        check_value(2, 11);
        check_value(3, 4);
        check_value(4, 0);

        // Q5.7 codes: 0, -128, -256, -384
        if ($signed(dut.scaled_score[0]) !== 0) begin
            $error("scaled_score[0]: expected 0, got %0d",
                   $signed(dut.scaled_score[0]));
            error_count = error_count + 1;
        end

        if ($signed(dut.scaled_score[1]) !== -128) begin
            $error("scaled_score[1]: expected -128, got %0d",
                   $signed(dut.scaled_score[1]));
            error_count = error_count + 1;
        end

        if ($signed(dut.scaled_score[2]) !== -256) begin
            $error("scaled_score[2]: expected -256, got %0d",
                   $signed(dut.scaled_score[2]));
            error_count = error_count + 1;
        end

        if ($signed(dut.scaled_score[3]) !== -384) begin
            $error("scaled_score[3]: expected -384, got %0d",
                   $signed(dut.scaled_score[3]));
            error_count = error_count + 1;
        end

        // ============================================================
        // Test 3: Mask behavior
        //
        // Positions 0 and 2 are valid and equal. Positions 1 and 3
        // contain large scores but are masked and must not affect max,
        // denominator, or output.
        //
        // Expected valid outputs: 0.5 * 128 = 64
        // ============================================================
        $display("\nTEST 3: masked entries must not affect softmax");

        clear_inputs();
        mask[0] = 1'b1;
        mask[2] = 1'b1;

        score_row[0] = 32'sd5000;
        score_row[1] = 32'sd2000000000; // masked
        score_row[2] = 32'sd5000;
        score_row[3] = 32'sd1900000000; // masked

        launch_and_wait();

        check_value(0, 64);
        check_value(1, 0);
        check_value(2, 64);
        check_value(3, 0);

        // ============================================================
        // Test 4: All positions masked
        //
        // The design should avoid division by zero and output all zero.
        // ============================================================
        $display("\nTEST 4: all positions masked");

        clear_inputs();

        for (i = 0; i < ROW_SIZE; i = i + 1)
            score_row[i] = $signed(i * 1234 - 50000);

        launch_and_wait();
        check_all_zero();

        // ============================================================
        // Final result
        // ============================================================
        if (error_count == 0)
            $display("\n========================================");
        else
            $display("\n========================================");

        if (error_count == 0)
            $display("ALL TESTS PASSED");
        else
            $display("TEST FAILED: %0d error(s)", error_count);

        $display("========================================\n");

        #20;
        $finish;
    end

endmodule
