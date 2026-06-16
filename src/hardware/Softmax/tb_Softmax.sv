`timescale 1ns/1ps

module tb_Softmax_Unit;

    localparam integer ROW_SIZE = 208;
    localparam integer CLK_PERIOD = 10;
    localparam integer TIMEOUT_CYCLES = 2000;

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
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Clear all DUT inputs before constructing each test case.
    task automatic clear_inputs;
        integer n;
        begin
            start   = 1'b0;
            q_shift = 6'd0;
            k_shift = 6'd0;
            mask    = '0;

            for (n = 0; n < ROW_SIZE; n = n + 1)
                score_row[n] = 32'sd0;
        end
    endtask

    // start must stay high for exactly one rising clock edge.
    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
        end
    endtask

    // Wait for the one-cycle done pulse and stop the simulation on timeout.
    task automatic wait_for_done;
        integer cycles;
        begin
            cycles = 0;

            while ((done !== 1'b1) && (cycles < TIMEOUT_CYCLES)) begin
                @(posedge clk);
                cycles = cycles + 1;
            end

            if (cycles >= TIMEOUT_CYCLES) begin
                $fatal(1, "Timeout: done was not asserted within %0d cycles", TIMEOUT_CYCLES);
            end

            // Sample outputs after nonblocking assignments have settled.
            #1;
            $display("  done received after %0d cycles", cycles);
        end
    endtask

    task automatic check_attention_exact (
        input integer position,
        input integer expected,
        input string  test_name
    );
        integer actual;
        begin
            actual = $signed(attention_row[position]);

            if (actual !== expected) begin
                $error("%s: attention_row[%0d] = %0d, expected %0d",
                       test_name, position, actual, expected);
                error_count = error_count + 1;
            end
        end
    endtask

    task automatic check_all_zero (
        input string test_name
    );
        integer n;
        begin
            for (n = 0; n < ROW_SIZE; n = n + 1)
                check_attention_exact(n, 0, test_name);
        end
    endtask

    initial begin
        error_count = 0;
        clear_inputs();

        // Active-low asynchronous reset.
        rst_n = 1'b0;
        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        // ========================================================
        // Test 1: four equal valid scores
        // scaled_score = [0, 0, 0, 0]
        // softmax = [0.25, 0.25, 0.25, 0.25]
        // Q0.7 = [32, 32, 32, 32]
        // ========================================================
        $display("TEST 1: four equal scores");
        clear_inputs();

        q_shift = 6'd0;
        k_shift = 6'd0;
        mask[3:0] = 4'b1111;

        score_row[0] = 32'sd0;
        score_row[1] = 32'sd0;
        score_row[2] = 32'sd0;
        score_row[3] = 32'sd0;

        pulse_start();
        wait_for_done();

        check_attention_exact(0, 32, "TEST 1");
        check_attention_exact(1, 32, "TEST 1");
        check_attention_exact(2, 32, "TEST 1");
        check_attention_exact(3, 32, "TEST 1");

        for (i = 4; i < ROW_SIZE; i = i + 1)
            check_attention_exact(i, 0, "TEST 1 masked output");

        // ========================================================
        // Test 2: four different scores
        // q_shift + k_shift + 3 = 3
        // raw score    = [0, -8, -16, -24]
        // scaled score = [0, -1, -2, -3]
        // expected Q0.7 approximately [82, 30, 11, 4]
        // ========================================================
        $display("TEST 2: integer exponential mapping");
        clear_inputs();

        q_shift = 6'd0;
        k_shift = 6'd0;
        mask[3:0] = 4'b1111;

        score_row[0] =  32'sd0;
        score_row[1] = -32'sd8;
        score_row[2] = -32'sd16;
        score_row[3] = -32'sd24;

        pulse_start();
        wait_for_done();

        check_attention_exact(0, 82, "TEST 2");
        check_attention_exact(1, 30, "TEST 2");
        check_attention_exact(2, 11, "TEST 2");
        check_attention_exact(3,  4, "TEST 2");

        // ========================================================
        // Test 3: masked large score must not affect max or sum
        // Only positions 0 and 1 are valid, both scaled scores are 0.
        // Position 2 contains a large value but is masked.
        // expected Q0.7 = [64, 64, 0]
        // ========================================================
        $display("TEST 3: mask excludes a large score");
        clear_inputs();

        q_shift = 6'd0;
        k_shift = 6'd0;
        mask[0] = 1'b1;
        mask[1] = 1'b1;
        mask[2] = 1'b0;

        score_row[0] = 32'sd0;
        score_row[1] = 32'sd0;
        score_row[2] = 32'sd8000;

        pulse_start();
        wait_for_done();

        check_attention_exact(0, 64, "TEST 3");
        check_attention_exact(1, 64, "TEST 3");
        check_attention_exact(2,  0, "TEST 3");

        // ========================================================
        // Test 4: all entries are masked
        // has_valid remains zero, exp_sum remains zero, and every
        // output must be zero without a divide-by-zero failure.
        // ========================================================
        $display("TEST 4: complete row masked");
        clear_inputs();

        for (i = 0; i < ROW_SIZE; i = i + 1)
            score_row[i] = $signed(i * 100 - 10000);

        pulse_start();
        wait_for_done();
        check_all_zero("TEST 4");

        // ========================================================
        // Test 5: total_shift >= 32 special branch
        // q_shift + k_shift + 3 = 43
        // positive raw score -> scaled score 0
        // negative raw score -> scaled score -1
        // softmax Q0.7 approximately [94, 34]
        // ========================================================
        $display("TEST 5: total_shift greater than or equal to 32");
        clear_inputs();

        q_shift = 6'd20;
        k_shift = 6'd20;
        mask[1:0] = 2'b11;

        score_row[0] =  32'sd100;
        score_row[1] = -32'sd100;

        pulse_start();
        wait_for_done();

        check_attention_exact(0, 94, "TEST 5");
        check_attention_exact(1, 34, "TEST 5");

        // ========================================================
        // Test 6: one valid entry
        // Softmax is mathematically 1.0. Q0.7 signed INT8 cannot
        // represent +128, so the result must saturate to 127.
        // ========================================================
        $display("TEST 6: one valid token and Q0.7 saturation");
        clear_inputs();

        q_shift = 6'd0;
        k_shift = 6'd0;
        mask[17] = 1'b1;
        score_row[17] = 32'sd12345;

        pulse_start();
        wait_for_done();

        check_attention_exact(17, 127, "TEST 6");

        for (i = 0; i < ROW_SIZE; i = i + 1) begin
            if (i != 17)
                check_attention_exact(i, 0, "TEST 6 masked output");
        end

        if (error_count == 0) begin
            $display("============================================================");
            $display("PASS: all Softmax_Unit test cases passed.");
            $display("============================================================");
        end
        else begin
            $fatal(1, "FAIL: %0d Softmax_Unit checks failed.", error_count);
        end

        repeat (5) @(posedge clk);
        $finish;
    end

endmodule
