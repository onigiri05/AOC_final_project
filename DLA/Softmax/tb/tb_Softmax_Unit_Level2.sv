`timescale 1ns/1ps
`include "Softmax_Unit.sv"

module tb_Softmax_Unit_Level2;

    localparam int NUM_ROWS       = 197;
    localparam int RTL_ROW_SIZE   = 208;
    localparam int TIMEOUT_CYCLES = 1500;

    localparam logic [5:0] Q_SHIFT_VALUE = 6'd4;
    localparam logic [5:0] K_SHIFT_VALUE = 6'd4;

    logic clk;
    logic rst_n;
    logic start;

    logic signed [31:0] score_row [0:RTL_ROW_SIZE-1];
    logic [5:0] q_shift;
    logic [5:0] k_shift;
    logic [RTL_ROW_SIZE-1:0] mask;

    logic signed [7:0] attention_row [0:RTL_ROW_SIZE-1];
    logic done;

    // Flattened Level 2 golden memories.
    // Address = query_row * 208 + key_index
    logic signed [31:0] g_score
        [0:NUM_ROWS*RTL_ROW_SIZE-1];

    logic signed [31:0] g_shifted
        [0:NUM_ROWS*RTL_ROW_SIZE-1];

    logic [7:0] g_mask
        [0:NUM_ROWS*RTL_ROW_SIZE-1];

    logic signed [31:0] g_max
        [0:NUM_ROWS-1];

    logic [15:0] g_exp
        [0:NUM_ROWS*RTL_ROW_SIZE-1];

    logic [31:0] g_sum
        [0:NUM_ROWS-1];

    logic signed [7:0] g_attn
        [0:NUM_ROWS*RTL_ROW_SIZE-1];

    integer row_idx;
    integer key_idx;
    integer base_addr;
    integer cycle_count;

    integer shifted_errors;
    integer max_errors;
    integer exp_errors;
    integer sum_errors;
    integer attn_errors;
    integer total_errors;
    integer shown_errors;

    // ------------------------------------------------------------------------
    // DUT
    // ------------------------------------------------------------------------
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
        forever #5 clk = ~clk;
    end

    // ------------------------------------------------------------------------
    // Load Level 2 golden data
    // ------------------------------------------------------------------------
    initial begin
        $readmemh("../hex/level2/score_int32_padded.hex", g_score);
        $readmemh("../hex/level2/mask_padded.hex", g_mask);
        $readmemh("../hex/level2/scores_shifted_int32_padded.hex", g_shifted);
        $readmemh("../hex/level2/max_score_shifted_int32.hex", g_max);
        $readmemh("../hex/level2/exp_uq15_padded.hex", g_exp);
        $readmemh("../hex/level2/exp_sum_uint32.hex", g_sum);
        $readmemh("../hex/level2/attention_q07_padded.hex", g_attn);
        $readmemh("../hex/exp_lut_10bit_Q1_15_range12.hex", dut.exp_lut_rom);
    end

    // ------------------------------------------------------------------------
    // Fail immediately if a HEX file was not loaded.
    // Check both the first and last elements to catch truncated files.
    // ------------------------------------------------------------------------
    task automatic check_golden_loaded;
        begin
            if ($isunknown(g_score[0]) ||
                $isunknown(g_score[NUM_ROWS*RTL_ROW_SIZE-1]))
                $fatal(1, "score_int32_padded.hex was not loaded completely.");

            if ($isunknown(g_shifted[0]) ||
                $isunknown(g_shifted[NUM_ROWS*RTL_ROW_SIZE-1]))
                $fatal(1, "scores_shifted_int32_padded.hex was not loaded completely.");

            if ($isunknown(g_mask[0]) ||
                $isunknown(g_mask[NUM_ROWS*RTL_ROW_SIZE-1]))
                $fatal(1, "mask_padded.hex was not loaded completely.");

            if ($isunknown(g_max[0]) ||
                $isunknown(g_max[NUM_ROWS-1]))
                $fatal(1, "max_score_shifted_int32.hex was not loaded completely.");

            if ($isunknown(g_exp[0]) ||
                $isunknown(g_exp[NUM_ROWS*RTL_ROW_SIZE-1]))
                $fatal(1, "exp_uq15_padded.hex was not loaded completely.");

            if ($isunknown(g_sum[0]) ||
                $isunknown(g_sum[NUM_ROWS-1]))
                $fatal(1, "exp_sum_uint32.hex was not loaded completely.");

            if ($isunknown(g_attn[0]) ||
                $isunknown(g_attn[NUM_ROWS*RTL_ROW_SIZE-1]))
                $fatal(1, "attention_q07_padded.hex was not loaded completely.");
        end
    endtask

    task automatic reset_dut;
        begin
            start   = 1'b0;
            rst_n   = 1'b0;
            q_shift = Q_SHIFT_VALUE;
            k_shift = K_SHIFT_VALUE;
            mask    = '0;

            repeat (4) @(posedge clk);
            rst_n = 1'b1;
            repeat (2) @(posedge clk);
        end
    endtask

    task automatic load_row(input integer row_number);
        begin
            base_addr = row_number * RTL_ROW_SIZE;

            for (key_idx = 0;
                 key_idx < RTL_ROW_SIZE;
                 key_idx = key_idx + 1) begin

                score_row[key_idx] = g_score[base_addr + key_idx];
                mask[key_idx]      = g_mask[base_addr + key_idx][0];
            end
        end
    endtask

    task automatic pulse_start;
        begin
            @(negedge clk);
            start = 1'b1;

            @(negedge clk);
            start = 1'b0;
        end
    endtask

    task automatic wait_for_state(
        input logic [2:0] expected_state,
        input integer row_number
    );
        begin
            cycle_count = 0;

            while ((dut.state !== expected_state) &&
                   (cycle_count < TIMEOUT_CYCLES)) begin
                @(posedge clk);
                #1;
                cycle_count = cycle_count + 1;
            end

            if (dut.state !== expected_state)
                $fatal(
                    1,
                    "Timeout waiting for state %0d at row %0d.",
                    expected_state,
                    row_number
                );
        end
    endtask

    task automatic wait_for_done(input integer row_number);
        begin
            cycle_count = 0;

            while ((done !== 1'b1) &&
                   (cycle_count < TIMEOUT_CYCLES)) begin
                @(posedge clk);
                #1;
                cycle_count = cycle_count + 1;
            end

            if (done !== 1'b1)
                $fatal(
                    1,
                    "Timeout waiting for done at row %0d.",
                    row_number
                );
        end
    endtask

    // ------------------------------------------------------------------------
    // Main regression test
    // ------------------------------------------------------------------------
    initial begin
        start       = 1'b0;
        rst_n       = 1'b0;
        q_shift     = Q_SHIFT_VALUE;
        k_shift     = K_SHIFT_VALUE;
        mask        = '0;

        shifted_errors = 0;
        max_errors     = 0;
        exp_errors     = 0;
        sum_errors     = 0;
        attn_errors    = 0;
        total_errors   = 0;
        shown_errors   = 0;

        for (key_idx = 0;
             key_idx < RTL_ROW_SIZE;
             key_idx = key_idx + 1)
            score_row[key_idx] = '0;

        // Allow $readmemh initial block to complete.
        #1;
        check_golden_loaded();
        reset_dut();

        $display("============================================================");
        $display("Starting Level 2 validation: head 0, 197 query rows");
        $display("q_shift=%0d, k_shift=%0d", q_shift, k_shift);
        $display("============================================================");

        for (row_idx = 0;
             row_idx < NUM_ROWS;
             row_idx = row_idx + 1) begin

            load_row(row_idx);
            pulse_start();

            // State 2 means all shifted scores are ready.
            wait_for_state(3'd2, row_idx);

            base_addr = row_idx * RTL_ROW_SIZE;

            for (key_idx = 0;
                 key_idx < RTL_ROW_SIZE;
                 key_idx = key_idx + 1) begin

                if ($signed(dut.scaled_score[key_idx]) !==
                    $signed(g_shifted[base_addr + key_idx])) begin

                    shifted_errors = shifted_errors + 1;
                    total_errors   = total_errors + 1;

                    if (shown_errors < 20) begin
                        $error(
                            "SHIFTED row=%0d key=%0d expected=%0d actual=%0d",
                            row_idx,
                            key_idx,
                            $signed(g_shifted[base_addr + key_idx]),
                            $signed(dut.scaled_score[key_idx])
                        );
                        shown_errors = shown_errors + 1;
                    end
                end
            end

            // State 3 means row-max search has completed.
            wait_for_state(3'd3, row_idx);

            if ($signed(dut.max_score) !== $signed(g_max[row_idx])) begin
                max_errors   = max_errors + 1;
                total_errors = total_errors + 1;

                if (shown_errors < 20) begin
                    $error(
                        "ROW_MAX row=%0d expected=%0d actual=%0d",
                        row_idx,
                        $signed(g_max[row_idx]),
                        $signed(dut.max_score)
                    );
                    shown_errors = shown_errors + 1;
                end
            end

            // State 4 means exp values and exp sum are ready.
            wait_for_state(3'd4, row_idx);

            for (key_idx = 0;
                 key_idx < RTL_ROW_SIZE;
                 key_idx = key_idx + 1) begin

                if (dut.exp_value[key_idx] !==
                    g_exp[base_addr + key_idx]) begin

                    exp_errors   = exp_errors + 1;
                    total_errors = total_errors + 1;

                    if (shown_errors < 20) begin
                        $error(
                            "EXP row=%0d key=%0d expected=%0d actual=%0d",
                            row_idx,
                            key_idx,
                            g_exp[base_addr + key_idx],
                            dut.exp_value[key_idx]
                        );
                        shown_errors = shown_errors + 1;
                    end
                end
            end

            if (dut.exp_sum !== g_sum[row_idx]) begin
                sum_errors   = sum_errors + 1;
                total_errors = total_errors + 1;

                if (shown_errors < 20) begin
                    $error(
                        "EXP_SUM row=%0d expected=%0d actual=%0d",
                        row_idx,
                        g_sum[row_idx],
                        dut.exp_sum
                    );
                    shown_errors = shown_errors + 1;
                end
            end

            wait_for_done(row_idx);

            for (key_idx = 0;
                 key_idx < RTL_ROW_SIZE;
                 key_idx = key_idx + 1) begin

                if ($signed(attention_row[key_idx]) !==
                    $signed(g_attn[base_addr + key_idx])) begin

                    attn_errors  = attn_errors + 1;
                    total_errors = total_errors + 1;

                    if (shown_errors < 20) begin
                        $error(
                            "ATTN row=%0d key=%0d expected=%0d actual=%0d",
                            row_idx,
                            key_idx,
                            $signed(g_attn[base_addr + key_idx]),
                            $signed(attention_row[key_idx])
                        );
                        shown_errors = shown_errors + 1;
                    end
                end
            end

            if ((row_idx % 20) == 0)
                $display(
                    "Progress: completed row %0d / %0d",
                    row_idx,
                    NUM_ROWS-1
                );

            // done is one cycle. Wait until DUT returns to idle before
            // applying the next start pulse.
            @(posedge clk);
            #1;

            while (dut.state !== 3'd0) begin
                @(posedge clk);
                #1;
            end
        end

        $display("============================================================");
        $display("Level 2 validation summary");
        $display("  shifted-score errors : %0d", shifted_errors);
        $display("  row-max errors       : %0d", max_errors);
        $display("  exp-value errors     : %0d", exp_errors);
        $display("  exp-sum errors       : %0d", sum_errors);
        $display("  attention errors     : %0d", attn_errors);
        $display("============================================================");

        if (total_errors == 0) begin
            $display(
                "LEVEL 2 PASS: all 197 rows and all internal stages match."
            );
        end
        else begin
            $fatal(
                1,
                "LEVEL 2 FAIL: total mismatch count = %0d",
                total_errors
            );
        end

        #20;
        $finish;
    end

initial begin
    `ifdef FSDB
    $fsdbDumpfile("top.fsdb"); 
    $fsdbDumpvars(0); //all signal
    `elsif FSDB_ALL
    $fsdbDumpfile("top.fsdb");
    $fsdbDumpvars(0, "+mda"); //expand memory/ array
    `endif
end

endmodule
