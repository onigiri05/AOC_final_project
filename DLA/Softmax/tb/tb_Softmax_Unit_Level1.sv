`timescale 1ns/1ps
`include "Softmax_Unit.sv"
// Self-checking testbench for Softmax_Unit_0618_1324.sv
// Put the referenced HEX files and exp_lut_10bit_Q1_15_range12.hex
// in XSim's working directory, or replace the filenames with absolute paths.

module tb_Softmax_Unit_Level1;
    localparam int ROW_SIZE = 208;
    localparam int TIMEOUT_CYCLES = 1500;

    logic clk=0, rst_n=0, start=0;
    logic signed [31:0] score_row [0:207];
    logic [5:0] q_shift=6'd4, k_shift=6'd4;
    logic [207:0] mask='0;
    logic signed [7:0] attention_row [0:207];
    logic done;

    logic signed [31:0] g_score [0:207];
    logic [7:0] g_mask [0:207];
    logic signed [31:0] g_shifted [0:207];
    logic signed [31:0] g_max [0:0];
    logic signed [31:0] g_diff [0:207];
    logic [15:0] g_lut_idx [0:207];
    logic [15:0] g_exp [0:207];
    logic [31:0] g_sum [0:0];
    logic signed [7:0] g_attn [0:207];

    integer i, errors, cycles;
    always #5 clk = ~clk;

    Softmax_Unit dut(
        .clk(clk), .rst_n(rst_n), .start(start),
        .score_row(score_row), .q_shift(q_shift), .k_shift(k_shift),
        .mask(mask), .attention_row(attention_row), .done(done)
    );

    initial begin
        $readmemh("../hex/level1/score_int32_padded.hex", g_score);
        $readmemh("../hex/level1/mask.hex", g_mask);
        $readmemh("../hex/level1/scores_shifted_int32_padded.hex", g_shifted);
        $readmemh("../hex/level1/max_score_shifted_int32.hex", g_max);
        $readmemh("../hex/level1/int_diff_int32_padded.hex", g_diff);
        $readmemh("../hex/level1/lut_index_uint10_padded.hex", g_lut_idx);
        $readmemh("../hex/level1/exp_uq15_padded.hex", g_exp);
        $readmemh("../hex/level1/exp_sum_uint32.hex", g_sum);
        $readmemh("../hex/level1/attention_q07_padded.hex", g_attn);
        $readmemh("../hex/exp_lut_10bit_Q1_15_range12.hex", dut.exp_lut_rom);

        #1;
        for (i=0;i<208;i=i+1) begin
            score_row[i]=g_score[i];
            mask[i]=g_mask[i][0];
        end

        repeat(4) @(posedge clk);
        rst_n=1;
        repeat(2) @(posedge clk);
        @(negedge clk); start=1;
        @(negedge clk); start=0;

        // State 2�Gshift stage �w�g����
wait(dut.state == 3'd2);
#1;

errors = 0;
for (i = 0; i < 208; i = i + 1) begin
    if ($signed(dut.scaled_score[i]) !==
        $signed(g_shifted[i])) begin

        if (errors < 10)
            $error(
                "SHIFTED i=%0d exp=%0d act=%0d",
                i,
                $signed(g_shifted[i]),
                $signed(dut.scaled_score[i])
            );

        errors = errors + 1;
    end
end

$display(
    "Level1 shifted-score mismatches: %0d",
    errors
);

// State 3�Gmax-search stage �w�g����
wait(dut.state == 3'd3);
#1;

if ($signed(dut.max_score) !== $signed(g_max[0])) begin
    $error(
        "ROW_MAX expected=%0d actual=%0d",
        $signed(g_max[0]),
        $signed(dut.max_score)
    );
end
else begin
    $display(
        "Level1 row max matches: %0d",
        $signed(dut.max_score)
    );
end
        $display("Level1 shifted-score mismatches: %0d",errors);

        wait(dut.state==3'd4); #1;
        errors=0;
        for (i=0;i<208;i=i+1)
            if (dut.exp_value[i]!==g_exp[i]) begin
                if (errors<10) $error("EXP i=%0d exp=%0d act=%0d",i,g_exp[i],dut.exp_value[i]);
                errors=errors+1;
            end
        if (dut.exp_sum!==g_sum[0]) begin
            $error("EXP_SUM expected=%0d actual=%0d",g_sum[0],dut.exp_sum);
            errors=errors+1;
        end
        $display("Level1 exp-stage mismatches: %0d",errors);

        cycles=0;
        while(done!==1'b1 && cycles<TIMEOUT_CYCLES) begin
            @(posedge clk); #1; cycles=cycles+1;
        end
        if(done!==1'b1) $fatal(1,"Timeout waiting for done");

        errors=0;
        for (i=0;i<208;i=i+1)
            if ($signed(attention_row[i])!==$signed(g_attn[i])) begin
                if(errors<20) $error("ATTN i=%0d exp=%0d act=%0d",i,$signed(g_attn[i]),$signed(attention_row[i]));
                errors=errors+1;
            end

        if(errors==0) $display("LEVEL 1 PASS: all 208 attention values match.");
        else $fatal(1,"LEVEL 1 FAIL: %0d final mismatches",errors);
        #20; $finish;
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
