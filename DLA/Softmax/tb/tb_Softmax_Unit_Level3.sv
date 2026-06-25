`timescale 1ns/1ps
`include "Softmax_Unit.sv"
// Self-checking testbench for Softmax_Unit_0618_1324.sv
// Put the referenced HEX files and exp_lut_10bit_Q1_15_range12.hex
// in XSim's working directory, or replace the filenames with absolute paths.

module tb_Softmax_Unit_Level3;
    localparam int ROW_SIZE=1182;
    localparam int TIMEOUT_CYCLES=1500;

    logic clk=0, rst_n=0, start=0;
    logic signed [31:0] score_row [0:207];
    logic [5:0] q_shift=6'd4, k_shift=6'd4;
    logic [207:0] mask='0;
    logic signed [7:0] attention_row [0:207];
    logic done;

    logic signed [31:0] all_score [0:ROW_SIZE*208-1];
    logic [7:0] all_mask [0:ROW_SIZE*208-1];
    logic signed [31:0] all_max [0:ROW_SIZE-1];
    logic [31:0] all_sum [0:ROW_SIZE-1];
    logic signed [7:0] all_attn [0:ROW_SIZE*208-1];

    integer r,k,base,cycles,total_errors,row_errors;

    always #5 clk=~clk;
    Softmax_Unit dut(
        .clk(clk),.rst_n(rst_n),.start(start),
        .score_row(score_row),.q_shift(q_shift),.k_shift(k_shift),
        .mask(mask),.attention_row(attention_row),.done(done)
    );

    initial begin
        $readmemh("../hex/level3/score_int32_padded.hex",all_score);
        $readmemh("../hex/level3/mask_padded.hex",all_mask);
        $readmemh("../hex/level3/max_score_shifted_int32.hex",all_max);
        $readmemh("../hex/level3/exp_sum_uint32.hex",all_sum);
        $readmemh("../hex/level3/attention_q07_padded.hex",all_attn);
        $readmemh("../hex/exp_lut_10bit_Q1_15_range12.hex", dut.exp_lut_rom);

        repeat(4) @(posedge clk);
        rst_n=1;
        repeat(2) @(posedge clk);
        total_errors=0;

        for(r=0;r<ROW_SIZE;r=r+1) begin
            base=r*208;
            for(k=0;k<208;k=k+1) begin
                score_row[k]=all_score[base+k];
                mask[k]=all_mask[base+k][0];
            end

            @(negedge clk); start=1;
            @(negedge clk); start=0;

            cycles=0;
            while(done!==1'b1 && cycles<TIMEOUT_CYCLES) begin
                @(posedge clk); #1; cycles=cycles+1;
            end
            if(done!==1'b1) $fatal(1,"Timeout at row %0d",r);

            row_errors=0;
            if($signed(dut.max_score)!==$signed(all_max[r])) begin
                if(total_errors<20) $error("MAX row=%0d exp=%0d act=%0d",r,$signed(all_max[r]),$signed(dut.max_score));
                row_errors=row_errors+1;
            end
            if(dut.exp_sum!==all_sum[r]) begin
                if(total_errors<20) $error("SUM row=%0d exp=%0d act=%0d",r,all_sum[r],dut.exp_sum);
                row_errors=row_errors+1;
            end
            for(k=0;k<208;k=k+1)
                if($signed(attention_row[k])!==$signed(all_attn[base+k])) begin
                    if(total_errors<20) $error("ATTN row=%0d key=%0d exp=%0d act=%0d",r,k,$signed(all_attn[base+k]),$signed(attention_row[k]));
                    row_errors=row_errors+1;
                end
            total_errors=total_errors+row_errors;

            // done is a one-cycle pulse; wait for DUT to return to idle.
            @(posedge clk); #1;
        end

        if(total_errors==0)
            $display("LEVEL 3 PASS: all %0d rows matched.",ROW_SIZE);
        else
            $fatal(1,"LEVEL 3 FAIL: %0d mismatch(es).",total_errors);
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
