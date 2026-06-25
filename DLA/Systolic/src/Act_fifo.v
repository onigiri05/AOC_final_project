module Act_fifo (
    input clk,
    input rst,

    input push_row,                 // write 1 row, when act valid && act ready
    input [127:0] act_row_in,

    input pop_row,                  // output 1 skewed row
    output [127:0] act_row_out
);

reg [7:0] act_r [15:0][15:0]; //16 row
reg [7:0] skew_latch [14:0][14:0]; //row, latch cycle, without row 0

assign act_row_out[7:0] = act_r[0][0];
genvar k;
generate
    for(k=1; k<16; k=k+1) begin
        assign act_row_out[k*8+:8] = skew_latch[0][k-1];    
    end
endgenerate

integer i, j;
always @(posedge clk) begin
    if(rst) begin
        for(i = 0; i<16; i = i+1) begin
            for(j = 0; j<16; j= j+1) begin
                act_r[i][j] <= 8'b0;
            end
        end
        for(i = 0; i<15; i = i+1) begin
            for(j=0; j<=i; j=j+1) begin
                skew_latch[j][i] <= 8'b0;
            end
        end
    end
    else begin
        if(push_row) begin
            for(i = 0; i<16; i = i+1) begin                 //row 15: input
                act_r[15][i] <= act_row_in[8*i+:8];
            end

            for(i=14; i>=0; i=i-1) begin                    //shift 1 row (row 15 => row 0)
                for(j=0; j<16; j=j+1) begin
                    act_r [i][j] <= act_r[i+1][j];
                end
            end
        end
        else if(pop_row) begin
            for(i=0; i<16; i=i+1) begin                     //row 15 feed 0
                act_r [15][i] <= 8'b0;
            end
            for(i=14; i>=0; i=i-1) begin                    //shift 1 row (row 15 => row 0)
                for(j=0; j<16; j=j+1) begin
                    act_r [i][j] <= act_r[i+1][j];
                end
            end

            for(i=0; i<15; i= i+1) begin                    //row 0 push to skew buffer
                skew_latch [i][i] <= act_r[0][i+1];
            end
            for(i = 0; i<15; i = i+1) begin                 // shift 1 row, row i => row0
                for(j=0; j<i; j=j+1) begin
                    skew_latch [j][i] <= skew_latch[j+1][i];
                end
            end
        end
    end
end

endmodule
