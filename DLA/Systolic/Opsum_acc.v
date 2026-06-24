module Opsum_acc(
    //input en,
    input clk,
    input rst,

    input push_bias,
    input [127:0] bias_in,
    
    input push_row,
    input [319:0] psum_in, //peak: 16*20', 0 at [19:0], 16 at [319:300]
    input end_of_acc, // last pusm input, m-tile finish (accumulate all k) 

    output module_ready, //ready to accept input
    output opsum_valid,
    output [31:0] opsum
);

reg signed [39:0] skew_latch [6:0][6:0]; //col, latch cycle, 2 pack psum, for 8-1 cols

reg [2:0] cs, ns;
localparam IDLE = 2'b0, OP = 2'd1, OUT = 2'd2;
reg signed [31:0] psum_buffer [15:0][15:0]; //acc buff
reg [3:0] acc_row_ptr;
reg [1:0] row_rst_flag;
reg [3:0] out_row, out_col; //0~255
reg module_ready_, opsum_valid_;
assign module_ready = module_ready_;
assign opsum_valid = opsum_valid_;
assign opsum = psum_buffer[out_row][out_col];

always @(posedge clk) begin
    if(rst) begin
        cs <= IDLE;
    end
    else begin
        cs <= ns;
    end
end

integer i, j;
always @(posedge clk) begin //FSM
    case (cs)
        IDLE: begin
            //reset all
            //skew latch
            for(i=0; i<7; i = i+1) begin
                for(j=i; j<7; j=j+1) begin
                    skew_latch [i][j] <= 40'b0; //not in FSM
                end
            end

            //accumulator
            acc_row_ptr <= 4'd9;
            for(i = 0; i<16; i = i +1) begin
                for(j=0; j<16; j=j+1) begin
                    psum_buffer [i][j]<= 32'b0;
                end
            end
            out_row <= 4'd0;
            out_col <= 4'd0;
            row_rst_flag <= 3'b0;
        end
        OP: begin //m-tile, no need to clear
            // == 1, acc
            if(push_bias) begin
                for(i = 0; i<16; i= i+1) begin
                    psum_buffer[i][12] <= bias_in[31:0];
                    psum_buffer[i][13] <= bias_in[63:32];    
                    psum_buffer[i][14] <= bias_in[95:64];    
                    psum_buffer[i][15] <= bias_in[127:96];
                end
                for(j=8; j>=0; j = j-4) begin
                    for(i = 0; i<16; i= i+1) begin
                        psum_buffer[i][j] <= psum_buffer[i][j+4];
                        psum_buffer[i][j+1] <= psum_buffer[i][j+4+1];    
                        psum_buffer[i][j+2] <= psum_buffer[i][j+4+2];    
                        psum_buffer[i][j+3] <= psum_buffer[i][j+4+3];
                    end
                end
            end
            else if(push_row) begin
                //skew latch
                for(i = 0; i<7; i= i+1) begin //input
                    skew_latch[i][i][19:0] <= psum_in[40*i+:20];
                    skew_latch[i][i][39:20] <= psum_in[40*(i)+20+:20];
                end
                for(i=0; i<6; i = i+1) begin //shift 1, no need to clear
                    for(j=i; j<6; j=j+1) begin
                        skew_latch [i][j+1] <= skew_latch[i][j];
                    end
                end
                
                if((&row_rst_flag) & (&acc_row_ptr)) begin
                    row_rst_flag <= 2'b0;
                    acc_row_ptr <= 4'd9;
                end
                else begin
                    acc_row_ptr <= acc_row_ptr + 4'd1;
                    row_rst_flag[0] <= (acc_row_ptr==4'd8) | row_rst_flag[0]; //all 0 => set flag
                    row_rst_flag[1] <= row_rst_flag[0] & (acc_row_ptr==4'd8) |row_rst_flag[1];
                end
                

                //accumulator
                psum_buffer[acc_row_ptr][15] <= psum_buffer[acc_row_ptr][15] + $signed(psum_in[319:300]); //direct from input
                psum_buffer[acc_row_ptr][14] <= psum_buffer[acc_row_ptr][14] + $signed(psum_in[299:280]); //direct from input

                for(i = 0; i<7; i=i+1) begin //column
                    psum_buffer[acc_row_ptr][i*2+1] <= psum_buffer[acc_row_ptr][i*2+1] + $signed(skew_latch[i][6][39:20]);
                    psum_buffer[acc_row_ptr][i*2] <= psum_buffer[acc_row_ptr][i*2] + $signed(skew_latch[i][6][19:0]);
                end
            end
        end
        OUT: begin
            out_col <= out_col + 4'd1;
            out_row <= out_row + &(out_col); //out_col==15 => add out_row
        end
    endcase
end

//state & out
always @(*) begin
    case (cs)
        IDLE: begin
            ns = OP;
            opsum_valid_ = 1'b0;
            module_ready_ = 1'b0;
        end
        OP: begin
            opsum_valid_ = 1'b0;
            module_ready_ = 1'b1;
            if(end_of_acc) //last row pushed
                ns = OUT;
            else
                ns = OP;
        end
        OUT: begin
            module_ready_ = 1'b0;
            opsum_valid_ = 1'b1;
            if(&(out_row & out_col)) begin //both == 15
                ns = IDLE;
            end
            else begin
                ns = OUT;
            end
        end
        default: begin
            module_ready_ = 1'b0;
            ns = IDLE;
            opsum_valid_ = 1'b0;

        end 
    endcase
end

endmodule
