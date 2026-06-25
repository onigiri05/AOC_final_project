// `include "../src/PE_pack.v"
// `include "../src/Act_fifo.v"
// `include "../src/Opsum_acc.v"

module Systolic(
    input clk,
    input rst_n,
    
    // ctrl & config for each m_tile
    input en,
    output reg module_ready,      // module able to use
    input [16:0] act_base_addr, // BRAM total: 4096 * 140 B, / 8B per word, 17' addr
    input [16:0] w_base_addr,
    input [16:0] bias_base_addr,
    input [6:0] k_tile_cnt, //at most 96, at FC2

    // BRAM(SRAM) input data, for each k-tile
    output act_bram_rd_en,
    output w_bram_rd_en,
    output bias_bram_rd_en,
    output [16:0] act_bram_addr, 
    output [16:0] w_bram_addr,
    output [16:0] bias_bram_addr,
    input w_bram_valid,
    input [31:0] act_bram_data,
    input [31:0] w_bram_data,
    input [31:0] bias_bram_data,

    //ppu.sv
    output opsum_valid,
    output [31:0] opsum,

    // Profiling/debug signals.
    output profile_load_active,
    output profile_op_active,
    output profile_weight_stall
);
reg [1:0] cs;
reg [1:0] ns;
parameter IDLE=2'd0, LOAD_PARA=2'd1, OP=2'd2;

reg [6:0] ctr64; //in state
reg [2:0] ctr8; // in state
reg [6:0] ctr_tile; //global
reg [4:0] ctr_bias; //global

reg [16:0] act_base_addr_r, w_base_addr_r, bias_base_addr_r;
reg [6:0] k_tile_cnt_r;

assign act_bram_addr = act_base_addr_r + ctr64;
assign w_bram_addr = w_base_addr_r + ctr64;
wire bias_load_done = (ctr_bias == 5'd16);
assign bias_bram_addr = bias_base_addr_r + (bias_load_done ? 17'd15 : {12'd0, ctr_bias});

reg load_word_valid;
reg [6:0] load_word_idx;
reg bias_word_valid;
reg [4:0] bias_word_idx;
reg [95:0] act_word_buf;
reg [95:0] w_word_buf;
reg [95:0] bias_word_buf;

wire load_row_done = load_word_valid & (load_word_idx[1:0] == 2'd3);
wire final_load_data = load_word_valid & (load_word_idx == 7'd63);
wire bias_group_done = bias_word_valid & (bias_word_idx[1:0] == 2'd3);
wire [127:0] act_row_in = {act_bram_data, act_word_buf};
wire [127:0] w_row_in = {w_bram_data, w_word_buf};
wire [127:0] bias_row_in = {bias_bram_data, bias_word_buf};
wire push_act_row = load_row_done;
wire push_w_row = load_row_done;
wire push_bias = bias_group_done;
wire psum_buff_ready;
wire can_load_word = w_bram_valid & (psum_buff_ready | bias_load_done);
wire load_word_request = (cs == LOAD_PARA) & (~final_load_data) & can_load_word & (ctr64 < 7'd64);
wire bias_word_request = (cs == LOAD_PARA) & (~final_load_data) & can_load_word & (~bias_load_done);
wire weight_wait = (cs == LOAD_PARA) & (~final_load_data) &
                   (psum_buff_ready | bias_load_done) &
                   (ctr64 < 7'd64) & (~w_bram_valid);
reg pass;
reg m_tile_end;

assign act_bram_rd_en = load_word_request;
assign w_bram_rd_en = load_word_request;
assign bias_bram_rd_en = bias_word_request;
assign profile_load_active = (cs == LOAD_PARA);
assign profile_op_active = (cs == OP);
assign profile_weight_stall = weight_wait;

//pe connection
//addition 1 wire is redundant
wire [7:0] W0 [15+1:0][7:0];
wire [29:0] W1 [15+1:0][7:0]; //ACOUT
wire [17:0] act [15:0][7+1:0]; //row, col
wire [19:0] psum [15+1:0][15:0];
//row 16 = output to psum buffer
//col interleave: psum0(0, 2,..14), psum1(1,3,..15)

wire [127:0] act_fifo_o;
wire [319:0] psum_buff_i;

always @(posedge clk) begin
    if(!rst_n) begin
        cs <= IDLE;
    end
    else begin
        case (cs)
            IDLE: begin
                cs <= en?LOAD_PARA:cs;
            end
            LOAD_PARA: begin
                cs <= final_load_data ? OP :LOAD_PARA;
            end
            OP: begin
                if(ctr64 == 6'd38 && ctr8 == 3'd4) begin //finish 16 by 16 GEMM
                    if(ctr_tile == k_tile_cnt_r) begin //finish m-tile, back to IDLE, opsum output
                        cs <= IDLE; 
                    end
                    else begin
                        cs <= LOAD_PARA;  
                    end
                end
            end
        endcase
    end
end

always @(posedge clk) begin
    if(!rst_n) begin
        act_base_addr_r <= 17'd0;
        w_base_addr_r <= 17'd0;
        bias_base_addr_r <= 17'd0;
        k_tile_cnt_r <= 7'd0;
        ctr64 <= 7'd0;
        ctr8 <= 3'd0;
        ctr_tile <= 7'd1;
        ctr_bias <= 5'd0;
        load_word_valid <= 1'b0;
        load_word_idx <= 7'd0;
        bias_word_valid <= 1'b0;
        bias_word_idx <= 5'd0;
        act_word_buf <= 96'd0;
        w_word_buf <= 96'd0;
        bias_word_buf <= 96'd0;
    end
    else begin
        case(cs)
            IDLE: begin
            if(en) begin //store para, for the whole m-tile
                act_base_addr_r <= act_base_addr;
                w_base_addr_r <= w_base_addr;
                bias_base_addr_r <= bias_base_addr;
                k_tile_cnt_r <= k_tile_cnt;
            end

            //reset data flow regs
            ctr64 <= 7'd0;
            ctr8 <= 3'd0;
            ctr_tile <= 7'd1;
            ctr_bias <= 5'd0;
            load_word_valid <= 1'b0;
            load_word_idx <= 7'd0;
            bias_word_valid <= 1'b0;
            bias_word_idx <= 5'd0;
            act_word_buf <= 96'd0;
            w_word_buf <= 96'd0;
            bias_word_buf <= 96'd0;
        end
        LOAD_PARA: begin
            //load 16 rows of w/act from 64 32-bit BRAM words
            if(load_word_valid) begin
                case(load_word_idx[1:0])
                    2'd0: begin
                        act_word_buf[31:0] <= act_bram_data;
                        w_word_buf[31:0] <= w_bram_data;
                    end
                    2'd1: begin
                        act_word_buf[63:32] <= act_bram_data;
                        w_word_buf[63:32] <= w_bram_data;
                    end
                    2'd2: begin
                        act_word_buf[95:64] <= act_bram_data;
                        w_word_buf[95:64] <= w_bram_data;
                    end
                    default: begin
                        act_word_buf <= 96'd0;
                        w_word_buf <= 96'd0;
                    end
                endcase
            end

            if(bias_word_valid) begin
                case(bias_word_idx[1:0])
                    2'd0: bias_word_buf[31:0] <= bias_bram_data;
                    2'd1: bias_word_buf[63:32] <= bias_bram_data;
                    2'd2: bias_word_buf[95:64] <= bias_bram_data;
                    default: bias_word_buf <= 96'd0;
                endcase
            end

            load_word_valid <= 1'b0;
            bias_word_valid <= 1'b0;

            if(final_load_data) begin
                ctr64 <= 7'd0;
                act_base_addr_r <= act_base_addr_r + 17'd64; //base for nxt k-tile
                w_base_addr_r <= w_base_addr_r + 17'd64;
            end
            else if(load_word_request) begin
                load_word_idx <= ctr64;
                load_word_valid <= 1'b1;
                ctr64 <= ctr64 + 7'd1;
            end

            if(bias_word_request) begin
                bias_word_idx <= ctr_bias;
                bias_word_valid <= 1'b1;
                ctr_bias <= ctr_bias + 5'd1;
            end
        end
        OP: begin
            load_word_valid <= 1'b0;
            bias_word_valid <= 1'b0;
            
            if(ctr8 == 3'd4) begin //count 5 cycle
                ctr8 <= 3'b0; //ctr8 ==0 => en(comb)
                
                if(ctr64 == 6'd38) begin //finish 16 by 16 GEMM
                    ctr64 <= 7'd0;
                    ctr_tile <= ctr_tile + 7'd1;
                end 
                else begin
                    ctr64 <= ctr64 + 7'd1;                    
                end
            end
            else begin
                ctr8 <= ctr8 + 3'b1;
            end
        end
        endcase
    end
end

//state, ctrl sig, i/o
always @(*) begin
    case (cs)
        IDLE: begin
            //ns = en?LOAD_PARA:cs;
            module_ready = 1'b1;
            //push_act_row = 1'b0;
            //push_w_row = 1'b0;
            pass = 1'b0;
            m_tile_end = 1'b0;
        end
        LOAD_PARA: begin
            module_ready = 1'b0;
            //ns = (ctr64 == 6'd16)?OP :LOAD_PARA;
            pass = 1'b0;
            m_tile_end = 1'b0;
        end
        OP: begin
            module_ready = 1'b0;
            //push_act_row = 1'b0;
            //push_w_row = 1'b0;
            pass = (ctr8 == 3'd0);

            if(ctr64 == 6'd38 && ctr8 == 3'd4) begin //finish 16 by 16 GEMM
                if(ctr_tile == k_tile_cnt_r) begin //finish m-tile, back to IDLE, opsum output
                    //ns = IDLE; 
                    m_tile_end = 1'b1;
                end
                else begin
                    //ns = LOAD_PARA;  
                    m_tile_end = 1'b0;                  
                end
            end
            else begin
                //ns = cs;
                m_tile_end = 1'b0;
            end
        end
        default: begin
            pass = 1'd0;
            module_ready = 1'b0;
            //ns = IDLE;
            //push_act_row = 1'b0;
            //push_w_row = 1'b0;
            m_tile_end = 1'b0;
        end
    endcase
end

Act_fifo act_buff(
    .clk(clk),
    .rst_n(rst_n),

    .push_row(push_act_row),                 // write 1 row, when act valid && act ready
    .act_row_in(act_row_in),

    .pop_row(pass),                  // output 1 skewed row
    .act_row_out(act_fifo_o)
);

Opsum_acc psum_buff(
    .clk(clk),
    .rst_n(rst_n),

    .push_bias(push_bias),
    .bias_in(bias_row_in),

    .push_row(pass),
    .psum_in(psum_buff_i), //peak: 16*20', 0 at [19:0], 16 at [319:300]
    .end_of_acc(m_tile_end), // last pusm input, m-tile finish (accumulate all k) 

    .module_ready(psum_buff_ready),
    .opsum_valid(opsum_valid),
    .opsum(opsum)
);

//pe array
genvar i,j;
generate
    for(j=0; j<8; j=j+1) begin // => row 15
        //weight row 0, fr input, interleave
        assign W0 [16][j] = w_row_in[8*2*j+:8];
        assign W1 [16][j] = {{6{w_row_in[8*(2*j+1)+7]}}, w_row_in[8*(2*j+1)+:8], 16'b0}; 
                            //sign [24:17], {6{W1[7]}}, W1, 16'b0
    end
    for (i=0; i<16; i=i+1) begin
        //act col 0, fr input
        assign act[i][0] = {10'b0, act_fifo_o[i*8+:8]}; //uint8, padding to 18'

        //psum row 0, assign to 0
        assign psum[0][i] = 20'b0;

        //psum row 16, feed to opsum buffer
        assign psum_buff_i[i*20+:20] = psum[16][i];
    end

    //pe array connection
    for(i=0; i<16; i=i+1) begin //row
        for(j=0; j<8; j=j+1) begin //col
            PE_pack pe(
                .clk(clk), 
                .rst_n(rst_n),
                .push_weight_row(push_w_row), //pre load weight, 16 cyc
                .en(pass),

                .W0(W0[i+1][j]),
                .W1(W1[i+1][j]),
                .act_in(act[i][j]), //uint8, zero padding to 18'
                .ipsum0(psum[i][2*j]),
                .ipsum1(psum[i][2*j+1]),

                .W0_bypass(W0[i][j]), //row 15 => row0
                .W1_bypass(W1[i][j]), //use ACOUT port
                .act_bypass(act[i][j+1]), //cout to dsp
                .opsum0(psum[i+1][2*j]),
                .opsum1(psum[i+1][2*j+1]) //output to nxt accumulator
            );
        end
    end
endgenerate


endmodule

module Opsum_acc(
    //input en,
    input clk,
    input rst_n,

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
    if(!rst_n) begin
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

module PE_pack(
    //ctrl signal
    input clk, 
    input rst_n,
    input push_weight_row, //pre load weight
    input en, // 4 cycle 0, 1 cyc en, totally 5 cyc

    input [7:0] W0, //int8
    input [29:0] W1, //use ACIN port
    input [17:0] act_in, //uint8, zero padding to 18'
    input [19:0] ipsum0, ipsum1,

    output [7:0] W0_bypass,
    output [29:0] W1_bypass, //use ACOUT port
    output [17:0] act_bypass, //cout to dsp
    output [19:0] opsum0, opsum1 //output to nxt accumulator
);

//register type, DSP
reg signed [29:0] A_dsp;
reg signed [24:0] D_dsp;
reg signed [17:0] B_dsp;
reg signed [24:0] W_pack_dsp;
reg signed [42:0] mul_dsp;
reg signed [47:0] P_dsp;

//latch ipsum for accumulate
reg [19:0] ipsum0_r, ipsum1_r;
wire [19:0] psum0 = {{4{P_dsp[15]}}, P_dsp[15:0]};
wire [19:0] psum1 = P_dsp[35:16];
assign opsum0 = ipsum0_r + psum0;
assign opsum1 = ipsum1_r + psum1;

assign act_bypass = B_dsp;
assign W0_bypass = D_dsp[7:0];
assign W1_bypass = A_dsp;

always @(posedge clk) begin
    if(!rst_n) begin
        ipsum0_r <= 20'd0;
        ipsum1_r <= 20'd0;
        A_dsp <= 30'd0; //sign [24:17], {6{W1[7]}}, W1, 16'b0
        D_dsp <= 25'd0; //sign [7:0]
        B_dsp <= 18'd0; //unsign, max: +255
        W_pack_dsp <= 25'd0;
        mul_dsp <= 43'd0; //sign mul
        P_dsp <= 48'd0; //correct the upper psum, when lower psum <0
    end
    else begin
        if(en) begin
            ipsum0_r <= ipsum0;
            ipsum1_r <= ipsum1;
        end 

        //DSP
        if(push_weight_row) begin //16 cycle
            A_dsp <= W1; //sign [24:17], {6{W1[7]}}, W1, 16'b0
            D_dsp <= {{17{W0[7]}}, W0}; //sign [7:0]
        end

        if(en) begin
            B_dsp <= act_in; //unsign, max: +255
        end
        W_pack_dsp <= A_dsp[24:0] + D_dsp;
        mul_dsp <= W_pack_dsp * B_dsp; //sign mul
        P_dsp <= mul_dsp + {26'b0, mul_dsp[15], 16'b0}; //correct the upper psum, when lower psum <0
    end
    
end

endmodule

module Act_fifo (
    input clk,
    input rst_n,

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
    if(!rst_n) begin
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
