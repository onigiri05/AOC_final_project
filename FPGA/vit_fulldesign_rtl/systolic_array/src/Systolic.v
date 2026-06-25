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
    input [7:0] k_tile_cnt, // at most 192 in the 8x8 FC2 path
    input [7:0] act_zero_point, // subtract before MAC; use 0 for raw/attention, 128 for zp128 activations

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
    input opsum_ready,
    output opsum_valid,
    output [31:0] opsum
);
reg [1:0] cs;
reg [1:0] ns;
parameter IDLE=2'd0, LOAD_PARA=2'd1, OP=2'd2;
localparam integer SYS_DIM = 8;
localparam integer PACK_COLS = SYS_DIM / 2;
localparam integer ROW_WORDS = SYS_DIM / 4;
localparam integer LOAD_WORDS = SYS_DIM * ROW_WORDS;
localparam integer BIAS_WORDS = SYS_DIM;
localparam [6:0] LOAD_LAST_WORD = LOAD_WORDS - 1;
localparam [4:0] BIAS_DONE_COUNT = BIAS_WORDS;
localparam [4:0] BIAS_LAST_WORD = BIAS_WORDS - 1;
localparam [6:0] OP_LAST_CTR = (SYS_DIM * 2) + 6;

reg [6:0] ctr64; //in state
reg [2:0] ctr8; // in state
reg [7:0] ctr_tile; //global
reg [4:0] ctr_bias; //global

reg [16:0] act_base_addr_r, w_base_addr_r, bias_base_addr_r;
reg [7:0] k_tile_cnt_r;

assign act_bram_addr = act_base_addr_r + ctr64;
assign w_bram_addr = w_base_addr_r + ctr64;
wire bias_load_done = (ctr_bias == BIAS_DONE_COUNT);
assign bias_bram_addr = bias_base_addr_r + (bias_load_done ? {12'd0, BIAS_LAST_WORD} : {12'd0, ctr_bias});

reg load_word_valid;
reg [6:0] load_word_idx;
reg bias_word_valid;
reg [4:0] bias_word_idx;
reg [95:0] act_word_buf;
reg [95:0] w_word_buf;
reg [95:0] bias_word_buf;

wire load_row_done = load_word_valid & (load_word_idx[0] == 1'b1);
wire final_load_data = load_word_valid & (load_word_idx == LOAD_LAST_WORD);
wire bias_group_done = bias_word_valid & (bias_word_idx[1:0] == 2'd3);
wire [127:0] act_row_in = {64'd0, act_bram_data, act_word_buf[31:0]};
wire [127:0] w_row_in = {64'd0, w_bram_data, w_word_buf[31:0]};
wire [127:0] bias_row_in = {bias_bram_data, bias_word_buf};
(* max_fanout = 32 *) wire push_act_row;
(* max_fanout = 32 *) wire push_w_row;
(* max_fanout = 32 *) wire push_bias;
assign push_act_row = load_row_done;
assign push_w_row = load_row_done;
assign push_bias = bias_group_done;
wire psum_buff_ready;
wire can_load_word = w_bram_valid & (psum_buff_ready | bias_load_done);
wire load_word_request = (cs == LOAD_PARA) & (~final_load_data) & can_load_word & (ctr64 < LOAD_WORDS);
wire bias_word_request = (cs == LOAD_PARA) & (~final_load_data) & can_load_word & (~bias_load_done);
(* max_fanout = 32 *) reg pass;
reg m_tile_end;

assign act_bram_rd_en = load_word_request;
assign w_bram_rd_en = load_word_request;
assign bias_bram_rd_en = bias_word_request;

//pe connection
//addition 1 wire is redundant
wire [7:0] W0 [SYS_DIM:0][PACK_COLS-1:0];
wire [29:0] W1 [SYS_DIM:0][PACK_COLS-1:0]; //ACOUT
wire [17:0] act [SYS_DIM-1:0][PACK_COLS:0]; //row, col
wire [19:0] psum [SYS_DIM:0][SYS_DIM-1:0];
//row SYS_DIM = output to psum buffer
//col interleave: psum0(0, 2,..14), psum1(1,3,..15)

wire [127:0] act_fifo_o;
wire [SYS_DIM*20-1:0] psum_buff_i;

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
                if(ctr64 == OP_LAST_CTR && ctr8 == 3'd4) begin //finish one systolic tile
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
        k_tile_cnt_r <= 8'd0;
        ctr64 <= 7'd0;
        ctr8 <= 3'd0;
        ctr_tile <= 8'd1;
        ctr_bias <= 5'd0;
        load_word_valid <= 1'b0;
        load_word_idx <= 7'd0;
        bias_word_valid <= 1'b0;
        bias_word_idx <= 5'd0;
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
            ctr_tile <= 8'd1;
            ctr_bias <= 5'd0;
            load_word_valid <= 1'b0;
            load_word_idx <= 7'd0;
            bias_word_valid <= 1'b0;
            bias_word_idx <= 5'd0;
        end
        LOAD_PARA: begin
            //load 8 rows of w/act from 16 32-bit BRAM words
            if(load_word_valid) begin
                case(load_word_idx[0])
                    1'b0: begin
                        act_word_buf[31:0] <= act_bram_data;
                        w_word_buf[31:0] <= w_bram_data;
                    end
                    default: begin
                    end
                endcase
            end

            if(bias_word_valid) begin
                case(bias_word_idx[1:0])
                    2'd0: bias_word_buf[31:0] <= bias_bram_data;
                    2'd1: bias_word_buf[63:32] <= bias_bram_data;
                    2'd2: bias_word_buf[95:64] <= bias_bram_data;
                    default: begin
                    end
                endcase
            end

            load_word_valid <= 1'b0;
            bias_word_valid <= 1'b0;

            if(final_load_data) begin
                ctr64 <= 7'd0;
                act_base_addr_r <= act_base_addr_r + LOAD_WORDS; //base for nxt k-tile
                w_base_addr_r <= w_base_addr_r + LOAD_WORDS;
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
                
                if(ctr64 == OP_LAST_CTR) begin //finish one systolic tile
                    ctr64 <= 7'd0;
                    ctr_tile <= ctr_tile + 8'd1;
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

            if(ctr64 == OP_LAST_CTR && ctr8 == 3'd4) begin //finish one systolic tile
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
    .opsum_ready(opsum_ready),
    .opsum_valid(opsum_valid),
    .opsum(opsum)
);

//pe array
genvar i,j;
generate
    for(j=0; j<PACK_COLS; j=j+1) begin
        //weight row 0, fr input, interleave
        assign W0 [SYS_DIM][j] = w_row_in[8*2*j+:8];
        assign W1 [SYS_DIM][j] = {{6{w_row_in[8*(2*j+1)+7]}}, w_row_in[8*(2*j+1)+:8], 16'b0}; 
                            //sign [24:17], {6{W1[7]}}, W1, 16'b0
    end
    for (i=0; i<SYS_DIM; i=i+1) begin
        //act col 0, fr input. Convert external uint8 to signed value by subtracting act_zero_point.
        wire signed [8:0] act_centered;
        assign act_centered = $signed({1'b0, act_fifo_o[i*8+:8]}) - $signed({1'b0, act_zero_point});
        assign act[i][0] = {{9{act_centered[8]}}, act_centered};

        //psum row 0, assign to 0
        assign psum[0][i] = 20'b0;

        //psum final row, feed to opsum buffer
        assign psum_buff_i[i*20+:20] = psum[SYS_DIM][i];
    end

    //pe array connection
    for(i=0; i<SYS_DIM; i=i+1) begin //row
        for(j=0; j<PACK_COLS; j=j+1) begin //packed col pair
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
    input [159:0] psum_in, //peak: 8*20', 0 at [19:0], 7 at [159:140]
    input end_of_acc, // last pusm input, m-tile finish (accumulate all k) 

    output module_ready, //ready to accept input
    input opsum_ready,
    output opsum_valid,
    output [31:0] opsum
);

// Do not reset the large datapath arrays.  They are re-initialized by the
// bias load sequence before every output tile.  Keeping reset off these arrays
// avoids thousands of extra control-set constrained FFs on Zynq-7020.
(* shreg_extract = "yes" *) reg signed [39:0] skew_latch [2:0][2:0]; //col, latch cycle, 2 pack psum, for 4-1 packed cols

reg [2:0] cs, ns;
localparam IDLE = 2'b0, OP = 2'd1, OUT = 2'd2;
reg signed [31:0] psum_buffer [7:0][7:0]; //acc buff
reg [2:0] acc_row_ptr;
reg [3:0] out_row, out_col; //0~255
(* shreg_extract = "yes" *) reg signed [31:0] out_row_buf [7:0];
reg out_row_loaded;
reg [5:0] push_count_q;
reg module_ready_, opsum_valid_;
reg signed [31:0] opsum_;
assign module_ready = module_ready_;
assign opsum_valid = opsum_valid_;
assign opsum = opsum_;

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
    if(!rst_n) begin
            opsum_valid_ <= 1'b0;
            acc_row_ptr <= 3'd5;
            out_row <= 4'd0;
            out_col <= 4'd0;
            out_row_loaded <= 1'b0;
            push_count_q <= 6'd0;
            opsum_ <= 32'sd0;
    end
    else begin
    case (cs)
        IDLE: begin
            opsum_valid_ <= 1'b0;
            acc_row_ptr <= 3'd5;
            out_row <= 4'd0;
            out_col <= 4'd0;
            out_row_loaded <= 1'b0;
            push_count_q <= 6'd0;
            opsum_ <= 32'sd0;
        end
        OP: begin //m-tile, no need to clear
            opsum_valid_ <= 1'b0;
            // == 1, acc
            if(push_bias) begin
                for(i = 0; i<8; i= i+1) begin
                    psum_buffer[i][4] <= bias_in[31:0];
                    psum_buffer[i][5] <= bias_in[63:32];    
                    psum_buffer[i][6] <= bias_in[95:64];    
                    psum_buffer[i][7] <= bias_in[127:96];
                end
                for(j=0; j>=0; j = j-4) begin
                    for(i = 0; i<8; i= i+1) begin
                        psum_buffer[i][j] <= psum_buffer[i][j+4];
                        psum_buffer[i][j+1] <= psum_buffer[i][j+4+1];    
                        psum_buffer[i][j+2] <= psum_buffer[i][j+4+2];    
                        psum_buffer[i][j+3] <= psum_buffer[i][j+4+3];
                    end
                end
            end
            else if(push_row) begin
                if (push_count_q != 6'd31)
                    push_count_q <= push_count_q + 6'd1;

                //skew latch
                for(i = 0; i<3; i= i+1) begin //input
                    skew_latch[i][i][19:0] <= psum_in[40*i+:20];
                    skew_latch[i][i][39:20] <= psum_in[40*(i)+20+:20];
                end
                for(i=0; i<2; i = i+1) begin //shift 1, no need to clear
                    for(j=i; j<2; j=j+1) begin
                        skew_latch [i][j+1] <= skew_latch[i][j];
                    end
                end
                
                if (acc_row_ptr == 3'd7)
                    acc_row_ptr <= 3'd0;
                else
                    acc_row_ptr <= acc_row_ptr + 3'd1;
                

                //accumulator
                psum_buffer[acc_row_ptr][7] <= psum_buffer[acc_row_ptr][7] + $signed(psum_in[159:140]); //direct from input
                psum_buffer[acc_row_ptr][6] <= psum_buffer[acc_row_ptr][6] + $signed(psum_in[139:120]); //direct from input

                for(i = 0; i<3; i=i+1) begin //column
                    if (push_count_q > (2 - i)) begin
                        psum_buffer[acc_row_ptr][i*2+1] <= psum_buffer[acc_row_ptr][i*2+1] + $signed(skew_latch[i][2][39:20]);
                        psum_buffer[acc_row_ptr][i*2] <= psum_buffer[acc_row_ptr][i*2] + $signed(skew_latch[i][2][19:0]);
                    end
                end
            end
        end
        OUT: begin
            if (!out_row_loaded) begin
                for(i = 0; i<8; i= i+1) begin
                    out_row_buf[i] <= psum_buffer[out_row][i];
                end
                out_col <= 4'd0;
                out_row_loaded <= 1'b1;
            end
            else if (!opsum_valid_) begin
                opsum_ <= out_row_buf[0];
                opsum_valid_ <= 1'b1;
            end
            else if (opsum_ready) begin
                opsum_valid_ <= 1'b0;
                for(i = 0; i<7; i= i+1) begin
                    out_row_buf[i] <= out_row_buf[i+1];
                end

                if (out_col == 4'd7) begin
                    out_col <= 4'd0;
                    out_row_loaded <= 1'b0;
                    if (out_row != 4'd7)
                        out_row <= out_row + 4'd1;
                end
                else begin
                    out_col <= out_col + 4'd1;
                end
            end
        end
    endcase
    end
end

//state & out
always @(*) begin
    case (cs)
        IDLE: begin
            ns = OP;
            module_ready_ = 1'b0;
        end
        OP: begin
            module_ready_ = 1'b1;
            if(end_of_acc) //last row pushed
                ns = OUT;
            else
                ns = OP;
        end
        OUT: begin
            module_ready_ = 1'b0;
            if ((out_row == 4'd7) && (out_col == 4'd7) && opsum_valid_ && opsum_ready) begin
                ns = IDLE;
            end
            else begin
                ns = OUT;
            end
        end
        default: begin
            module_ready_ = 1'b0;
            ns = IDLE;

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
`ifndef SYNTHESIS
    if (!rst_n) begin
        ipsum0_r <= 20'd0;
        ipsum1_r <= 20'd0;
        A_dsp <= 30'd0;
        D_dsp <= 25'd0;
        B_dsp <= 18'd0;
        W_pack_dsp <= 25'd0;
        mul_dsp <= 43'd0;
        P_dsp <= 48'd0;
    end
    else
`endif
    if (push_weight_row) begin
        // With the 8x8 array the local clear cost is much lower than the
        // old 16x16 version, and it avoids stale warm-up values entering
        // the first output row after a tile switch.
        ipsum0_r <= 20'd0;
        ipsum1_r <= 20'd0;
        A_dsp <= W1; //sign [24:17], {6{W1[7]}}, W1, 16'b0
        D_dsp <= {{17{W0[7]}}, W0}; //sign [7:0]
        B_dsp <= 18'd0;
        W_pack_dsp <= 25'd0;
        mul_dsp <= 43'd0;
        P_dsp <= 48'd0;
    end
    else begin
        if(en) begin
            ipsum0_r <= ipsum0;
            ipsum1_r <= ipsum1;
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

(* shreg_extract = "yes" *) reg [7:0] act_r [7:0][7:0]; //8 row
(* shreg_extract = "yes" *) reg [7:0] skew_latch [6:0][6:0]; //row, latch cycle, without row 0
reg [3:0] pop_count_q;

assign act_row_out[7:0] = act_r[0][0];
assign act_row_out[127:64] = 64'd0;
genvar k;
generate
    for(k=1; k<8; k=k+1) begin
        localparam [3:0] COL_IDX = k;
        assign act_row_out[k*8+:8] = (pop_count_q >= COL_IDX) ? skew_latch[0][k-1] : 8'd0;
    end
endgenerate

integer i, j;
always @(posedge clk) begin
`ifndef SYNTHESIS
        if(!rst_n) begin
            pop_count_q <= 4'd0;
            for(i = 0; i<8; i = i+1) begin
                for(j = 0; j<8; j= j+1) begin
                    act_r[i][j] <= 8'b0;
                end
            end
            for(i = 0; i<7; i = i+1) begin
                for(j=0; j<=i; j=j+1) begin
                    skew_latch[j][i] <= 8'b0;
                end
            end
        end
        else
`endif
        if(push_row) begin
            pop_count_q <= 4'd0;
            for(i = 0; i<8; i = i+1) begin                 //row 7: input
                act_r[7][i] <= act_row_in[8*i+:8];
            end

            for(i=6; i>=0; i=i-1) begin                    //shift 1 row (row 7 => row 0)
                for(j=0; j<8; j=j+1) begin
                    act_r [i][j] <= act_r[i+1][j];
                end
            end
        end
        else if(pop_row) begin
            if (pop_count_q != 4'd7)
                pop_count_q <= pop_count_q + 4'd1;

            for(i=0; i<8; i=i+1) begin                     //row 7 feed 0
                act_r [7][i] <= 8'b0;
            end
            for(i=6; i>=0; i=i-1) begin                    //shift 1 row (row 7 => row 0)
                for(j=0; j<8; j=j+1) begin
                    act_r [i][j] <= act_r[i+1][j];
                end
            end

            for(i=0; i<7; i= i+1) begin                    //row 0 push to skew buffer
                skew_latch [i][i] <= act_r[0][i+1];
            end
            for(i = 0; i<7; i = i+1) begin                 // shift 1 row, row i => row0
                for(j=0; j<i; j=j+1) begin
                    skew_latch [j][i] <= skew_latch[j+1][i];
                end
            end
        end
end

endmodule
