`include "PE_pack.v"
`include "Act_fifo.v"
`include "Opsum_acc.v"

module Systolic(
    input clk,
    input rst,
    
    // ctrl & config for each m_tile
    input en,
    output reg module_ready,      // module able to use
    input [16:0] act_base_addr, // BRAM total: 4096 * 140 B, / 8B per word, 17' addr
    input [16:0] w_base_addr,
    input [6:0] k_tile_cnt, //at most 96, at FC2

    // BRAM(SRAM) input data, for each k-tile
    output [16:0] act_bram_addr, 
    output [16:0] w_bram_addr,
    input act_bram_valid, 
    input w_bram_valid,
    input [127:0] act_bram_row,
    input [127:0] w_bram_row,

    //ppu.sv
    output opsum_valid,
    output [31:0] opsum
);
reg [1:0] cs, ns;
parameter IDLE=2'd0, LOAD_PARA=2'd1, OP=2'd2;

reg unsigned [5:0] ctr64; //in state
reg unsigned [2:0] ctr8; // in state
reg unsigned [6:0] ctr_tile; //global

reg [16:0] act_base_addr_r, w_base_addr_r;
reg [6:0] k_tile_cnt_r;

assign act_bram_addr = act_base_addr_r + ctr64;
assign w_bram_addr = w_base_addr_r + ctr64;
reg push_act_row, push_w_row;
reg pass;
reg m_tile_end;

always @(posedge clk) begin
    if(rst) begin
        cs <= IDLE;
    end
    else begin
        cs <= ns;
    end
end

always @(posedge clk) begin
    case(cs)
        IDLE: begin
            if(en) begin //store para, for the whole m-tile
                act_base_addr_r <= act_base_addr;
                w_base_addr_r <= w_base_addr;
                k_tile_cnt_r <= k_tile_cnt;
            end

            //reset data flow regs
            ctr64 <= 6'b0;
            ctr8 <= 3'd0;
            ctr_tile <= 7'd0;
        end
        LOAD_PARA: begin
            //load 16 row of w, act
            ctr64 <= (act_bram_valid && w_bram_valid) ? ctr64 + 6'd1: ctr64;
            
            if(ctr64 == 6'd15) begin
                ctr64 <= 6'd0;
                act_base_addr_r <= act_base_addr_r + 17'd16; //base for nxt k-tile
                w_base_addr_r <= w_base_addr_r + 17'd16;
            end

            ctr8 <= 3'd0;
        end
        OP: begin
            if(psum_buff_ready) begin
                if(ctr8 == 3'd4) begin //count 5 cycle
                    ctr8 <= 3'b0; //ctr8 ==0 => en(comb)
                    
                    if(ctr64 == 6'd38) begin //finish 16 by 16 GEMM
                        ctr64 <= 6'd0;
                        ctr_tile <= ctr_tile + 7'd1;
                    end 
                    else begin
                        ctr64 <= ctr64 + 6'd1;                    
                    end
                end
                else begin
                    ctr8 <= ctr8 + 3'b1;
                end
            end
            
        end
    endcase
end

//state, ctrl sig, i/o
always @(*) begin
    case (cs)
        IDLE: begin
            ns = en?LOAD_PARA:IDLE;
            module_ready = 1'b1;
            push_act_row = 1'b0;
            push_w_row = 1'b0;
            pass = 1'b0;
            m_tile_end = 1'b0;
        end
        LOAD_PARA: begin
            module_ready = 1'b0;
            ns = (ctr64 == 6'd15)?OP :LOAD_PARA;
            push_act_row = (act_bram_valid && w_bram_valid);
            push_w_row = (act_bram_valid && w_bram_valid);
            pass = 1'b0;
            m_tile_end = 1'b0;
        end
        OP: begin
            module_ready = 1'b0;
            push_act_row = 1'b0;
            push_w_row = 1'b0;
            pass = (ctr8 == 3'd0);

            if(ctr64 == 6'd38 && ctr8 == 3'd4) begin //finish 16 by 16 GEMM
                if(ctr_tile == k_tile_cnt_r) begin //finish m-tile, back to IDLE, opsum output
                    ns = IDLE; 
                    m_tile_end = 1'b1;
                end
                else begin
                    ns = LOAD_PARA;  
                    m_tile_end = 1'b0;                  
                end
            end
            else begin
                ns = cs;
                m_tile_end = 1'b0;
            end
        end
        default: begin
            pass = 1'd0;
            module_ready = 1'b0;
            ns = IDLE;
            push_act_row = 1'b0;
            push_w_row = 1'b0;
            m_tile_end = 1'b0;
        end
    endcase
end

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
wire psum_buff_ready;

Act_fifo act_buff(
    .clk(clk),
    .rst(rst),

    .push_row(push_act_row),                 // write 1 row, when act valid && act ready
    .act_row_in(act_bram_row),

    .pop_row(pass),                  // output 1 skewed row
    .act_row_out(act_fifo_o)
);

Opsum_acc psum_buff(
    .clk(clk),
    .rst(rst),

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
    for(j=0; j<8; j=j+1) begin //row 0
        //weight row 0, fr input, interleave
        assign W0 [0][j] = w_bram_row[8*2*j+:8];
        assign W1 [0][j] = {{5{w_bram_row[8*(2*j+1)+7]}}, w_bram_row[8*(2*j+1)+:8], 17'b0}; 
                            //sign [24:17], {5{W1[7]}}, W1, 17'b0
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
                .rst(rst),
                .push_weight_row(push_w_row), //pre load weight, 16 cyc
                .en(pass),

                .W0(W0[i][j]),
                .W1(W1[i][j]),
                .act_in(act[i][j]), //uint8, zero padding to 18'
                .ipsum0(psum[i][2*j]),
                .ipsum1(psum[i][2*j+1]),

                .W0_bypass(W0[i+1][j]),
                .W1_bypass(W1[i+1][j]), //use ACOUT port
                .act_bypass(act[i][j+1]), //cout to dsp
                .opsum0(psum[i+1][2*j]),
                .opsum1(psum[i+1][2*j+1]) //output to nxt accumulator
            );
        end
    end
endgenerate

endmodule
