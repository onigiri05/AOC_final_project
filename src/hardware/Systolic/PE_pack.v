module PE_pack(
    //ctrl signal
    input clk, 
    input rst,
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
wire [19:0] psum0 = {{3{P_dsp[16]}}, P_dsp[16:0]};
wire [19:0] psum1 = {{3{P_dsp[33]}}, P_dsp[33:17]};
assign opsum0 = ipsum0_r + psum0;
assign opsum1 = ipsum1_r + psum1;

assign act_bypass = B_dsp;
assign W0_bypass = D_dsp[7:0];
assign W1_bypass = A_dsp;

always @(posedge clk) begin
    if(rst) begin
        ipsum0_r <= 20'd0;
        ipsum1_r <= 20'd0;
        A_dsp <= 30'd0; //sign [24:17], {5{W1[7]}}, W1, 17'b0
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
            A_dsp <= W1; //sign [24:17], {5{W1[7]}}, W1, 17'b0
            D_dsp <= {{17{W0[7]}}, W0}; //sign [7:0]
        end

        if(en) begin
            B_dsp <= act_in; //unsign, max: +255
        end
        W_pack_dsp <= A_dsp[24:0] + D_dsp;
        mul_dsp <= W_pack_dsp * B_dsp; //sign mul
        P_dsp <= mul_dsp + {25'b0, mul_dsp[16], 17'b0}; //correct the upper psum, when lower psum <0
    end
    
end

endmodule