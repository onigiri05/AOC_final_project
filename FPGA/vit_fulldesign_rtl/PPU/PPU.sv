`include "ASIC.svh"

// ============================================================
// Module: PPU
//
// Resource-oriented implementation.
//
// The previous version instantiated one GELU + one Requant + residual/RMS
// datapath for every element of a 16x16 tile.  That was fast, but on PYNQ-Z2
// it expanded to hundreds of LUT/CARRY paths.  This version keeps the tile
// control interface, but streams systolic opsum one element at a time.  That
// removes the extra full psum tile bus/register layer in the top level.
//
// Mode map:
//   2'b00: residual add + RMS stat, used by Output Projection -> X_mid
//   2'b01: GELU + requant, used by FC1
//   2'b10: residual add only, used by FC2 -> X_out
//   2'b11: requant only, used by QKV/Q-tile/Attention*V
// ============================================================
module PPU #(
    parameter int TOKEN_NUM       = 197,
    parameter int CHANNEL_NUM     = 384,
    parameter int TOKEN_TILE      = 8,
    parameter int CHANNEL_TILE    = 8,
    parameter int DATA_W          = 8,
    parameter int SUM_W           = 32,
    parameter int TOKEN_W         = 8,
    parameter int CHANNEL_TILE_W  = 6,
    parameter logic [7:0] ZERO_POINT = 8'd128
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [1:0] ppu_mode_i,
    input  logic [5:0] scaling_factor_i,

    input  logic tile_valid_i,
    output logic tile_ready_o,

    input  logic                     psum_valid_i,
    output logic                     psum_ready_o,
    input  logic signed [`DATA_BITS-1:0] psum_i,
    input  logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0]     residual_tile_i,

    input  logic [TOKEN_W-1:0]        base_token_idx_i,
    input  logic [CHANNEL_TILE_W-1:0] channel_tile_idx_i,
    input  logic [TOKEN_TILE-1:0]     token_valid_mask_i,

    output logic data_tile_valid_o,
    input  logic data_tile_ready_i,
    output logic [TOKEN_TILE*CHANNEL_TILE*DATA_W-1:0] data_tile_o,
    output logic data_word_valid_o,
    input  logic data_word_ready_i,
    output logic [5:0] data_word_idx_o,
    output logic [31:0] data_word_o,
    output logic data_word_last_o,

    output logic stat_valid_o,
    input  logic stat_ready_i,
    output logic [TOKEN_W-1:0] stat_token_idx_o,
    output logic [SUM_W-1:0]   sum_sq_o
);

    localparam logic [1:0] PPU_MODE_RESID_STAT = 2'b00;
    localparam logic [1:0] PPU_MODE_FC1_GELU   = 2'b01;
    localparam logic [1:0] PPU_MODE_RESID_ONLY = 2'b10;
    localparam logic [1:0] PPU_MODE_REQUANT    = 2'b11;

    localparam int TILE_ELEMS        = TOKEN_TILE * CHANNEL_TILE;
    localparam int TILE_ELEM_W       = (TILE_ELEMS <= 1) ? 1 : $clog2(TILE_ELEMS);
    localparam int NUM_CHANNEL_TILES = CHANNEL_NUM / CHANNEL_TILE;
    localparam int CHANNEL_TILE_LOG2 = $clog2(CHANNEL_TILE);
    localparam logic [TILE_ELEM_W-1:0] CHANNEL_TILE_MASK = CHANNEL_TILE - 1;

    typedef enum logic [2:0] {
        ST_IDLE,
        ST_WAIT_PSUM,
        ST_COMMIT,
        ST_STAT_READ,
        ST_STAT_WRITE
    } state_t;

    state_t state_q;

    logic [1:0] ppu_mode_q;
    logic [5:0] scaling_factor_q;
    logic [TOKEN_W-1:0] base_token_idx_q;
    logic [CHANNEL_TILE_W-1:0] channel_tile_idx_q;
    logic [TOKEN_TILE-1:0] token_valid_mask_q;

    (* max_fanout = 64 *) logic [TILE_ELEM_W-1:0] elem_idx_q;
    (* max_fanout = 64 *) logic [3:0] row_idx_q;
    (* max_fanout = 64 *) logic [3:0] stat_idx_q;

    logic signed [`DATA_BITS-1:0] lane_psum_q;
    logic [DATA_W-1:0] lane_residual_q;
    logic signed [`DATA_BITS-1:0] gelu_in_q;
    logic gelu_en_q;
    logic signed [`DATA_BITS-1:0] gelu_out;

    logic signed [`DATA_BITS-1:0] requant_in;
    logic [DATA_W-1:0] requant_out;
    logic [DATA_W-1:0] selected_out;

    logic [SUM_W-1:0] row_sum_q [0:TOKEN_TILE-1];
    (* ram_style = "block" *) logic [SUM_W-1:0] partial_sum_mem [0:TOKEN_NUM-1];

    logic stat_acc_en_q;
    logic last_channel_tile_q;
    integer stat_token_idx_int;
    logic [TOKEN_W-1:0] stat_token_idx_q;
    logic stat_token_valid_q;
    logic [SUM_W-1:0] partial_sum_rd_q;
    logic [SUM_W-1:0] stat_final_sum_next;
    logic [31:0] data_word_pack_q;
    logic [31:0] data_word_pack_next;
    logic [SUM_W-1:0] selected_sq;
    logic word_complete_q;
    logic commit_can_advance;

    assign tile_ready_o = (state_q == ST_IDLE) &&
                          ((!data_tile_valid_o) || data_tile_ready_i) &&
                          ((!data_word_valid_o) || data_word_ready_i) &&
                          ((!stat_valid_o) || stat_ready_i);
    assign psum_ready_o = (state_q == ST_WAIT_PSUM);
    assign data_tile_o = '0;
    assign word_complete_q = (elem_idx_q[1:0] == 2'd3);
    assign commit_can_advance = (!word_complete_q) ||
                                ((!data_word_valid_o) || data_word_ready_i);
    assign selected_sq = square_u8_zp128(selected_out);

    always_comb begin
        data_word_pack_next = data_word_pack_q;
        unique case (elem_idx_q[1:0])
            2'd0: data_word_pack_next[7:0]   = selected_out;
            2'd1: data_word_pack_next[15:8]  = selected_out;
            2'd2: data_word_pack_next[23:16] = selected_out;
            default: data_word_pack_next[31:24] = selected_out;
        endcase
    end

    GELU_Unit u_gelu (
        .clk(clk),
        .rst_n(rst_n),
        .en(gelu_en_q),
        .data_in(gelu_in_q),
        .data_out(gelu_out)
    );

    Requant_Unit u_requant (
        .data_in(requant_in),
        .scaling_factor(scaling_factor_q),
        .data_out(requant_out)
    );

    function automatic logic [DATA_W-1:0] add_u8_zp128_clamp;
        input logic [DATA_W-1:0] main_q;
        input logic [DATA_W-1:0] residual_q;
        logic signed [10:0] sum_q;
        begin
            sum_q = $signed({1'b0, main_q}) +
                    $signed({1'b0, residual_q}) -
                    $signed({1'b0, ZERO_POINT});

            if (sum_q < 11'sd0)
                add_u8_zp128_clamp = 8'd0;
            else if (sum_q > 11'sd255)
                add_u8_zp128_clamp = 8'd255;
            else
                add_u8_zp128_clamp = sum_q[DATA_W-1:0];
        end
    endfunction

    function automatic logic [SUM_W-1:0] square_u8_zp128;
        input logic [DATA_W-1:0] q;
        logic signed [9:0] centered_x;
        logic signed [19:0] square_x;
        begin
            centered_x = $signed({1'b0, q}) - $signed({2'b00, ZERO_POINT});
            square_x   = centered_x * centered_x;
            square_u8_zp128 = {{(SUM_W-20){1'b0}}, square_x[19:0]};
        end
    endfunction

    always_comb begin
        requant_in = (ppu_mode_q == PPU_MODE_FC1_GELU) ? gelu_out : lane_psum_q;

        unique case (ppu_mode_q)
            PPU_MODE_RESID_STAT,
            PPU_MODE_RESID_ONLY: selected_out = add_u8_zp128_clamp(requant_out, lane_residual_q);
            default:             selected_out = requant_out;
        endcase
    end

    assign stat_final_sum_next = partial_sum_rd_q + row_sum_q[stat_idx_q];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state_q             <= ST_IDLE;
            ppu_mode_q          <= PPU_MODE_REQUANT;
            scaling_factor_q    <= 6'd0;
            base_token_idx_q    <= '0;
            channel_tile_idx_q  <= '0;
            token_valid_mask_q  <= '0;
            elem_idx_q          <= '0;
            row_idx_q           <= 4'd0;
            stat_idx_q          <= 4'd0;
            gelu_en_q           <= 1'b0;
            stat_acc_en_q       <= 1'b0;
            last_channel_tile_q <= 1'b0;
            stat_token_idx_q    <= '0;
            stat_token_valid_q  <= 1'b0;
            data_tile_valid_o   <= 1'b0;
            data_word_valid_o   <= 1'b0;
            data_word_idx_o     <= 6'd0;
            data_word_last_o    <= 1'b0;
            stat_valid_o        <= 1'b0;
            stat_token_idx_o    <= '0;
        end
        else begin
            gelu_en_q <= 1'b0;

            if (data_tile_valid_o && data_tile_ready_i) begin
                data_tile_valid_o <= 1'b0;
            end

            if (data_word_valid_o && data_word_ready_i) begin
                data_word_valid_o <= 1'b0;
                data_word_last_o  <= 1'b0;
            end

            if (stat_valid_o && stat_ready_i) begin
                stat_valid_o <= 1'b0;
            end

            case (state_q)
                ST_IDLE: begin
                    if (tile_valid_i && tile_ready_o) begin
                        ppu_mode_q          <= ppu_mode_i;
                        scaling_factor_q    <= scaling_factor_i;
                        base_token_idx_q    <= base_token_idx_i;
                        channel_tile_idx_q  <= channel_tile_idx_i;
                        token_valid_mask_q  <= token_valid_mask_i;
                        elem_idx_q          <= '0;
                        stat_idx_q          <= 4'd0;
                        stat_acc_en_q       <= (ppu_mode_i == PPU_MODE_RESID_STAT);
                        last_channel_tile_q <= (channel_tile_idx_i == (NUM_CHANNEL_TILES - 1));
                        data_word_pack_q    <= 32'd0;

                        state_q <= ST_WAIT_PSUM;
                    end
                end

                ST_WAIT_PSUM: begin
                    if (psum_valid_i) begin
                        row_idx_q       <= elem_idx_q >> CHANNEL_TILE_LOG2;
                        lane_psum_q     <= psum_i;
                        lane_residual_q <= residual_tile_i[elem_idx_q*DATA_W +: DATA_W];
                        gelu_in_q       <= psum_i;
                        gelu_en_q       <= (ppu_mode_q == PPU_MODE_FC1_GELU);
                        state_q         <= ST_COMMIT;
                    end
                end

                ST_COMMIT: begin
                    if (commit_can_advance) begin
                        data_word_pack_q <= data_word_pack_next;

                        if (word_complete_q) begin
                            data_word_valid_o <= 1'b1;
                            data_word_idx_o   <= elem_idx_q[TILE_ELEM_W-1:2];
                            data_word_o       <= data_word_pack_next;
                            data_word_last_o  <= (elem_idx_q == (TILE_ELEMS - 1));
                            data_word_pack_q  <= 32'd0;
                        end

                        if (stat_acc_en_q && token_valid_mask_q[row_idx_q]) begin
                            if ((elem_idx_q & CHANNEL_TILE_MASK) == '0)
                                row_sum_q[row_idx_q] <= selected_sq;
                            else
                                row_sum_q[row_idx_q] <= row_sum_q[row_idx_q] + selected_sq;
                        end

                        if (elem_idx_q == (TILE_ELEMS - 1)) begin
                            data_tile_valid_o <= 1'b1;
                            stat_idx_q        <= 4'd0;
                            state_q           <= stat_acc_en_q ? ST_STAT_READ : ST_IDLE;
                        end
                        else begin
                            elem_idx_q <= elem_idx_q + {{(TILE_ELEM_W-1){1'b0}}, 1'b1};
                            state_q    <= ST_WAIT_PSUM;
                        end
                    end
                end

                ST_STAT_READ: begin
                    if ((!stat_valid_o) || stat_ready_i) begin
                        stat_token_idx_int = base_token_idx_q + stat_idx_q;
                        stat_token_idx_q <= stat_token_idx_int[TOKEN_W-1:0];
                        stat_token_valid_q <= token_valid_mask_q[stat_idx_q] &&
                                              (stat_token_idx_int < TOKEN_NUM);

                        if (token_valid_mask_q[stat_idx_q] && (stat_token_idx_int < TOKEN_NUM)) begin
                            if (channel_tile_idx_q == '0)
                                partial_sum_rd_q <= '0;
                            else
                                partial_sum_rd_q <= partial_sum_mem[stat_token_idx_int];
                        end
                        else begin
                            partial_sum_rd_q <= '0;
                        end

                        state_q <= ST_STAT_WRITE;
                    end
                end

                ST_STAT_WRITE: begin
                    if (stat_token_valid_q) begin
                        if (last_channel_tile_q) begin
                            stat_valid_o     <= 1'b1;
                            stat_token_idx_o <= stat_token_idx_q;
                            sum_sq_o         <= stat_final_sum_next;
                            partial_sum_mem[stat_token_idx_q] <= '0;
                        end
                        else begin
                            partial_sum_mem[stat_token_idx_q] <= stat_final_sum_next;
                        end
                    end

                    if ((!last_channel_tile_q) || (!stat_token_valid_q) ||
                        ((!stat_valid_o) || stat_ready_i)) begin
                        if (stat_idx_q == (TOKEN_TILE - 1)) begin
                            state_q <= ST_IDLE;
                        end
                        else begin
                            stat_idx_q <= stat_idx_q + 4'd1;
                            state_q    <= ST_STAT_READ;
                        end
                    end
                end

                default: begin
                    state_q <= ST_IDLE;
                end
            endcase
        end
    end

endmodule
