`timescale 1ns/1ps
// systolic_array.sv — 14×14 Output-Stationary Systolic Array (Timing Model, INT8)
//
// Computes:  out_mat[i][j] = Σ_k  a_mat[i][k] × b_mat[j][k]
//            (i.e.  C = A × B^T,  where B is stored row-major)
//            INT8 operands → INT32 accumulator (no overflow: max depth×127²=1,032,256).
//
// Used twice per FlashAttention tile pair:
//   GEMM1: a_mat = Q_tile_int8 [Br×d],  b_mat = K_tile_int8 [Br×d],  depth = d
//          → out_mat = S_int32 [Br×Br]  (dequantized to fp64 by wrapper)
//
//   GEMM2: a_mat = P_tile_int8 [Br×Br], b_mat = V^T_chunk_int8 [Br×chunk], depth = Br
//          → out_mat = ΔO_int32 [Br×chunk]  (dequantized and accumulated by wrapper)
//
// Cycle-accurate timing model (staggered-input systolic array):
//   Latency = 2×(SA_SIZE−1) + depth  clock cycles after the start pulse.
//   The actual arithmetic is performed behaviorally on the final cycle so that
//   simulation stays fast; the cycle count matches a real pipelined SA.

module systolic_array #(
    parameter int SA_SIZE = 14,   // PE grid dimension (rows = cols = Br)
    parameter int MAX_D   = 64    // maximum inner dimension (= max d or Br)
)(
    input  logic       ACLK,
    input  logic       ARESETn,

    // ── Control ──────────────────────────────────────────────────────────────
    input  logic       start,        // 1-cycle pulse: begin GEMM
    input  logic [7:0] depth,        // inner-product length k ∈ [0, depth)
    input  logic [7:0] active_rows,  // valid output rows  (≤ SA_SIZE)
    input  logic [7:0] active_cols,  // valid output cols  (≤ SA_SIZE)
    output logic       done,         // 1-cycle pulse: out_mat is ready

    // ── Input matrices: quantized INT8 operands (stable throughout computation) ──
    // a_mat[row][k]  —  row-major, first index is PE row
    // b_mat[col][k]  —  b_mat[j][k] = B^T[k][j], first index is PE column
    input  byte signed a_mat [0:SA_SIZE-1][0:MAX_D-1],
    input  byte signed b_mat [0:SA_SIZE-1][0:MAX_D-1],

    // ── Output: INT32 accumulator [SA_SIZE × SA_SIZE], valid one cycle after done ──
    output int         out_mat [0:SA_SIZE-1][0:SA_SIZE-1]
);

// ── Run counter ───────────────────────────────────────────────────────────────
// Total latency from start to done:  2*(SA_SIZE-1) + depth  cycles
logic [8:0] run_cnt;
logic       running;

// ── Main sequential block ─────────────────────────────────────────────────────
always_ff @(posedge ACLK or negedge ARESETn) begin
    if (!ARESETn) begin
        running  <= 1'b0;
        done     <= 1'b0;
        run_cnt  <= '0;
        foreach (out_mat[i,j]) out_mat[i][j] <= 0;

    end else begin
        done <= 1'b0;  // default de-assert

        if (start) begin
            // Begin new GEMM: clear accumulators, reset counter
            running <= 1'b1;
            run_cnt <= 9'd0;
            foreach (out_mat[i,j]) out_mat[i][j] <= 0;

        end else if (running) begin
            if (run_cnt == 9'(2*(SA_SIZE-1)) + 9'(depth) - 9'd1) begin

                // Final cycle: behavioral INT8×INT8→INT32 GEMM
                begin : sa_gemm
                    int ii, jj, kk;
                    int acc;
                    for (ii = 0; ii < SA_SIZE; ii++) begin
                        for (jj = 0; jj < SA_SIZE; jj++) begin
                            if (ii < int'(active_rows) && jj < int'(active_cols)) begin
                                acc = 0;
                                for (kk = 0; kk < int'(depth); kk++)
                                    acc += int'(a_mat[ii][kk]) * int'(b_mat[jj][kk]);
                                out_mat[ii][jj] <= acc;
                            end
                            // Inactive entries remain 0 from the start-clear above
                        end
                    end
                end

                running <= 1'b0;
                done    <= 1'b1;

            end else begin
                run_cnt <= run_cnt + 9'd1;
            end
        end
    end
end

endmodule
