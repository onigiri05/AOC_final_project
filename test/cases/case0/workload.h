// case0: N=4, d=4, Br=Bc=4
// Single tile per head — all N rows fit in one Q/K/V tile,
// so no online-softmax correction is needed (j-loop runs once).
// Simplest possible test: verifies basic Q*K^T/sqrt(d), softmax, and *V.

#ifndef WORKLOAD_H
#define WORKLOAD_H

#define CASE_N   4
#define CASE_D   4
#define CASE_BR  4

#endif  // WORKLOAD_H
