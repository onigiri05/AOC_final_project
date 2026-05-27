// case1: N=8, d=8, Br=Bc=4
// Two Q-tiles (i=0..3, i=4..7) × two K/V-tiles per Q-tile.
// Exercises the online-softmax correction factor:
//   m_new = max(m_old, row_max(S_ij)) may differ from m_old on j=1 tile,
//   requiring O_i *= exp(m_old - m_new) before accumulating.

#ifndef WORKLOAD_H
#define WORKLOAD_H

#define CASE_N   8
#define CASE_D   8
#define CASE_BR  4

#endif  // WORKLOAD_H
