// case2: N=196, d=64, Br=Bc=14  (ViT-Small/16 attention head)
//
// N=196 = 14×14 patch tokens (no CLS token for divisibility).
// d=64  = head_dim for ViT-Small (384/6 heads = 64).
// Br=14 = PYNQ-Z2 14×14 systolic array tile size.
//
// 14 i-tiles × 14 j-tiles × 14 rows × 64 words per row = 2,752 DMA requests.
// Tests realistic ViT attention workload and DMA bandwidth.

#ifndef WORKLOAD_H
#define WORKLOAD_H

#define CASE_N   196
#define CASE_D   64
#define CASE_BR  14

#endif  // WORKLOAD_H
