#ifndef RUNTIME_H
#define RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/*
    FlashAttention Accelerator Runtime API
    ======================================
    Single-head attention: O = flash_attention(Q, K, V)
    where Q, K, V, O are all (N × d) fp32 matrices stored row-major.

    Hardware parameters:
      N        : sequence length (must be divisible by Br)
      d        : head dimension
      Br = Bc  : tile size (PYNQ-Z2 systolic array → default 14)
*/

/**
 * @brief  Stop the FlashAttention accelerator.
 */
void fa_stop();

/**
 * @brief  Run one head of FlashAttention on the hardware accelerator.
 *
 * @param  Q        Input Q matrix [N × d], row-major fp32 in DRAM.
 * @param  K        Input K matrix [N × d], row-major fp32 in DRAM.
 * @param  V        Input V matrix [N × d], row-major fp32 in DRAM.
 * @param  O        Output O matrix [N × d], row-major fp32 in DRAM.
 * @param  N        Sequence length (must be divisible by Br).
 * @param  d        Head dimension.
 * @param  Br       Tile row size (= Bc, tile column size).
 * @return 0 on success.
 */
int flash_attention(float* Q, float* K, float* V, float* O,
                    uint32_t N, uint32_t d, uint32_t Br);

/**
 * @brief  CPU reference implementation of standard (non-tiled) attention.
 *
 * Computes O = softmax(Q * K^T / sqrt(d)) * V  for verification.
 * Uses fp32 throughout; O(N^2 * d) time and O(N^2) scratch space.
 *
 * @param  Q   [N × d] fp32
 * @param  K   [N × d] fp32
 * @param  V   [N × d] fp32
 * @param  O   [N × d] fp32 output
 * @param  N   Sequence length
 * @param  d   Head dimension
 */
void standard_attention_cpu(const float* Q, const float* K, const float* V,
                             float* O, uint32_t N, uint32_t d);

#ifdef __cplusplus
}
#endif

#endif  // RUNTIME_H
