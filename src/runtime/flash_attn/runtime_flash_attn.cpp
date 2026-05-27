// runtime_flash_attn.cpp — high-level FlashAttention runtime API
//
// flash_attention(): quantize Q/K/V to INT8 globally, configure hardware,
//                   trigger computation, wait for IRQ.
// standard_attention_cpu(): CPU reference for correctness verification.

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "driver_flash_attn.h"
#include "runtime.h"

/* ── CPU reference: standard (non-tiled) attention ────────────────────────── */
void standard_attention_cpu(const float* Q, const float* K, const float* V,
                             float* O, uint32_t N, uint32_t d) {
    float scale = 1.0f / sqrtf((float)d);

    float* S = (float*)malloc(N * N * sizeof(float));

    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < d; k++)
                s += Q[i * d + k] * K[j * d + k];
            S[i * N + j] = s * scale;
        }
    }

    for (uint32_t i = 0; i < N; i++) {
        float m = S[i * N + 0];
        for (uint32_t j = 1; j < N; j++)
            if (S[i * N + j] > m) m = S[i * N + j];

        float sum = 0.0f;
        for (uint32_t j = 0; j < N; j++) {
            S[i * N + j] = expf(S[i * N + j] - m);
            sum += S[i * N + j];
        }
        for (uint32_t j = 0; j < N; j++)
            S[i * N + j] /= sum;
    }

    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t k = 0; k < d; k++) {
            float o = 0.0f;
            for (uint32_t j = 0; j < N; j++)
                o += S[i * N + j] * V[j * d + k];
            O[i * d + k] = o;
        }
    }

    free(S);
}

/* ── Symmetric INT8 quantization helpers ────────────────────────────────── */

/* Compute global symmetric scale: max_abs(matrix) / 127 */
static float compute_scale(const float* data, uint32_t len) {
    float max_abs = 0.0f;
    for (uint32_t i = 0; i < len; i++) {
        float v = fabsf(data[i]);
        if (v > max_abs) max_abs = v;
    }
    return (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
}

/* Quantize fp32 matrix to packed INT8 (1 byte per element, natural layout).
 * Hardware reads 4 consecutive bytes as one 32-bit AXI word (little-endian).
 * out[i] = clamp(round(data[i] / scale), -127, 127)
 */
static void quantize_to_int8(const float* data, int8_t* out,
                              uint32_t len, float scale) {
    for (uint32_t i = 0; i < len; i++) {
        float v = data[i] / scale;
        int   ri = (int)(v >= 0.0f ? v + 0.5f : v - 0.5f);
        if (ri >  127) ri =  127;
        if (ri < -127) ri = -127;
        out[i] = (int8_t)ri;
    }
}

/* ── Hardware FlashAttention ─────────────────────────────────────────────── */
void fa_stop() {
    fa_reg_write(FA_CONTROL_OFFSET, 0);
}

/* ── Scale tying ─────────────────────────────────────────────────────────────
 * Build with -DFA_SCALE_TYING to enable.
 *
 * Without tying (default):
 *   q_scale = max(|Q|)/127    k_scale = max(|K|)/127    (independent)
 *   Hardware dequantizes GEMM1 result as: S_fp = S_int × q_scale × k_scale
 *   Advantage: optimal per-matrix quantization range.
 *
 * With tying:
 *   qk_scale = max(max(|Q|), max(|K|)) / 127    (shared)
 *   Hardware dequantizes GEMM1 result as: S_fp = S_int × qk_scale²
 *   Advantage: single scale factor avoids potential range mismatch between
 *   Q and K that can cause INT8 overflow in extreme cases.  Also simplifies
 *   the hardware dequantization path (one multiply instead of two distinct
 *   scale loads).
 * ─────────────────────────────────────────────────────────────────────────── */

int flash_attention(float* Q, float* K, float* V, float* O,
                    uint32_t N, uint32_t d, uint32_t Br) {
    uint32_t len = N * d;

    /* Compute global symmetric scales */
    float v_scale = compute_scale(V, len);

#ifdef FA_SCALE_TYING
    /* Tied Q/K scale: use max of both ranges for the shared scale. */
    float qk_scale = fmaxf(compute_scale(Q, len), compute_scale(K, len));
    float q_scale  = qk_scale;
    float k_scale  = qk_scale;
#else
    /* Independent Q and K scales (default). */
    float q_scale = compute_scale(Q, len);
    float k_scale = compute_scale(K, len);
#endif

    /* Allocate INT8 packed buffers (1 byte/element, 4 fit in one 32-bit AXI word) */
    int8_t* Q_int8 = (int8_t*)malloc(len * sizeof(int8_t));
    int8_t* K_int8 = (int8_t*)malloc(len * sizeof(int8_t));
    int8_t* V_int8 = (int8_t*)malloc(len * sizeof(int8_t));

    quantize_to_int8(Q, Q_int8, len, q_scale);
    quantize_to_int8(K, K_int8, len, k_scale);
    quantize_to_int8(V, V_int8, len, v_scale);

    /* Configure accelerator */
    fa_set_shape(N, d);
    fa_set_tile(Br);
    fa_set_q_addr(Q_int8);
    fa_set_k_addr(K_int8);
    fa_set_v_addr(V_int8);
    fa_set_o_addr(O);
    fa_set_q_scale(q_scale);
    fa_set_k_scale(k_scale);
    fa_set_v_scale(v_scale);

    /* Reset runtime counter so metrics only cover this invocation */
    get_fa_hal()->reset_runtime_info();

    /* Kick off computation */
    fa_start();

    /* Block until accelerator signals done */
    get_fa_hal()->wait_for_irq();

    fa_stop();

    free(Q_int8);
    free(K_int8);
    free(V_int8);

    return 0;
}
