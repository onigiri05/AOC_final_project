/* vit_ops.cpp — ViT building-block operations (CPU, fp32) */

#include "vit_ops.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── Linear ──────────────────────────────────────────────────────────────────
   weight[out_d, in_d] (PyTorch row-major)
   output[m][o] = Σ_k input[m][k] * weight[o][k] + bias[o]
   ─────────────────────────────────────────────────────────────────────────── */
void linear(const float* input, const float* weight, const float* bias,
            float* output, int M, int in_d, int out_d) {
    for (int m = 0; m < M; m++) {
        const float* in_row = input + m * in_d;
        float*       out_row = output + m * out_d;
        for (int o = 0; o < out_d; o++) {
            float sum = bias ? bias[o] : 0.0f;
            const float* w_row = weight + o * in_d;
            for (int k = 0; k < in_d; k++)
                sum += in_row[k] * w_row[k];
            out_row[o] = sum;
        }
    }
}

/* ── LayerNorm ───────────────────────────────────────────────────────────── */
void layer_norm(float* x, const float* gamma, const float* beta,
                int N, int d, float eps) {
    for (int i = 0; i < N; i++) {
        float* row = x + i * d;

        float mean = 0.0f;
        for (int j = 0; j < d; j++) mean += row[j];
        mean /= (float)d;

        float var = 0.0f;
        for (int j = 0; j < d; j++) {
            float v = row[j] - mean;
            var += v * v;
        }
        var /= (float)d;

        float inv_std = 1.0f / sqrtf(var + eps);
        for (int j = 0; j < d; j++)
            row[j] = (row[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

/* ── RMSNorm ─────────────────────────────────────────────────────────────────
   No mean subtraction, no beta.  Cheaper than LayerNorm by ~1/3 FLOPs.
   y_i = x_i / sqrt(1/d * Σ x_j² + eps) * gamma_i
   ─────────────────────────────────────────────────────────────────────────── */
void rms_norm(float* x, const float* gamma, int N, int d, float eps) {
    for (int i = 0; i < N; i++) {
        float* row = x + i * d;

        float sumsq = 0.0f;
        for (int j = 0; j < d; j++)
            sumsq += row[j] * row[j];

        float inv_rms = 1.0f / sqrtf(sumsq / (float)d + eps);
        for (int j = 0; j < d; j++)
            row[j] = row[j] * inv_rms * gamma[j];
    }
}

/* ── GELU (erf form, matches PyTorch default) ───────────────────────────── */
void gelu_inplace(float* x, int len) {
    /* GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))) */
    static const float INV_SQRT2 = 0.7071067811865476f;
    for (int i = 0; i < len; i++) {
        float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * INV_SQRT2));
    }
}

/* ── Patch extraction ────────────────────────────────────────────────────────
   img[C=3, H=224, W=224] → patches[196, 768]
   Patch order: row-major over the 14×14 grid.
   Within each patch: channel-major (all ch0 pixels, then ch1, ch2).
   ─────────────────────────────────────────────────────────────────────────── */
void extract_patches(const float* img, float* patches) {
    for (int pr = 0; pr < 14; pr++) {
        for (int pc = 0; pc < 14; pc++) {
            float* dst = patches + (pr * 14 + pc) * 768;
            int idx = 0;
            for (int ch = 0; ch < 3; ch++) {
                for (int py = 0; py < 16; py++) {
                    int y = pr * 16 + py;
                    for (int px = 0; px < 16; px++) {
                        int x_pix = pc * 16 + px;
                        dst[idx++] = img[ch * 224 * 224 + y * 224 + x_pix];
                    }
                }
            }
        }
    }
}

/* ── Prepend CLS token + positional embedding ───────────────────────────── */
void prepend_cls_pos(float* tokens, const float* patch_out,
                     const float* cls, const float* pos_embed) {
    const int D = 384;
    /* Row 0: CLS token + pos_embed[0] */
    for (int d = 0; d < D; d++)
        tokens[d] = cls[d] + pos_embed[d];
    /* Rows 1..196: patch embedding + pos_embed[1..196] */
    for (int i = 0; i < 196; i++) {
        const float* pe  = pos_embed + (i + 1) * D;
        const float* src = patch_out + i * D;
        float*       dst = tokens    + (i + 1) * D;
        for (int d = 0; d < D; d++)
            dst[d] = src[d] + pe[d];
    }
}

/* ── Residual add ────────────────────────────────────────────────────────── */
void residual_add(float* x, const float* delta, int len) {
    for (int i = 0; i < len; i++)
        x[i] += delta[i];
}

/* ── Softmax (in-place, numerically stable) ─────────────────────────────── */
void softmax_inplace(float* x, int len) {
    float m = x[0];
    for (int i = 1; i < len; i++) if (x[i] > m) m = x[i];
    float s = 0.0f;
    for (int i = 0; i < len; i++) { x[i] = expf(x[i] - m); s += x[i]; }
    for (int i = 0; i < len; i++) x[i] /= s;
}

/* ── Argmax ──────────────────────────────────────────────────────────────── */
int argmax(const float* x, int len) {
    int best = 0;
    for (int i = 1; i < len; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

/* ── Top-k (simple selection sort, k is small) ─────────────────────────── */
void topk(const float* x, int len, int k, int* indices, float* values) {
    /* Use a tiny scratch array to avoid modifying x */
    float* tmp = (float*)malloc(len * sizeof(float));
    int*   idx = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) { tmp[i] = x[i]; idx[i] = i; }
    for (int i = 0; i < k; i++) {
        int best = i;
        for (int j = i + 1; j < len; j++)
            if (tmp[j] > tmp[best]) best = j;
        float tf = tmp[i]; tmp[i] = tmp[best]; tmp[best] = tf;
        int   ti = idx[i]; idx[i] = idx[best]; idx[best] = ti;
        indices[i] = idx[i];
        values[i]  = tmp[i];
    }
    free(tmp);
    free(idx);
}
