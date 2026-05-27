#ifndef VIT_OPS_H
#define VIT_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Linear (fully-connected) layer ─────────────────────────────────────────
   weight[out_d, in_d] stored row-major (PyTorch convention).
   output[M, out_d] = input[M, in_d] × weight^T + bias[out_d]
   Pass bias=NULL to skip bias addition.
   ────────────────────────────────────────────────────────────────────────── */
void linear(const float* input, const float* weight, const float* bias,
            float* output, int M, int in_d, int out_d);

/* ── LayerNorm (row-wise) ─────────────────────────────────────────────────
   x[N, d] normalized in-place along the last dim with affine transform.
   gamma (scale) and beta (shift) are both [d].
   eps typically 1e-6.
   Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
   ────────────────────────────────────────────────────────────────────────── */
void layer_norm(float* x, const float* gamma, const float* beta,
                int N, int d, float eps);

/* ── RMSNorm (row-wise) ──────────────────────────────────────────────────
   Cheaper alternative to LayerNorm: no mean subtraction, no beta offset.
   Formula: y = x / sqrt(mean(x^2) + eps) * gamma
   gamma [d] only (no beta).
   Used in LLaMA-style models; included here for comparative benchmarking.
   NOTE: ViT-Small/16 pretrained weights were trained with LayerNorm.
         Switching to RMSNorm will degrade accuracy without re-training.
         Build with -DVIT_USE_RMSNORM to activate globally in runtime_vit.
   ────────────────────────────────────────────────────────────────────────── */
void rms_norm(float* x, const float* gamma, int N, int d, float eps);

/* ── GELU activation (in-place, erf approximation) ──────────────────────── */
void gelu_inplace(float* x, int len);

/* ── Patch extraction ────────────────────────────────────────────────────────
   img     : [3, 224, 224] float32, channel-first (C,H,W)
   patches : [196, 768]    float32, output — 196 patches of 16×16×3=768
   ────────────────────────────────────────────────────────────────────────── */
void extract_patches(const float* img, float* patches);

/* ── Prepend CLS + add positional embedding ─────────────────────────────────
   patch_out : [196, 384]  patch embedding output
   cls       : [384]       CLS token vector
   pos_embed : [197, 384]  positional embedding (row 0 = CLS pos)
   tokens    : [197, 384]  output — tokens[0]=CLS+pos[0], tokens[1:]=patch+pos[1:]
   ────────────────────────────────────────────────────────────────────────── */
void prepend_cls_pos(float* tokens, const float* patch_out,
                     const float* cls, const float* pos_embed);

/* ── Residual add ────────────────────────────────────────────────────────────
   x[len] += delta[len]  (element-wise)
   ────────────────────────────────────────────────────────────────────────── */
void residual_add(float* x, const float* delta, int len);

/* ── Softmax over logits (in-place) ─────────────────────────────────────── */
void softmax_inplace(float* x, int len);

/* ── Argmax ──────────────────────────────────────────────────────────────── */
int  argmax(const float* x, int len);

/* ── Top-k indices + values (sorted descending) ─────────────────────────── */
void topk(const float* x, int len, int k, int* indices, float* values);

#ifdef __cplusplus
}
#endif

#endif /* VIT_OPS_H */
