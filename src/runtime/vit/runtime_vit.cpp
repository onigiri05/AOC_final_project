/* runtime_vit.cpp — ViT-Small/16 end-to-end inference engine.
 *
 * Attention is dispatched to the FA hardware accelerator (one head at a time).
 * All other ops (LayerNorm, QKV/O projection, MLP) run on the host CPU (fp32).
 *
 * CLS token handling:
 *   The FA hardware is fixed at N=196, Br=14 (patch tokens only; N%Br==0).
 *   Patch tokens attend to each other via the HW accelerator (N=196).
 *   CLS attention (Q_cls attending to all 197 tokens) is computed in software
 *   using single_query_attention() — one query row × 197 KV pairs per head.
 *   This restores standard ViT behaviour for the classification token.
 */

#include "runtime_vit.h"
#include "vit_weights.h"
#include "vit_ops.h"
#include "runtime.h"          /* flash_attention(), standard_attention_cpu() */
#include "driver_flash_attn.h"/* get_fa_hal()                                */
#include "hal.hpp"             /* runtime_info struct                         */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Norm selector ───────────────────────────────────────────────────────────
   Build with -DVIT_USE_RMSNORM to substitute RMSNorm for LayerNorm globally.
   RMSNorm is cheaper (no mean, no beta) but requires a model trained with it.
   ViT-Small/16 pretrained weights use LayerNorm; switching will hurt accuracy.
   Use RMSNorm only for benchmarking or when loading RMSNorm-trained weights.
   ─────────────────────────────────────────────────────────────────────────── */
#ifdef VIT_USE_RMSNORM
#define NORM(x, gamma, beta, N, d, eps) rms_norm((x), (gamma), (N), (d), (eps))
#else
#define NORM(x, gamma, beta, N, d, eps) layer_norm((x), (gamma), (beta), (N), (d), (eps))
#endif

/* Shorthand for dimension constants */
#define T   VIT_TOKENS      /* 197 */
#define N   VIT_PATCH_N     /* 196 */
#define D   VIT_EMBED_DIM   /* 384 */
#define H   VIT_HEADS       /* 6   */
#define HD  VIT_HEAD_DIM    /* 64  */
#define M   VIT_MLP_DIM     /* 1536 */
#define C   VIT_CLASSES     /* 1000 */
#define BR  VIT_TILE_BR     /* 14  */
#define EPS 1e-6f

/* ── HW stats accumulator ────────────────────────────────────────────────────
   flash_attention() calls reset_runtime_info() before each HW run, so the
   HAL only ever holds one call's data at a time.  We accumulate immediately
   after each call before the next one can reset the counters.
   ─────────────────────────────────────────────────────────────────────────── */

/* Global accumulator — zeroed by vit_hw_stats_reset(); read by testbench.
   Declared volatile so the compiler does not elide or reorder the
   read-modify-write sequences across successive flash_attention() calls. */
static volatile VitHWStats g_hw_stats;

void vit_hw_stats_reset(VitHWStats* s) {
    memset(s, 0, sizeof(*s));
    /* Cast away volatile for memset — safe here since we own the object. */
    memset((void*)&g_hw_stats, 0, sizeof(g_hw_stats));
}

void vit_hw_stats_accumulate(VitHWStats* s) {
    /* Copy volatile fields one-by-one to avoid undefined volatile struct copy. */
    s->total_cycles      = g_hw_stats.total_cycles;
    s->total_memory_read  = g_hw_stats.total_memory_read;
    s->total_memory_write = g_hw_stats.total_memory_write;
    s->fa_call_count      = g_hw_stats.fa_call_count;
    s->last_call_cycles   = g_hw_stats.last_call_cycles;
}

/* Internal: call flash_attention() and immediately capture per-call stats.
   Explicit read-modify-write (not compound +=) ensures the compiler emits
   a load, an add, and a store — preventing potential elision with -O2.
   flash_attention() calls reset_runtime_info() internally before running,
   so ri contains ONLY this call's elapsed cycles and DMA bytes. */
static void fa_call_and_accumulate(float* Q, float* K, float* V, float* O,
                                   uint32_t n, uint32_t d, uint32_t br) {
    /* FA_interrupt stays HIGH after each computation and is only cleared by
       hardware reset (not by fa_stop() or by asserting start).  Without this
       reset, wait_for_irq() on calls 2+ exits immediately — still seeing the
       previous call's asserted interrupt — producing garbage output for all
       subsequent heads.  reset() pulses ARESETn, clears FA_interrupt, and
       returns the state machine to idle.  flash_attention() then re-writes all
       configuration registers before kicking off the new computation. */
    get_fa_hal()->reset();

    flash_attention(Q, K, V, O, n, d, br);

    struct runtime_info ri = get_fa_hal()->get_runtime_info();

    /* Explicit read-modify-write into volatile fields. */
    uint64_t cy  = g_hw_stats.total_cycles;
    uint64_t mr  = g_hw_stats.total_memory_read;
    uint64_t mw  = g_hw_stats.total_memory_write;
    uint32_t cnt = g_hw_stats.fa_call_count;

    g_hw_stats.total_cycles       = cy  + (uint64_t)ri.elapsed_cycle;
    g_hw_stats.total_memory_read  = mr  + (uint64_t)ri.memory_read;
    g_hw_stats.total_memory_write = mw  + (uint64_t)ri.memory_write;
    g_hw_stats.fa_call_count      = cnt + 1u;
    g_hw_stats.last_call_cycles   = (uint64_t)ri.elapsed_cycle;

#ifdef FA_STATS_DEBUG
    fprintf(stderr,
            "[FA-STATS] call %2u: ri.cycles=%-10llu  accum=%-10llu  "
            "ri.read=%-7u  ri.write=%-7u\n",
            cnt + 1u,
            (unsigned long long)ri.elapsed_cycle,
            (unsigned long long)(cy + ri.elapsed_cycle),
            ri.memory_read,
            ri.memory_write);
#endif
}

/* ── Per-head static buffers for DMA stability ───────────────────────────
   flash_attention() passes these addresses to the HAL's AXI DMA.
   Static allocation ensures they share the same upper-32-bit address region
   as the FlashAttnHAL object (required by the vm_addr_h_ mechanism).
   ─────────────────────────────────────────────────────────────────────── */
static float s_Q_h[N * HD];
static float s_K_h[N * HD];
static float s_V_h[N * HD];
static float s_O_h[N * HD];

/* ── CLS attention helper ────────────────────────────────────────────────────
   Computes attention for a SINGLE query row against T key/value pairs.
   Used to compute the CLS token's attention output in fp32 software, since the
   hardware FA is fixed at N=196 (patch tokens only, N%BR==0 required).

   q    : [HD]        — single query vector
   K    : [T, HD]     — key matrix for all T tokens
   V    : [T, HD]     — value matrix for all T tokens
   out  : [HD]        — output attention vector
   ─────────────────────────────────────────────────────────────────────────── */
static void single_query_attention(const float* q, const float* K, const float* V,
                                   float* out, int tokens, int hd) {
    float scale = 1.0f / sqrtf((float)hd);
    float* scores = (float*)malloc((size_t)tokens * sizeof(float));

    /* scores[t] = dot(q, K[t]) * scale */
    for (int t = 0; t < tokens; t++) {
        float dot = 0.0f;
        for (int d = 0; d < hd; d++)
            dot += q[d] * K[t * hd + d];
        scores[t] = dot * scale;
    }

    /* softmax in-place (numerically stable) */
    float maxval = scores[0];
    for (int t = 1; t < tokens; t++)
        if (scores[t] > maxval) maxval = scores[t];
    float sumexp = 0.0f;
    for (int t = 0; t < tokens; t++) {
        scores[t] = expf(scores[t] - maxval);
        sumexp += scores[t];
    }
    for (int t = 0; t < tokens; t++)
        scores[t] /= sumexp;

    /* out = scores @ V */
    memset(out, 0, (size_t)hd * sizeof(float));
    for (int t = 0; t < tokens; t++)
        for (int d = 0; d < hd; d++)
            out[d] += scores[t] * V[t * hd + d];

    free(scores);
}

/* ── Attention layer ─────────────────────────────────────────────────────── */
void vit_attention_layer(const float* tokens_in, float* attn_out,
                         const ViTWeights* w, int layer, int use_hw) {
    /* Step 1: QKV projection  tokens_in[T,D] → qkv_out[T, 3D] */
    float* qkv_out = (float*)malloc((size_t)T * 3 * D * sizeof(float));
    linear(tokens_in, w->qkv_w[layer], w->qkv_b[layer], qkv_out, T, D, 3*D);

    /* Step 2a: Patch-token multi-head attention via HW FA (or SW reference).
       Patch tokens are rows 1..T-1 (N=196). CLS row 0 is skipped here.
       concat_out[N, D] — merged H-head output for all patch tokens.          */
    float* concat_out = (float*)calloc((size_t)N * D, sizeof(float));

    for (int h = 0; h < H; h++) {
        /* Extract head h Q/K/V for patch tokens.
           qkv_out: [T, 3D]  blocks → Q=[:, 0:D], K=[:, D:2D], V=[:, 2D:3D]
           Head h slice within each D block: columns [h*HD : (h+1)*HD]        */
        for (int n = 0; n < N; n++) {
            int t = n + 1;   /* t=0 is CLS; patch tokens start at t=1 */
            const float* qkv_row = qkv_out + t * 3 * D;
            memcpy(s_Q_h + n * HD, qkv_row +         h * HD, (size_t)HD * sizeof(float));
            memcpy(s_K_h + n * HD, qkv_row + D     + h * HD, (size_t)HD * sizeof(float));
            memcpy(s_V_h + n * HD, qkv_row + 2 * D + h * HD, (size_t)HD * sizeof(float));
        }

        if (use_hw) {
            /* Hardware FlashAttention-2: INT8 DMA + K/V Ping-Pong.
               fa_call_and_accumulate() reads HAL stats before next reset. */
            fa_call_and_accumulate(s_Q_h, s_K_h, s_V_h, s_O_h, N, HD, BR);
        } else {
            /* CPU reference: O(N²d), fp32 */
            standard_attention_cpu(s_Q_h, s_K_h, s_V_h, s_O_h, N, HD);
        }

        /* Merge head h output into concat_out[N, D] */
        for (int n = 0; n < N; n++)
            memcpy(concat_out + n * D + h * HD, s_O_h + n * HD,
                   (size_t)HD * sizeof(float));
    }

    /* Step 2b: CLS-token attention in software.
       Q_cls attends to ALL T tokens (including itself).  This is O(T·HD) per
       head — cheap compared to the HW FA O(N²·HD) — and restores standard
       ViT behaviour for the classification token.
       K_all[T,HD] and V_all[T,HD] are heap-allocated per head to avoid
       large stack frames (T*HD = 197*64 = 12,608 floats ≈ 49 KB).          */
    float* cls_concat = (float*)malloc((size_t)D * sizeof(float));

    for (int h = 0; h < H; h++) {
        const float* q_cls_h = qkv_out + 0 * 3 * D + h * HD;   /* CLS query, head h */

        float* K_all = (float*)malloc((size_t)T * HD * sizeof(float));
        float* V_all = (float*)malloc((size_t)T * HD * sizeof(float));
        for (int t = 0; t < T; t++) {
            const float* row = qkv_out + t * 3 * D;
            memcpy(K_all + t * HD, row + D     + h * HD, (size_t)HD * sizeof(float));
            memcpy(V_all + t * HD, row + 2 * D + h * HD, (size_t)HD * sizeof(float));
        }

        float o_cls_h[HD];
        single_query_attention(q_cls_h, K_all, V_all, o_cls_h, T, HD);
        free(K_all);
        free(V_all);

        memcpy(cls_concat + h * HD, o_cls_h, (size_t)HD * sizeof(float));
    }
    free(qkv_out);

    /* Step 3: Output projection */
    float* patch_attn = (float*)malloc((size_t)N * D * sizeof(float));
    linear(concat_out, w->proj_w[layer], w->proj_b[layer], patch_attn, N, D, D);
    free(concat_out);

    float* cls_attn = (float*)malloc((size_t)D * sizeof(float));
    linear(cls_concat, w->proj_w[layer], w->proj_b[layer], cls_attn, 1, D, D);
    free(cls_concat);

    /* Step 4: Assemble attn_out[T, D]
       Row 0: CLS attention output (computed in SW above).
       Rows 1..T-1: patch attention output (computed by HW FA).              */
    memcpy(attn_out,     cls_attn,   (size_t)D * sizeof(float));
    memcpy(attn_out + D, patch_attn, (size_t)N * D * sizeof(float));
    free(cls_attn);
    free(patch_attn);
}

/* ── One complete transformer block ─────────────────────────────────────── */
static void transformer_block(float* tokens, const ViTWeights* w,
                               int layer, int use_hw) {
    /* --- Pre-attention norm (LayerNorm or RMSNorm via NORM macro) --- */
    float* x_norm = (float*)malloc((size_t)T * D * sizeof(float));
    memcpy(x_norm, tokens, (size_t)T * D * sizeof(float));
    NORM(x_norm, w->norm1_w[layer], w->norm1_b[layer], T, D, EPS);

    /* --- Multi-head attention --- */
    float* attn_out = (float*)malloc((size_t)T * D * sizeof(float));
    vit_attention_layer(x_norm, attn_out, w, layer, use_hw);

    /* --- Residual add: tokens += attn_out --- */
    residual_add(tokens, attn_out, T * D);
    free(attn_out);

    /* --- Pre-MLP norm --- */
    memcpy(x_norm, tokens, (size_t)T * D * sizeof(float));
    NORM(x_norm, w->norm2_w[layer], w->norm2_b[layer], T, D, EPS);

    /* --- MLP: FC1 → GELU → FC2 --- */
    float* mlp_mid = (float*)malloc((size_t)T * M * sizeof(float));
    linear(x_norm, w->mlp_fc1_w[layer], w->mlp_fc1_b[layer], mlp_mid, T, D, M);
    gelu_inplace(mlp_mid, T * M);

    float* mlp_out = (float*)malloc((size_t)T * D * sizeof(float));
    linear(mlp_mid, w->mlp_fc2_w[layer], w->mlp_fc2_b[layer], mlp_out, T, M, D);
    free(mlp_mid);

    /* --- Residual add: tokens += mlp_out --- */
    residual_add(tokens, mlp_out, T * D);
    free(mlp_out);
    free(x_norm);
}

/* ── Top-level ViT inference ─────────────────────────────────────────────── */
int vit_inference(const float* image, float* logits,
                  const ViTWeights* w, int use_hw) {
    /* ── Step 1: Patch Embedding ── */
    float* patches   = (float*)malloc((size_t)N * 768 * sizeof(float));
    float* patch_out = (float*)malloc((size_t)N * D * sizeof(float));

    extract_patches(image, patches);
    linear(patches, w->patch_embed_w, w->patch_embed_b, patch_out, N, 768, D);
    free(patches);

    /* ── Step 2: Prepend CLS + positional embedding → tokens[T, D] ── */
    float* tokens = (float*)malloc((size_t)T * D * sizeof(float));
    prepend_cls_pos(tokens, patch_out, w->cls_token, w->pos_embed);
    free(patch_out);

    /* ── Step 3: 12 × Transformer Block ── */
    for (int layer = 0; layer < VIT_LAYERS; layer++) {
        fprintf(stdout, "[VIT] Block %2d/%d ...\n", layer + 1, VIT_LAYERS);
        fflush(stdout);
        transformer_block(tokens, w, layer, use_hw);
    }

    /* ── Step 4: Final norm ── */
    NORM(tokens, w->norm_w, w->norm_b, T, D, EPS);

    /* ── Step 5: Classification head on CLS token (row 0) ── */
    /* logits[C] = tokens[0, :] × head_w^T + head_b */
    linear(tokens, w->head_w, w->head_b, logits, 1, D, C);

    free(tokens);
    return argmax(logits, C);
}
