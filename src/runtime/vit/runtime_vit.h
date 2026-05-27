#ifndef RUNTIME_VIT_H
#define RUNTIME_VIT_H

#include "vit_weights.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * vit_inference — Run a full ViT-Small/16 forward pass.
 *
 * Parameters:
 *   image   : [3, 224, 224] float32, channel-first, ImageNet-normalized
 *   logits  : [1000] float32 output (caller allocates)
 *   w       : pre-loaded weights from vit_weights_load()
 *   use_hw  : non-zero = use FA hardware accelerator for attention
 *             zero    = CPU-only (standard_attention_cpu for all heads)
 *
 * Returns: predicted class index (0..999)
 *
 * Notes:
 *   - CLS token (row 0) is excluded from hardware FA (N=196, not 197).
 *     CLS attention (Q_cls attending to all T=197 tokens) is computed in
 *     software (fp32, O(T·HD) per head) to satisfy the N%BR==0 constraint.
 *     Patch tokens (N=196) are handled by the hardware accelerator.
 *   - All non-attention ops (LayerNorm, QKV/O proj, MLP) run on CPU (fp32).
 *   - Hardware FA must be initialised before calling this function:
 *       static FlashAttnHAL hal(...); set_fa_hal(&hal); hal.init();
 */
int vit_inference(const float* image, float* logits,
                  const ViTWeights* w, int use_hw);

/*
 * vit_attention_layer — Run one multi-head attention block (no norms, no residual).
 * Useful for testing individual layers in isolation.
 *
 * tokens_in  : [VIT_TOKENS, VIT_EMBED_DIM]  input (already layer-normed)
 * attn_out   : [VIT_TOKENS, VIT_EMBED_DIM]  output (caller allocates)
 * w          : weights
 * layer      : 0..11
 * use_hw     : same as vit_inference
 */
void vit_attention_layer(const float* tokens_in, float* attn_out,
                         const ViTWeights* w, int layer, int use_hw);

/* Cumulative HW stats across all flash_attention() calls since last reset.
 *
 * total_*  : accumulated across ALL calls since vit_hw_stats_reset().
 * last_call_cycles : cycles for the MOST RECENT single flash_attention() call.
 *                   Use as a per-call baseline; should equal ~747K for ViT-S/16.
 * fa_call_count    : number of flash_attention() calls since last reset.
 *
 * Cross-check: total_cycles  ≈  fa_call_count × last_call_cycles
 */
typedef struct {
    uint64_t total_cycles;
    uint64_t total_memory_read;
    uint64_t total_memory_write;
    uint32_t fa_call_count;
    uint64_t last_call_cycles;    /* single-call reference */
} VitHWStats;

/* Reset accumulator. */
void vit_hw_stats_reset(VitHWStats* s);

/* Merge latest HAL runtime_info into accumulator. */
void vit_hw_stats_accumulate(VitHWStats* s);

#ifdef __cplusplus
}
#endif

#endif /* RUNTIME_VIT_H */
