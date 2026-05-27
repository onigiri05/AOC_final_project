#ifndef VIT_WEIGHTS_H
#define VIT_WEIGHTS_H

#include <stdint.h>

/* ── ViT-Small/16 architecture constants ─────────────────────────────────── */
#define VIT_LAYERS    12    /* transformer blocks                             */
#define VIT_EMBED_DIM 384   /* token embedding dimension                      */
#define VIT_HEADS     6     /* attention heads per block                      */
#define VIT_HEAD_DIM  64    /* head dimension = EMBED_DIM / HEADS             */
#define VIT_MLP_DIM   1536  /* MLP hidden dim = 4 × EMBED_DIM                */
#define VIT_PATCH_N   196   /* 14×14 patch tokens (CLS excluded for FA HW)   */
#define VIT_TOKENS    197   /* 196 patches + 1 CLS token                     */
#define VIT_PATCH_DIM 768   /* 16×16×3 flattened patch size                  */
#define VIT_CLASSES   1000  /* ImageNet-1k output classes                    */
#define VIT_TILE_BR   14    /* FA tile size = systolic array side length      */

/* ── Weight layout notes ─────────────────────────────────────────────────
   All weights are stored row-major float32, matching PyTorch's convention:
     Linear(in, out).weight shape = [out, in]
   So: output = input × weight^T + bias
   ─────────────────────────────────────────────────────────────────────── */

typedef struct {
    /* Patch embedding */
    float* patch_embed_w;       /* [384, 768]           */
    float* patch_embed_b;       /* [384]                */
    float* cls_token;           /* [384]                */
    float* pos_embed;           /* [197, 384]           */

    /* Per-block weights (index 0..11) */
    float* norm1_w [VIT_LAYERS]; /* [384] LayerNorm gamma before attention */
    float* norm1_b [VIT_LAYERS]; /* [384] LayerNorm beta                  */
    float* qkv_w   [VIT_LAYERS]; /* [1152, 384] fused Q/K/V projection    */
    float* qkv_b   [VIT_LAYERS]; /* [1152]                                */
    float* proj_w  [VIT_LAYERS]; /* [384, 384]  attention output proj      */
    float* proj_b  [VIT_LAYERS]; /* [384]                                 */
    float* norm2_w [VIT_LAYERS]; /* [384] LayerNorm gamma before MLP      */
    float* norm2_b [VIT_LAYERS]; /* [384] LayerNorm beta                  */
    float* mlp_fc1_w[VIT_LAYERS];/* [1536, 384] MLP first linear          */
    float* mlp_fc1_b[VIT_LAYERS];/* [1536]                                */
    float* mlp_fc2_w[VIT_LAYERS];/* [384, 1536] MLP second linear         */
    float* mlp_fc2_b[VIT_LAYERS];/* [384]                                 */

    /* Final LayerNorm + classification head */
    float* norm_w;              /* [384]        */
    float* norm_b;              /* [384]        */
    float* head_w;              /* [1000, 384]  */
    float* head_b;              /* [1000]       */
} ViTWeights;

#ifdef __cplusplus
extern "C" {
#endif

/* Load all weights from binary files in dir.  Returns NULL on error. */
ViTWeights* vit_weights_load(const char* dir);

/* Free all allocated weight memory. */
void vit_weights_free(ViTWeights* w);

#ifdef __cplusplus
}
#endif

#endif /* VIT_WEIGHTS_H */
