/* vit_weights.cpp — Load ViT-Small/16 weight binaries exported by export_weights.py */

#include "vit_weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Load n_floats from <dir>/<name>.bin.  Returns NULL and prints error on failure. */
static float* load_bin(const char* dir, const char* name, size_t n_floats) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s.bin", dir, name);

    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[VIT-W] Cannot open: %s\n", path);
        return NULL;
    }

    float* buf = (float*)malloc(n_floats * sizeof(float));
    if (!buf) {
        fclose(fp);
        fprintf(stderr, "[VIT-W] OOM for: %s\n", path);
        return NULL;
    }

    size_t got = fread(buf, sizeof(float), n_floats, fp);
    fclose(fp);

    if (got != n_floats) {
        fprintf(stderr, "[VIT-W] %s: expected %zu floats, got %zu\n",
                path, n_floats, got);
        free(buf);
        return NULL;
    }
    return buf;
}

/* Convenience macros to shorten dimension expressions */
#define D  VIT_EMBED_DIM   /* 384  */
#define P  VIT_PATCH_DIM   /* 768  */
#define T  VIT_TOKENS      /* 197  */
#define M  VIT_MLP_DIM     /* 1536 */
#define C  VIT_CLASSES     /* 1000 */
#define L  VIT_LAYERS      /* 12   */

ViTWeights* vit_weights_load(const char* dir) {
    ViTWeights* w = (ViTWeights*)calloc(1, sizeof(ViTWeights));
    if (!w) return NULL;

    /* Global tensors */
    w->patch_embed_w = load_bin(dir, "patch_embed_weight", (size_t)D * P);
    w->patch_embed_b = load_bin(dir, "patch_embed_bias",   (size_t)D);
    w->cls_token     = load_bin(dir, "cls_token",          (size_t)D);
    w->pos_embed     = load_bin(dir, "pos_embed",          (size_t)T * D);

    /* Per-block tensors */
    char name[64];
    int ok = 1;
    for (int i = 0; i < L && ok; i++) {
#define LOAD(field, fmt, n) \
        snprintf(name, sizeof(name), fmt, i); \
        w->field[i] = load_bin(dir, name, (size_t)(n)); \
        ok &= (w->field[i] != NULL)

        LOAD(norm1_w,    "block_%02d_norm1_w",   D);
        LOAD(norm1_b,    "block_%02d_norm1_b",   D);
        LOAD(qkv_w,      "block_%02d_qkv_w",     3*D*D);
        LOAD(qkv_b,      "block_%02d_qkv_b",     3*D);
        LOAD(proj_w,     "block_%02d_proj_w",    D*D);
        LOAD(proj_b,     "block_%02d_proj_b",    D);
        LOAD(norm2_w,    "block_%02d_norm2_w",   D);
        LOAD(norm2_b,    "block_%02d_norm2_b",   D);
        LOAD(mlp_fc1_w,  "block_%02d_mlp_fc1_w", M*D);
        LOAD(mlp_fc1_b,  "block_%02d_mlp_fc1_b", M);
        LOAD(mlp_fc2_w,  "block_%02d_mlp_fc2_w", D*M);
        LOAD(mlp_fc2_b,  "block_%02d_mlp_fc2_b", D);
#undef LOAD
    }

    /* Final norm + head */
    w->norm_w = load_bin(dir, "norm_w", (size_t)D);
    w->norm_b = load_bin(dir, "norm_b", (size_t)D);
    w->head_w = load_bin(dir, "head_w", (size_t)C * D);
    w->head_b = load_bin(dir, "head_b", (size_t)C);

    /* Check all non-NULL */
    if (!w->patch_embed_w || !w->patch_embed_b || !w->cls_token || !w->pos_embed ||
        !w->norm_w || !w->norm_b || !w->head_w || !w->head_b || !ok) {
        fprintf(stderr, "[VIT-W] One or more weight files missing in: %s\n", dir);
        vit_weights_free(w);
        return NULL;
    }

    fprintf(stdout, "[VIT-W] Loaded all weights from: %s\n", dir);
    return w;
}

void vit_weights_free(ViTWeights* w) {
    if (!w) return;
    free(w->patch_embed_w);
    free(w->patch_embed_b);
    free(w->cls_token);
    free(w->pos_embed);
    for (int i = 0; i < VIT_LAYERS; i++) {
        free(w->norm1_w[i]);  free(w->norm1_b[i]);
        free(w->qkv_w[i]);    free(w->qkv_b[i]);
        free(w->proj_w[i]);   free(w->proj_b[i]);
        free(w->norm2_w[i]);  free(w->norm2_b[i]);
        free(w->mlp_fc1_w[i]);free(w->mlp_fc1_b[i]);
        free(w->mlp_fc2_w[i]);free(w->mlp_fc2_b[i]);
    }
    free(w->norm_w);
    free(w->norm_b);
    free(w->head_w);
    free(w->head_b);
    free(w);
}
