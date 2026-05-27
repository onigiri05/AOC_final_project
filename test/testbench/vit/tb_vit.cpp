/* tb_vit.cpp — ViT-Small/16 Integration Testbench
 *
 * Mode A (default — no arguments):
 *   Synthetic single-block test.
 *   Generates random tokens[197,384] and random weights for one transformer
 *   block, then runs:
 *     1. SW reference: 6 × standard_attention_cpu()
 *     2. HW version:   6 × flash_attention()
 *   Compares patch-token attention outputs (rows 1..196) and prints stats.
 *   Takes ~4.5M cycles (6 × 747 K) in the Verilator simulation.
 *
 * Mode B (./tb_vit <weights_dir>):
 *   Full 12-block ViT inference with real weights on a synthetic image.
 *   Prints predicted class and top-5 distribution.
 *   WARNING: takes ~54 M cycles in Verilator — use only if you have time.
 *
 * Mode C (./tb_vit <weights_dir> <image.bin> [expected_class]):
 *   Full inference on a real preprocessed image (float32 CHW from
 *   scripts/preprocess_image.py).  Checks top-1 accuracy if class given.
 *
 * Mode D (./tb_vit <weights_dir> --batch <batch_dir> [--hw]):
 *   Batch accuracy test.  Reads all img_NNNN.bin + labels.txt from batch_dir
 *   (created by scripts/batch_preprocess.py) and reports Top-1 / Top-5 accuracy.
 *   Default: use_hw=0 (SW attention, fast).  Pass --hw to use hardware FA.
 *   Example:  ./tb_vit weights/ --batch /tmp/batch_50/
 *             ./tb_vit weights/ --batch /tmp/batch_10/ --hw
 */

#include <dirent.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flash_attn_hal.hpp"
#include "driver_flash_attn.h"
#include "hal.hpp"
#include "runtime.h"

#include "vit_weights.h"
#include "vit_ops.h"
#include "runtime_vit.h"

/* ── Colour helpers (same palette as tb.cpp) ─────────────────────────────── */
#define COL_RESET "\033[0m"
#define COL_GREY  "\033[0;37m"
#define COL_WHITE "\033[1;37m"
#define COL_GREEN "\033[0;32m"
#define COL_RED   "\033[0;31m"
#define COL_CYAN  "\033[0;36m"

#define LOG_INFO(fmt,...) fprintf(stdout, COL_GREY  fmt COL_RESET"\n", ##__VA_ARGS__)
#define LOG_RES(fmt,...)  fprintf(stdout, COL_WHITE fmt COL_RESET"\n", ##__VA_ARGS__)
#define LOG_OK(fmt,...)   fprintf(stdout, COL_GREEN fmt COL_RESET"\n", ##__VA_ARGS__)
#define LOG_ERR(fmt,...)  fprintf(stderr, COL_RED   fmt COL_RESET"\n", ##__VA_ARGS__)
#define LOG_CYN(fmt,...)  fprintf(stdout, COL_CYAN  fmt COL_RESET"\n", ##__VA_ARGS__)

/* Tolerance for HW vs SW attention comparison (INT8 quantisation noise) */
#define ATTN_TOL 5e-2f

/* ── Static HAL — must live in the same 4 GB address region as DMA buffers ─*/
static FlashAttnHAL hal(FA_MMIO_BASE_ADDR, FA_MMIO_SIZE);

/* ── Deterministic data generators ──────────────────────────────────────── */
static void gen_tokens(float* tokens, int T, int D, unsigned seed) {
    for (int i = 0; i < T * D; i++) {
        seed = seed * 1664525u + 1013904223u;  /* LCG */
        tokens[i] = 0.3f * (float)((int)(seed >> 16) - 32768) / 32768.0f;
    }
}

static void gen_weight(float* w, int rows, int cols, unsigned seed) {
    float scale = 0.02f / sqrtf((float)cols);
    for (int i = 0; i < rows * cols; i++) {
        seed = seed * 1664525u + 1013904223u;
        w[i] = scale * (float)((int)(seed >> 16) - 32768) / 32768.0f;
    }
}

static void gen_bias(float* b, int len, unsigned seed) {
    for (int i = 0; i < len; i++) {
        seed = seed * 1664525u + 1013904223u;
        b[i] = 0.001f * (float)((int)(seed >> 16) - 32768) / 32768.0f;
    }
}

/* ── Mode A: single-block synthetic test ─────────────────────────────────── */
static int run_synthetic_test(void) {
    const int T  = VIT_TOKENS;    /* 197 */
    const int N  = VIT_PATCH_N;   /* 196 */
    const int D  = VIT_EMBED_DIM; /* 384 */

    LOG_INFO("[TB/VIT] ========================================");
    LOG_INFO("[TB/VIT] Mode A — Synthetic single-block test");
    LOG_INFO("[TB/VIT] Tokens=%d  Patch=%d  D=%d  Heads=%d  HD=%d",
             T, N, D, VIT_HEADS, VIT_HEAD_DIM);
    LOG_INFO("[TB/VIT] ========================================");

    /* Build a minimal ViTWeights with random data for block 0 only */
    ViTWeights w;
    memset(&w, 0, sizeof(w));

    float* norm1_w  = (float*)malloc(D * sizeof(float));
    float* norm1_b  = (float*)malloc(D * sizeof(float));
    float* qkv_w    = (float*)malloc(3 * D * D * sizeof(float));
    float* qkv_b    = (float*)malloc(3 * D * sizeof(float));
    float* proj_w_  = (float*)malloc(D * D * sizeof(float));
    float* proj_b_  = (float*)malloc(D * sizeof(float));

    /* gamma=1, beta=0 for norm (identity) so output is comparable */
    for (int i = 0; i < D; i++) { norm1_w[i] = 1.0f; norm1_b[i] = 0.0f; }
    gen_weight(qkv_w,   3*D, D, 0x1234);
    gen_bias  (qkv_b,   3*D,    0xABCD);
    gen_weight(proj_w_, D,   D, 0x5678);
    gen_bias  (proj_b_, D,      0xEF01);

    w.norm1_w[0] = norm1_w;  w.norm1_b[0] = norm1_b;
    w.qkv_w  [0] = qkv_w;   w.qkv_b  [0] = qkv_b;
    w.proj_w [0] = proj_w_;  w.proj_b [0] = proj_b_;

    /* Input tokens (static for DMA address stability) */
    static float tokens_sw[VIT_TOKENS * VIT_EMBED_DIM];
    static float tokens_hw[VIT_TOKENS * VIT_EMBED_DIM];
    gen_tokens(tokens_sw, T, D, 0xDEAD);
    memcpy(tokens_hw, tokens_sw, (size_t)T * D * sizeof(float));

    /* Output buffers */
    static float attn_sw[VIT_TOKENS * VIT_EMBED_DIM];
    static float attn_hw[VIT_TOKENS * VIT_EMBED_DIM];

    /* ── 1. SW reference ── */
    LOG_INFO("[TB/VIT] Running SW reference (6 × standard_attention_cpu)...");
    vit_attention_layer(tokens_sw, attn_sw, &w, 0, /*use_hw=*/0);

    /* ── 2. HW version ── */
    LOG_INFO("[TB/VIT] Running HW version (6 × flash_attention)...");
    /* vit_hw_stats_reset() also resets the internal global accumulator in
       runtime_vit.cpp.  Each fa_call_and_accumulate() inside
       vit_attention_layer() will add to it immediately after the HW call. */
    VitHWStats stats;
    vit_hw_stats_reset(&stats);

    vit_attention_layer(tokens_hw, attn_hw, &w, 0, /*use_hw=*/1);

    /* Snapshot accumulated totals from all 6 heads. */
    vit_hw_stats_accumulate(&stats);

    /* ── 3. Compare patch-token rows (rows 1..196; row 0=CLS computed in SW) ── */
    int errors = 0;
    float max_diff = 0.0f;
    int   max_idx  = 0;
    for (int i = D; i < T * D; i++) {   /* skip CLS row (first D elements) */
        float diff = fabsf(attn_hw[i] - attn_sw[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
        if (diff > ATTN_TOL) errors++;
    }

    /* ── 4. Print results ── */
    printf("\n");
    LOG_RES("===== VIT Single-Block Attention Result =====");
    LOG_RES("  Mode              : SW reference vs HW (FlashAttention)");
    LOG_RES("  Tokens (T)        : %d  (incl. CLS)", T);
    LOG_RES("  Patch tokens (N)  : %d  (sent to FA hardware)", N);
    LOG_RES("  Heads             : %d  (serial, 1 HW call each)", VIT_HEADS);
    LOG_RES("  FA calls          : %u", stats.fa_call_count);
    LOG_RES("  Total cycles      : %llu  (expected ~%llu)",
            (unsigned long long)stats.total_cycles,
            (unsigned long long)stats.last_call_cycles * stats.fa_call_count);
    LOG_RES("  Per-call cycles   : %llu  (last call baseline)",
            (unsigned long long)stats.last_call_cycles);
    LOG_RES("  Total DMA reads   : %llu B", (unsigned long long)stats.total_memory_read);
    LOG_RES("  Total DMA writes  : %llu B", (unsigned long long)stats.total_memory_write);
    LOG_RES("  Max abs diff      : %.2e  at elem=%d (patch-token space)",
            max_diff, max_idx - D);
    LOG_RES("  Errors (>%.0e)   : %d  %s",
            (double)ATTN_TOL, errors, errors == 0 ? "[PASS]" : "[FAIL]");
    LOG_RES("=============================================");
    printf("\n");

    if (errors == 0) {
        LOG_OK("[TB/VIT] *** SINGLE-BLOCK TEST PASSED ***");
    } else {
        LOG_ERR("[TB/VIT] *** SINGLE-BLOCK TEST FAILED  errors=%d ***", errors);
    }

    /* Projection to 12-layer full model */
    if (stats.total_cycles > 0) {
        uint64_t est_full = stats.total_cycles * (uint64_t)VIT_LAYERS;
        float    est_ms   = (float)est_full * 5e-6f;  /* 5 ns/cycle */
        LOG_CYN("[TB/VIT] Estimated full-model FA cycles: %llu  (%.1f ms @ 200MHz)",
                (unsigned long long)est_full, est_ms);
    }

    /* Cleanup */
    free(norm1_w); free(norm1_b);
    free(qkv_w);   free(qkv_b);
    free(proj_w_); free(proj_b_);

    return errors == 0 ? 0 : 1;
}

/* ── Mode B/C: full inference with real weights ──────────────────────────── */
static int run_real_inference(const char* weights_dir,
                              const char* image_bin,
                              int expected_class) {
    LOG_INFO("[TB/VIT] ========================================");
    LOG_INFO("[TB/VIT] Mode %s — Full 12-block ViT inference",
             image_bin ? "C" : "B");
    LOG_INFO("[TB/VIT] Weights: %s", weights_dir);
    if (image_bin) LOG_INFO("[TB/VIT] Image:   %s", image_bin);
    LOG_INFO("[TB/VIT] ========================================");
    LOG_INFO("[TB/VIT] WARNING: 12 layers × 6 heads ≈ 54 M FA cycles in Verilator.");
    LOG_INFO("[TB/VIT] This may take several minutes of wall-clock time.");

    /* Load weights */
    ViTWeights* w = vit_weights_load(weights_dir);
    if (!w) {
        LOG_ERR("[TB/VIT] Weight load failed.");
        return 1;
    }

    /* Load or generate image */
    static float image[3 * 224 * 224];

    if (image_bin) {
        FILE* fp = fopen(image_bin, "rb");
        if (!fp) {
            LOG_ERR("[TB/VIT] Cannot open image: %s", image_bin);
            vit_weights_free(w);
            return 1;
        }
        size_t got = fread(image, sizeof(float), 3 * 224 * 224, fp);
        fclose(fp);
        if (got != (size_t)(3 * 224 * 224)) {
            LOG_ERR("[TB/VIT] Image file too small (%zu floats, need %d)", got, 3*224*224);
            vit_weights_free(w);
            return 1;
        }
        LOG_INFO("[TB/VIT] Loaded image from %s", image_bin);
    } else {
        /* Synthetic image: mild sin/cos pattern */
        LOG_INFO("[TB/VIT] Using synthetic image (sin/cos pattern)");
        for (int i = 0; i < 3 * 224 * 224; i++)
            image[i] = 0.5f * sinf((float)i * 0.001f);
    }

    /* Run inference */
    static float logits[VIT_CLASSES];
    LOG_INFO("[TB/VIT] Starting inference...");

    VitHWStats stats;
    vit_hw_stats_reset(&stats);

    int pred = vit_inference(image, logits, w, /*use_hw=*/1);
    vit_hw_stats_accumulate(&stats);

    /* Top-5 */
    int   top5_idx[5];
    float top5_val[5];
    softmax_inplace(logits, VIT_CLASSES);
    topk(logits, VIT_CLASSES, 5, top5_idx, top5_val);

    printf("\n");
    LOG_RES("===== VIT Full-Model Inference Result =====");
    LOG_RES("  Layers            : %d", VIT_LAYERS);
    LOG_RES("  FA calls total    : %u  (%d layers × %d heads)",
            stats.fa_call_count, VIT_LAYERS, VIT_HEADS);
    LOG_RES("  Total FA cycles   : %llu  (expected ~%llu)",
            (unsigned long long)stats.total_cycles,
            (unsigned long long)stats.last_call_cycles * stats.fa_call_count);
    LOG_RES("  Per-call cycles   : %llu  (last call baseline)",
            (unsigned long long)stats.last_call_cycles);
    LOG_RES("  Total DMA reads   : %.1f KB",
            (float)stats.total_memory_read / 1024.0f);
    LOG_RES("  Predicted class   : %d", pred);
    if (expected_class >= 0)
        LOG_RES("  Expected class    : %d  %s",
                expected_class,
                pred == expected_class ? "[CORRECT]" : "[WRONG]");
    LOG_RES("  Top-5 predictions :");
    for (int k = 0; k < 5; k++)
        LOG_RES("    #%d  class=%4d  prob=%.4f%s", k+1, top5_idx[k], top5_val[k],
                (expected_class >= 0 && top5_idx[k] == expected_class) ? " ←" : "");
    LOG_RES("===========================================");
    printf("\n");

    int top1_ok = (expected_class < 0) || (pred == expected_class);
    if (top1_ok) LOG_OK("[TB/VIT] *** INFERENCE DONE%s ***",
                        expected_class >= 0 ? " — TOP-1 CORRECT" : "");
    else         LOG_ERR("[TB/VIT] *** TOP-1 MISMATCH  pred=%d expected=%d ***",
                         pred, expected_class);

    vit_weights_free(w);
    return top1_ok ? 0 : 1;
}

/* ── Mode D: batch accuracy test ─────────────────────────────────────────── */

/* Read labels.txt from batch_dir.  Returns number of labels read (≤ max_n).
   Labels file: one integer per line; -1 means "unknown". */
static int read_labels(const char* batch_dir, int* labels, int max_n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/labels.txt", batch_dir);
    FILE* fp = fopen(path, "r");
    if (!fp) {
        LOG_ERR("[TB/VIT/BATCH] Cannot open %s", path);
        return -1;
    }
    int n = 0;
    while (n < max_n && fscanf(fp, "%d", &labels[n]) == 1)
        n++;
    fclose(fp);
    return n;
}

/* Count img_NNNN.bin files in batch_dir (sequential from 0000). */
static int count_images(const char* batch_dir) {
    char path[1024];
    int n = 0;
    while (1) {
        snprintf(path, sizeof(path), "%s/img_%04d.bin", batch_dir, n);
        FILE* fp = fopen(path, "rb");
        if (!fp) break;
        fclose(fp);
        n++;
    }
    return n;
}

static int run_batch_accuracy(const char* weights_dir,
                               const char* batch_dir,
                               int use_hw) {
    LOG_INFO("[TB/VIT/BATCH] ==========================================");
    LOG_INFO("[TB/VIT/BATCH] Mode D — Batch Accuracy Test");
    LOG_INFO("[TB/VIT/BATCH] Weights : %s", weights_dir);
    LOG_INFO("[TB/VIT/BATCH] Batch   : %s", batch_dir);
    LOG_INFO("[TB/VIT/BATCH] Mode    : %s", use_hw ? "HW (flash_attention)" : "SW (reference)");
    if (use_hw) {
        LOG_INFO("[TB/VIT/BATCH] WARNING: HW mode runs ~54M cycles per image.");
        LOG_INFO("[TB/VIT/BATCH]          Recommend ≤ 10 images for HW validation.");
    }
    LOG_INFO("[TB/VIT/BATCH] ==========================================");

    /* Load weights once */
    ViTWeights* w = vit_weights_load(weights_dir);
    if (!w) { LOG_ERR("[TB/VIT/BATCH] Weight load failed."); return 1; }

    /* Discover image count */
    int n_images = count_images(batch_dir);
    if (n_images == 0) {
        LOG_ERR("[TB/VIT/BATCH] No img_NNNN.bin files found in %s", batch_dir);
        vit_weights_free(w);
        return 1;
    }
    LOG_INFO("[TB/VIT/BATCH] Found %d images.", n_images);

    /* Read ground-truth labels */
    int* labels = (int*)malloc((size_t)n_images * sizeof(int));
    int n_labels = read_labels(batch_dir, labels, n_images);
    if (n_labels < 0) {
        LOG_ERR("[TB/VIT/BATCH] labels.txt missing — run batch_preprocess.py first.");
        free(labels);
        vit_weights_free(w);
        return 1;
    }
    /* Use the smaller of the two counts */
    if (n_labels < n_images) n_images = n_labels;

    int top1_correct = 0;
    int top5_correct = 0;
    int known_labels = 0;  /* images with label >= 0 */

    static float image[3 * 224 * 224];
    static float logits[VIT_CLASSES];

    printf("\n");
    for (int i = 0; i < n_images; i++) {
        /* Load image binary */
        char img_path[1024];
        snprintf(img_path, sizeof(img_path), "%s/img_%04d.bin", batch_dir, i);
        FILE* fp = fopen(img_path, "rb");
        if (!fp) {
            LOG_ERR("[TB/VIT/BATCH] Cannot open %s — skipping.", img_path);
            continue;
        }
        size_t got = fread(image, sizeof(float), 3 * 224 * 224, fp);
        fclose(fp);
        if (got != (size_t)(3 * 224 * 224)) {
            LOG_ERR("[TB/VIT/BATCH] %s too small (%zu floats) — skipping.", img_path, got);
            continue;
        }

        /* Inference */
        int pred = vit_inference(image, logits, w, use_hw);

        /* Top-5 */
        int   top5_idx[5];
        float top5_val[5];
        float logits_copy[VIT_CLASSES];
        memcpy(logits_copy, logits, sizeof(logits));
        softmax_inplace(logits_copy, VIT_CLASSES);
        topk(logits_copy, VIT_CLASSES, 5, top5_idx, top5_val);

        int gt = labels[i];
        if (gt >= 0) {
            known_labels++;
            int is_top1 = (pred == gt) ? 1 : 0;
            int is_top5 = 0;
            for (int k = 0; k < 5; k++)
                if (top5_idx[k] == gt) { is_top5 = 1; break; }

            top1_correct += is_top1;
            top5_correct += is_top5;

            LOG_INFO("[%3d/%d] gt=%-4d pred=%-4d top1=%s top5=%s  p=%.4f",
                     i+1, n_images, gt, pred,
                     is_top1 ? "OK" : "--",
                     is_top5 ? "OK" : "--",
                     top5_val[0]);
        } else {
            LOG_INFO("[%3d/%d] pred=%-4d (no gt label)  p=%.4f",
                     i+1, n_images, pred, top5_val[0]);
        }
    }

    /* Summary */
    printf("\n");
    LOG_RES("===== Batch Accuracy Summary =====");
    LOG_RES("  Images processed  : %d", n_images);
    LOG_RES("  Images with label : %d", known_labels);
    LOG_RES("  Mode              : %s", use_hw ? "HW (FlashAttention)" : "SW (reference)");
    if (known_labels > 0) {
        float top1_pct = 100.0f * top1_correct / known_labels;
        float top5_pct = 100.0f * top5_correct / known_labels;
        LOG_RES("  Top-1 correct     : %d / %d  (%.1f%%)", top1_correct, known_labels, top1_pct);
        LOG_RES("  Top-5 correct     : %d / %d  (%.1f%%)", top5_correct, known_labels, top5_pct);
        LOG_RES("  Reference (timm)  : Top-1 ~79.8%%  Top-5 ~94.9%%");

        int pass = (top1_pct >= 70.0f);
        if (pass) LOG_OK("[TB/VIT/BATCH] *** ACCURACY PASS  (Top-1 %.1f%% >= 70%%) ***", top1_pct);
        else       LOG_ERR("[TB/VIT/BATCH] *** ACCURACY LOW  (Top-1 %.1f%% < 70%%) ***", top1_pct);
    } else {
        LOG_RES("  (No ground-truth labels — cannot compute accuracy)");
    }
    LOG_RES("==================================");
    printf("\n");

    free(labels);
    vit_weights_free(w);
    return 0;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(int argc, char** argv) {
    /* Initialise FA hardware HAL */
    set_fa_hal(&hal);
    hal.init();

    int rc;
    if (argc == 1) {
        /* Mode A: synthetic single-block */
        rc = run_synthetic_test();
    } else if (argc >= 4 && strcmp(argv[2], "--batch") == 0) {
        /* Mode D: batch accuracy test
         *   argv[1] = weights_dir
         *   argv[2] = "--batch"
         *   argv[3] = batch_dir
         *   argv[4] = "--hw"  (optional; default SW)        */
        int use_hw = (argc >= 5 && strcmp(argv[4], "--hw") == 0) ? 1 : 0;
        rc = run_batch_accuracy(argv[1], argv[3], use_hw);
    } else {
        /* Mode B/C: real weights */
        const char* weights_dir  = argv[1];
        const char* image_bin    = (argc >= 3) ? argv[2] : NULL;
        int         expected_cls = (argc >= 4) ? atoi(argv[3]) : -1;
        rc = run_real_inference(weights_dir, image_bin, expected_cls);
    }

    hal.final();
    return rc;
}
