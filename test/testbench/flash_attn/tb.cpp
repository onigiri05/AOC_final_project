/* tb.cpp — FlashAttention Accelerator Testbench
 *
 * For each case:
 *   1. Generate Q, K, V deterministically (sin/cos pattern).
 *   2. Run CPU reference: standard_attention_cpu() → O_ref[].
 *   3. Run hardware simulation: flash_attention() → O_hw[].
 *   4. Compare element-wise with tolerance FLOAT_TOL.
 *   5. Print cycles, bandwidth, and PASS/FAIL.
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "flash_attn_hal.hpp"
#include "driver_flash_attn.h"
#include "hal.hpp"
#include "runtime.h"

#if   CASE_NUM == 0
#include "../../../test/cases/case0/workload.h"
#elif CASE_NUM == 1
#include "../../../test/cases/case1/workload.h"
#elif CASE_NUM == 2
#include "../../../test/cases/case2/workload.h"
#else
#error "CASE_NUM must be defined to 0, 1, or 2"
#endif

#define COL_RESET "\033[0m"
#define COL_GREY  "\033[0;37m"
#define COL_WHITE "\033[1;37m"
#define COL_GREEN "\033[0;32m"
#define COL_RED   "\033[0;31m"
#define COL_CYAN  "\033[0;36m"

#define LOG_INFO(fmt, ...) fprintf(stdout, COL_GREY  fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_OK(fmt, ...)   fprintf(stdout, COL_GREEN fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_ERR(fmt, ...)  fprintf(stderr, COL_RED   fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_RES(fmt, ...)  fprintf(stdout, COL_WHITE fmt COL_RESET "\n", ##__VA_ARGS__)
#define LOG_CYN(fmt, ...)  fprintf(stdout, COL_CYAN  fmt COL_RESET "\n", ##__VA_ARGS__)

/* INT8 symmetric quantisation introduces ~1-5% absolute error in O.
 * Tolerance 5e-2 (5%) covers the worst-case rounding through Q/K/V/P chains. */
#define FLOAT_TOL 5e-2f

/* Static placement ensures upper 32b matches HAL's vm_addr_h_ */
static float Q_data [CASE_N * CASE_D];
static float K_data [CASE_N * CASE_D];
static float V_data [CASE_N * CASE_D];
static float O_hw   [CASE_N * CASE_D];  // hardware output
static float O_ref  [CASE_N * CASE_D];  // CPU reference output

static void gen_qkv(int N, int d) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            int idx = i * d + j;
            /* Deterministic but non-trivial pattern that exercises all rows */
            Q_data[idx] = 0.3f * sinf((float)(i * d + j) * 0.17f);
            K_data[idx] = 0.3f * cosf((float)(i * d + j) * 0.13f + 0.5f);
            V_data[idx] = 0.2f * sinf((float)(j * N + i) * 0.19f + 1.0f);
        }
    }
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;

    const int  N  = CASE_N;
    const int  d  = CASE_D;
    const int  Br = CASE_BR;
    char label[64];
    snprintf(label, sizeof(label), "case%d  (N=%d d=%d Br=%d)", CASE_NUM, N, d, Br);

    LOG_INFO("[TB/FA] Starting FlashAttention simulation — %s", label);

    /* ── 1. Generate input matrices ────────────────────────────────────── */
    gen_qkv(N, d);

    /* ── 2. CPU reference ──────────────────────────────────────────────── */
    LOG_INFO("[TB/FA] Computing CPU reference...");
    standard_attention_cpu(Q_data, K_data, V_data, O_ref, N, d);

    /* ── 3. Hardware simulation ────────────────────────────────────────── */
    LOG_INFO("[TB/FA] Running hardware simulation...");

    /* Static HAL: vm_addr_h_ upper-32b must match the data arrays above */
    static FlashAttnHAL hal(FA_MMIO_BASE_ADDR, FA_MMIO_SIZE);
    set_fa_hal(&hal);
    hal.init();

    memset(O_hw, 0, sizeof(O_hw));
    flash_attention(Q_data, K_data, V_data, O_hw, N, d, Br);

    struct runtime_info ri = hal.get_runtime_info();
    hal.final();

    /* ── 4. Verification ───────────────────────────────────────────────── */
    int err = 0;
    float max_diff = 0.0f;
    int   max_idx  = 0;

    for (int i = 0; i < N * d; i++) {
        float diff = fabsf(O_hw[i] - O_ref[i]);
        if (diff > max_diff) { max_diff = diff; max_idx = i; }
        if (diff > FLOAT_TOL) {
            if (err < 10)
                LOG_ERR("[ERR] idx=%4d  hw=%.6f  ref=%.6f  diff=%.2e",
                        i, O_hw[i], O_ref[i], diff);
            err++;
        }
    }

    /* ── 5. Results ────────────────────────────────────────────────────── */
    printf("\n");
    LOG_RES("===== FlashAttention Simulation Result =====");
    LOG_RES("  Case              : %s", label);
    LOG_RES("  Cycles            : %llu",
            (unsigned long long)ri.elapsed_cycle);
    LOG_RES("  Time (s)          : %.6f",
            (float)ri.elapsed_time / 1e9f);
    LOG_RES("  Mem reads  (Bytes): %u  [%.1f KB]",
            ri.memory_read,  ri.memory_read  / 1024.0f);
    LOG_RES("  Mem writes (Bytes): %u  [%.1f KB]",
            ri.memory_write, ri.memory_write / 1024.0f);
    LOG_RES("  Expected reads    : %u  (3×N×d×1 byte INT8-packed)",
            3 * N * d * 1);
    LOG_RES("  Expected writes   : %u  (1×N×d×4 bytes)",
            1 * N * d * 4);
    LOG_RES("  Max abs diff      : %.2e  at idx=%d",
            max_diff, max_idx);
    LOG_RES("  Errors (>%.0e)    : %d  %s",
            (double)FLOAT_TOL, err, err == 0 ? "[PASS]" : "[FAIL]");
    LOG_RES("=============================================");

    if (err == 0) {
        LOG_OK("[TB/FA] *** TEST PASSED  (cycles=%llu) ***",
               (unsigned long long)ri.elapsed_cycle);
    } else {
        LOG_ERR("[TB/FA] *** FAILED  err=%d ***", err);
    }
    printf("\n");

    /* Bandwidth analysis */
    if (ri.elapsed_cycle > 0) {
        float bw_rd = (float)ri.memory_read  / ((float)ri.elapsed_time * 1e-9f) / 1e9f;
        float bw_wr = (float)ri.memory_write / ((float)ri.elapsed_time * 1e-9f) / 1e9f;
        LOG_CYN("[TB/FA] DMA bandwidth: read %.3f GB/s  write %.3f GB/s", bw_rd, bw_wr);
        /* Compute utilization: each tile needs Br*Br*d MACs for QK^T + Br*Br*d for O */
        long long mac_total = (long long)(N/Br) * (N/Br) * 2LL * Br * Br * d;
        float mac_per_cycle = (float)mac_total / (float)ri.elapsed_cycle;
        LOG_CYN("[TB/FA] Estimated MACs: %lld  (%.2f MAC/cycle)", mac_total, mac_per_cycle);
        printf("\n");
    }

    return err == 0 ? 0 : 1;
}
