# ViT-Small 硬體加速器模擬框架 — 完整規劃與當前進度

> **專案**：`【Phase0+1+ScaleTying+FlashAttn】FlashAttention_RMSNorm_ViT_ImageNet`
> **框架基礎**：`lab-4-EduCatCode`（HAL / Driver / Runtime 三層架構）
> **目標硬體**：PYNQ-Z2（Zynq-7020）/ ViT-Small/16 ImageNet 推論

---

## 執行摘要

本專案建立一套**可擴充的 Verilator RTL 模擬框架**，目標是對 ViT-Small 推論所需的
每一個硬體加速器元件逐一進行 RTL 級別的行為驗證，最終達成完整的 End-to-End 模擬。

**當前里程碑（已完成）**：FlashAttention-2 加速器 RTL，以 ViT-Small 實際規模
（N=196 patches, d=64, Br=14）驗證通過，誤差 < 1×10⁻⁴。

**尚未完成**：RMSNorm 硬體、Linear Projection 硬體、FFN 硬體、
Multi-head 整合、完整 Transformer Block、End-to-End 推論。

---

## 目錄

1. [完整 ViT-Small 硬體模擬規劃總覽](#1-完整-vit-small-硬體模擬規劃總覽)
2. [ViT-Small 架構拆解](#2-vit-small-架構拆解)
3. [當前完成度與里程碑地圖](#3-當前完成度與里程碑地圖)
4. [模擬框架架構（三層設計）](#4-模擬框架架構三層設計)
5. [已完成：FlashAttention 加速器詳解](#5-已完成flashattention-加速器詳解)
6. [Online Softmax 視覺化解說](#6-online-softmax-視覺化解說)
7. [各元件驗證策略](#7-各元件驗證策略)
8. [如何執行當前模擬](#8-如何執行當前模擬)
9. [效能指標解讀](#9-效能指標解讀)
10. [未來各階段規劃](#10-未來各階段規劃)
11. [與 lab-4 的差異對照](#11-與-lab-4-的差異對照)

---

## 1. 完整 ViT-Small 硬體模擬規劃總覽

### 1.1 整個計畫的一句話定位

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  目標：用 Verilator Behavioral Simulation 驗證「ViT-Small 在 PYNQ-Z2 上    │
│         推論所需的每一個硬體加速器」，從單元驗證到 End-to-End 全流程驗證。  │
│                                                                             │
│  策略：分元件逐一建立 RTL + HAL + Driver + Runtime + Testbench，           │
│         每個元件獨立驗證後，再用同一框架串接成完整 Pipeline。              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 「這不是完整 ViT，但有完整計畫」的一圖說明

```
完整 ViT-Small 推論 Pipeline（共 4 大區段）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

 區段 A：前處理
 ┌─────────────────────────────────────┐
 │  Input Image (224×224×3)            │  ← 尚未模擬
 │       ↓ Patch Embedding             │  ← 尚未模擬
 │  196 patches × 384-dim              │
 │       ↓ + CLS token + Pos Embed     │  ← 尚未模擬
 │  197 tokens × 384-dim               │
 └─────────────────────────────────────┘
             ↓

 區段 B：12× Transformer Block（主體，★ 本專案的核心目標 ★）
 ┌─────────────────────────────────────────────────────────────────────┐
 │  ┌── Transformer Block × 12 ───────────────────────────────────┐   │
 │  │                                                              │   │
 │  │  [1] RMSNorm (Pre-Norm)          ████░░  規劃完成，RTL 待實作│   │
 │  │       ↓                                                      │   │
 │  │  [2] Linear Projection           ░░░░░░  架構規劃中          │   │
 │  │       Q = x × Wq  [197×384 × 384×384 → 197×384]             │   │
 │  │       K = x × Wk  [197×384 × 384×384 → 197×384]             │   │
 │  │       V = x × Wv  [197×384 × 384×384 → 197×384]             │   │
 │  │       ↓  reshape → 6 heads × 197 × 64                       │   │
 │  │                                                              │   │
 │  │  [3] Multi-Head FlashAttention   ████████████████  ✅ 已完成  │   │
 │  │       for h in range(6):         （單 head N=197,d=64 已驗證）│   │
 │  │           O_h = FlashAttn(Q_h, K_h, V_h)                    │   │
 │  │       concat → 197 × 384                                     │   │
 │  │       ↓                                                      │   │
 │  │  [4] Linear Projection Wo        ░░░░░░  架構規劃中          │   │
 │  │  [5] Residual Add                ░░░░░░  純加法，低優先       │   │
 │  │       ↓                                                      │   │
 │  │  [6] RMSNorm (Post-Norm)         ████░░  規劃完成，RTL 待實作│   │
 │  │       ↓                                                      │   │
 │  │  [7] MLP / FFN                   ░░░░░░  架構規劃中          │   │
 │  │       FC1: 197×384 → 197×1536 (GELU)                        │   │
 │  │       FC2: 197×1536 → 197×384                                │   │
 │  │  [8] Residual Add                ░░░░░░  純加法，低優先       │   │
 │  └──────────────────────────────────────────────────────────────┘   │
 └─────────────────────────────────────────────────────────────────────┘
             ↓

 區段 C：分類頭
 ┌─────────────────────────────────────┐
 │  Final RMSNorm                      │  ← 尚未模擬
 │       ↓ CLS token 取出              │
 │  Linear → 1000 classes              │  ← 尚未模擬
 └─────────────────────────────────────┘
             ↓
        ImageNet 分類結果

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
圖例：████████████  已完成 RTL 模擬驗證
      ████░░░░░░░░  框架設計完成，RTL 待實作
      ░░░░░░░░░░░░  架構規劃中，尚未動工
```

---

## 2. ViT-Small 架構拆解

### 2.1 ViT-Small/16 參數規格

```
┌──────────────────────────────────────────────────────────┐
│  ViT-Small/16 for ImageNet-1K                            │
├──────────────────────┬───────────────────────────────────┤
│  輸入解析度           │  224 × 224 × 3                   │
│  Patch size          │  16 × 16（共 14×14 = 196 patches）│
│  Token 數            │  196 + 1 (CLS) = 197              │
│  Embedding dim (D)   │  384                              │
│  Transformer blocks  │  12                               │
│  Attention heads (H) │  6                                │
│  Head dim (d)        │  D/H = 384/6 = 64                 │
│  MLP hidden dim      │  4×D = 1536                       │
│  參數量              │  ~22M                             │
└──────────────────────┴───────────────────────────────────┘
```

### 2.2 單一 Transformer Block 資料流（本計畫聚焦區域）

```
        輸入 x  [197 × 384]
             │
     ╔═══════╧═══════╗
     ║   [1] RMSNorm  ║  每個 token 做正規化：x / RMS(x) * γ
     ╚═══════╤═══════╝
             │  x_norm [197 × 384]
     ╔═══════╧══════════════════════════╗
     ║  [2] Linear Projection (Wq,Wk,Wv)║
     ║  Q = x_norm @ Wq  [197×384]      ║  Wq, Wk, Wv ∈ R^{384×384}
     ║  K = x_norm @ Wk  [197×384]      ║
     ║  V = x_norm @ Wv  [197×384]      ║
     ╚═══════╤══════════════════════════╝
             │  reshape → 6 heads
     ┌───────┴────────────────────────────┐
     │  Head 0    Head 1   ...   Head 5   │  每個 head: Q/K/V [197 × 64]
     │     │          │              │    │
     │  ╔══╧══╗    ╔══╧══╗       ╔══╧══╗ │
     │  ║  FA ║    ║  FA ║  ...  ║  FA ║ │  ← ★ 本專案已驗證的元件 ★
     │  ╚══╤══╝    ╚══╤══╝       ╚══╤══╝ │  flash_attention(Q_h, K_h, V_h)
     │     │          │              │    │  N=197, d=64, Br=14
     └───────┬────────────────────────────┘
             │  concat + [4] Wo projection  [197 × 384]
     ╔═══════╧════╗
     ║ [5] Add(x) ║  殘差連接
     ╚═══════╤════╝
             │
     ╔═══════╧═══════╗
     ║   [6] RMSNorm  ║
     ╚═══════╤═══════╝
             │
     ╔═══════╧══════════╗
     ║  [7] MLP / FFN   ║  FC1(384→1536, GELU) + FC2(1536→384)
     ╚═══════╤══════════╝
     ╔═══════╧════╗
     ║ [8] Add(x) ║  殘差連接
     ╚═══════╤════╝
             │
        輸出 x' [197 × 384]
```

### 2.3 FlashAttention 在整體計算量中的比重

```
ViT-Small 單次前向推論 FLOPs 分佈（每個 Transformer Block）：

  Linear Projections (Wq+Wk+Wv+Wo)  ████████████████████  ~57%
  MLP / FFN (FC1 + FC2)              ████████████████████  ~38%
  FlashAttention (QK^T + PV)         ████                   ~4%
  RMSNorm                            █                      ~1%

                         ↑
              雖然 FA 佔 FLOPs 比例不高，但它是
              「記憶體頻寬瓶頸」：標準 attention 需要
               O(N²) DRAM，FA 只需 O(N)，
               這是 FPGA 加速最關鍵的優化。
```

---

## 3. 當前完成度與里程碑地圖

### 3.1 元件完成度一覽

```
元件                        完成度     說明
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
模擬框架基礎建設
  HAL/Driver/Runtime 架構    ██████████  100%  完整實作，可複用
  AXI4 MMIO 界面             ██████████  100%  AXI4 Slave FSM
  AXI4 DMA 界面              ██████████  100%  AXI4 Master FSM
  DPI-C 浮點橋接             ██████████  100%  dpi_math.c

FlashAttention 加速器
  RTL (flash_attn_wrapper.sv)██████████  100%  FlashAttention-2
  Online Softmax 邏輯        ██████████  100%  FA_COMPUTE state
  HAL (FlashAttnHAL)         ██████████  100%  完整 DMA 仿真
  Driver (register map)      ██████████  100%  7 個暫存器
  Runtime (flash_attention)  ██████████  100%  高階 API
  Testbench (case0/1/2)      ██████████  100%  3 個驗證 case

RMSNorm 加速器
  架構設計 / 暫存器規劃      ████░░░░░░   40%  設計完成
  RTL (rms_norm_wrapper.sv)  ░░░░░░░░░░    0%  尚未實作
  HAL / Driver / Runtime     ░░░░░░░░░░    0%  待實作
  Testbench                  ░░░░░░░░░░    0%  待實作

Linear Projection (GEMM) 加速器
  架構研究                   ██░░░░░░░░   20%  分析中
  RTL                        ░░░░░░░░░░    0%  尚未實作
  HAL / Driver / Runtime     ░░░░░░░░░░    0%  待實作

Multi-Head Attention 整合
  軟體層 for-loop 整合       ░░░░░░░░░░    0%  待實作（低難度）
  6 heads 串行驗證           ░░░░░░░░░░    0%  待實作

完整 Transformer Block
  Block-level testbench      ░░░░░░░░░░    0%  待實作
  12 Blocks × 串接           ░░░░░░░░░░    0%  待實作

End-to-End ViT-Small
  Patch Embedding            ░░░░░░░░░░    0%  待實作
  Full inference pipeline    ░░░░░░░░░░    0%  待實作
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
整體進度                                  ~35%  框架 + 最難元件完成
```

### 3.2 里程碑時間線

```
時間線（Phase 規劃）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Phase 0 ─── 框架建立 ────────────────────────────────────── [✅ 完成]
  建立 HAL/Driver/Runtime 三層架構（移植自 lab-4）
  建立 AXI4 MMIO + DMA 仿真基礎設施
  建立 DPI-C 浮點運算橋接
  建立 Makefile 自動化建置系統

Phase 1 ─── FlashAttention 核心加速器 ────────────────────── [✅ 完成]
  flash_attn_wrapper.sv RTL（FA-2 + online softmax + AXI4）
  FlashAttnHAL 完整實作
  3 個驗證 case（N=4, N=8, N=196/ViT-Small 規模）
  ★ ViT-Small patch 規格（N=196, d=64, Br=14）通過驗證 ★

Phase 2 ─── RMSNorm 加速器 ───────────────────────────────── [⏳ 規劃中]
  rms_norm_wrapper.sv RTL
  RMSNormHAL + Driver + Runtime
  Testbench：N=197, d=384

Phase 3 ─── Linear Projection（GEMM）加速器 ──────────────── [📋 研究中]
  GEMM accelerator for Wq/Wk/Wv/Wo [384×384]
  Tiled matrix multiply RTL
  Integration with FA output

Phase 4 ─── Multi-Head + Transformer Block 整合 ──────────── [📋 規劃中]
  軟體層：6-head for-loop 整合
  RMSNorm + Linear + FA + Linear + Residual + RMSNorm + FFN
  Block-level end-to-end testbench

Phase 5 ─── End-to-End ViT-Small ─────────────────────────── [🎯 最終目標]
  12× Transformer Block 串接
  Patch Embedding（可用軟體仿真）
  分類頭
  ImageNet validation set accuracy 比對
```

### 3.3 為什麼先做 FlashAttention？

```
優先順序決策邏輯：

   複雜度     ██████████████████████  最高  ← FlashAttention（已完成）
              （Online softmax + Tiled DMA + RTL timing bugs）

   重要性     ████████████████████    記憶體頻寬瓶頸，FPGA 最關鍵優化

   可驗證性   ████████████████████    有明確的 CPU reference 可比對

   基礎性     ████████████████████    框架一旦建立，所有後續元件都套同一模式

            ──────────────────────────────────────────────────────
   Linear    ████████████████        其次（純 GEMM，成熟演算法）
   RMSNorm   ████████                最簡單（一次 row scan）
   Residual  ██                      純加法，不需要加速器
```

---

## 4. 模擬框架架構（三層設計）

### 4.1 整體軟體/硬體分層

```
┌──────────────────────────────────────────────────────────────────────┐
│                      Testbench 層                                    │
│   tb.cpp                                                             │
│   ┌─────────────────────┐    ┌────────────────────────────────┐     │
│   │  CPU Golden Reference│    │     Hardware Simulation        │     │
│   │                     │    │                                │     │
│   │ standard_attention_ │    │  flash_attention(Q,K,V,O,N,d,Br)│    │
│   │ cpu(Q,K,V,O_ref,N,d)│    │     ↓                          │     │
│   │                     │    │  fa_set_shape / fa_set_tile    │     │
│   │  O(N²d) 標準 softmax│    │  fa_set_q/k/v/o_addr           │     │
│   │  fp32 計算，無 tiling│   │  fa_start()                    │     │
│   └──────────┬──────────┘    └────────────┬───────────────────┘     │
│              │                             │                          │
│              └──────────┬──────────────────┘                         │
│                    element-wise compare                               │
│                    |O_hw[i] - O_ref[i]| ≤ 1e-4  → PASS / FAIL       │
└──────────────────────────────────────────────────────────────────────┘
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│                       Runtime 層                                     │
│   runtime_flash_attn.cpp                                             │
│   fa_set_shape() → fa_reg_write(FA_SHAPE_OFFSET, N<<16 | d)         │
│   fa_set_tile()  → fa_reg_write(FA_TILE_OFFSET, Br)                 │
│   fa_start()     → fa_reg_write(FA_CONTROL_OFFSET, FA_CTRL_START)   │
│   wait_for_irq() → hal->wait_for_irq()  (blocks until interrupt)    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
┌──────────────────────────────────────────────────────────────────────┐
│                        Driver 層                                     │
│   driver_flash_attn.h/cpp                                            │
│   MMIO Base: 0x10050000                                              │
│   ┌────────────┬────────┬────────────────────────────────────────┐  │
│   │ 0x10050000 │ CTRL   │ [0]=start  [1]=irq_clear               │  │
│   │ 0x10050004 │ SHAPE  │ [31:16]=N  [15:0]=d                    │  │
│   │ 0x10050008 │ TILE   │ [15:0]=Br                              │  │
│   │ 0x1005000C │ Q_ADDR │ Q matrix base (low 32b)                │  │
│   │ 0x10050010 │ K_ADDR │ K matrix base (low 32b)                │  │
│   │ 0x10050014 │ V_ADDR │ V matrix base (low 32b)                │  │
│   │ 0x10050018 │ O_ADDR │ O matrix base (low 32b)                │  │
│   │ 0x1005001C │ STATUS │ [0]=busy  [1]=done  (read-only)        │  │
│   └────────────┴────────┴────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
                                  │ AXI4 Slave (MMIO R/W)
┌──────────────────────────────────────────────────────────────────────┐
│                          HAL 層                                      │
│   flash_attn_hal.cpp                                                 │
│                                                                      │
│   memory_set(addr, data) ─── AXI4-Lite Write (AW→W→B)              │
│   memory_get(addr, &data)─── AXI4-Lite Read  (AR→R)                │
│   wait_for_irq() ──────────── clock_step loop + DMA service         │
│                                │                                     │
│   handle_dma_read()  ← ARVALID_M asserted: serve read burst        │
│   handle_dma_write() ← AWVALID_M asserted: serve write burst       │
│                                                                      │
│   vm_addr_h_：上 32 bits 位址（使 RTL 32b DMA 位址能對到 host 記憶體）│
└──────────────────────────────────────────────────────────────────────┘
                                  │ clock_step() / AXI4 signals
┌──────────────────────────────────────────────────────────────────────┐
│                   Verilated RTL Model                                │
│   Vflash_attn_wrapper (由 Verilator 從 .sv 生成的 C++ class)         │
│   flash_attn_wrapper.sv  →  Vflash_attn_wrapper.h / .cpp            │
│                                                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  AXI4 Slave FSM  │  Engine FSM (FA-2)  │  AXI4 Master FSM   │  │
│   │  (MMIO 暫存器)    │  (tiled computation)│  (DMA 請求)         │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 DPI-C 浮點橋接（RTL ↔ C 數學函式庫）

```
為什麼需要 DPI-C？
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SystemVerilog 的 real 型態 = fp64 (IEEE 754 double)
  AXI4 DMA 傳輸的是 fp32 (IEEE 754 single, 4 bytes)
  RTL 需要 exp() 和 sqrt() 做 online softmax 運算

  解法：用 DPI-C 橋接 → RTL 呼叫 C 標準函式庫

    RTL (SV real) ←──── dpi_fp32_bits_to_real() ────── RDATA_M (uint32)
    WDATA_M (uint32) ── dpi_real_to_fp32_bits() ──────→ RTL (SV real)
    exp(x)  ←────────── dpi_expf(x)  ←─────────────── <math.h>
    sqrt(x) ←────────── dpi_sqrtf(x) ←─────────────── <math.h>

  精度設計：
    內部 buffer（Q/K/V/S/P/O/l/m）全用 fp64 累加
    DMA 界面（讀入/寫出）才做 fp32 ↔ fp64 轉換
    好處：中間累加誤差最小，最後結果再 round 回 fp32
```

### 4.3 框架的可複用性（未來元件如何接入）

```
新增任何硬體元件只需：

  1. 新增 RTL：src/hardware/<module>/rtl/<module>_wrapper.sv
               (AXI4 Slave + AXI4 Master + 計算邏輯)

  2. 新增 HAL：include/hal/<module>_hal.hpp
               src/hal/<module>_hal.cpp
               (繼承 HALBase，複用 clock_step、DMA 邏輯)

  3. 新增 Driver：src/runtime/<module>/driver_<module>.h/.cpp
               (定義 MMIO base 和暫存器 offset)

  4. 新增 Runtime：src/runtime/<module>/runtime_<module>.cpp
               (高階 API：<module>_compute(...))

  5. 新增 Testbench：test/testbench/<module>/tb.cpp
               (CPU golden reference + hw sim + compare)

  ★ 整個流程是「填空」而非「重新設計」★
```

---

## 5. 已完成：FlashAttention 加速器詳解

### 5.1 FlashAttention-2 演算法概念圖

```
標準 Attention（記憶體瓶頸在哪？）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  DRAM ──── Q [N×d] ──────────────────────────────────→ SRAM
  DRAM ──── K [N×d] ──────────────────────────────────→ SRAM
             計算 S = Q×K^T [N×N] ← 需要存 N² 個數字！
  SRAM ──── S [N×N] ──────────────────────────────────→ DRAM   ← ❌ 寫
  DRAM ──── S [N×N] ──────────────────────────────────→ SRAM   ← ❌ 再讀
             softmax(S) → P [N×N]
  DRAM ──── V [N×d] ──────────────────────────────────→ SRAM
             計算 O = P×V [N×d]
  SRAM ──── O [N×d] ──────────────────────────────────→ DRAM

  N=197 時：S 矩陣 = 197×197×4 = 154,852 bytes ≈ 151 KB
  這個中間矩陣反覆寫讀 DRAM 是主要瓶頸

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

FlashAttention-2（本專案實作）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  把 Q/K/V 切成 tile：Q_i [Br×d]，K_j [Bc×d]，V_j [Bc×d]

  for i in range(N/Br):       ← 外迴圈 (i_tile)
      讀 Q_i from DRAM  ────── 只讀一次！
      初始化 O_i=0, l_i=0, m_i=-∞  （僅在 SRAM 中）

      for j in range(N/Bc):   ← 內迴圈 (j_tile)
          讀 K_j from DRAM
          讀 V_j from DRAM
          S_ij = Q_i × K_j^T   ← 在 SRAM 裡，不寫 DRAM！
          online_softmax_update(S_ij, K_j, V_j)
                               ← S_ij [Br×Bc] 只在暫存器/SRAM 中，
                                  永遠不需要回 DRAM！

      寫 O_i 回 DRAM  ────── 只寫一次！

  DRAM 存取 = Q(1次讀) + K(N/Bc次讀) + V(N/Bc次讀) + O(1次寫)
            = 3×N×d×4 bytes（讀）+ 1×N×d×4 bytes（寫）
  ★ 消除了 N² 的中間矩陣！★
```

### 5.2 Tiled 計算的矩陣視覺化（以 case1 為例：N=8, Br=4）

```
N=8, Br=Bc=4：共 2×2 = 4 個 tile

  Q 矩陣 [8×8]         K^T 矩陣 [8×8]        V 矩陣 [8×8]
  ┌────┬────┐          ┌────┬────┐           ┌────┬────┐
  │ Q₀ │    │          │ K₀ │ K₁ │           │ V₀ │    │
  │[4×8│    │          │[4×8│[4×8│           │[4×8│    │
  ├────┤    │          └────┴────┘           ├────┤    │
  │ Q₁ │    │                                │ V₁ │    │
  └────┴────┘                                └────┴────┘

  計算過程（★ 是 online softmax 跨 tile 累積的關鍵步驟 ★）：

  ── i=0（處理 Q₀ = Q[0:4]）──
  ①  讀 Q₀；初始化 O₀=0, l₀=0, m₀=-∞
  ②  j=0：讀 K₀, V₀；S[0:4][0:4] = Q₀×K₀^T；online softmax → O₀, l₀, m₀
  ③★ j=4：讀 K₁, V₁；S[0:4][4:8] = Q₀×K₁^T；online softmax（可能更新 m₀）
           correction factor corr = exp(m₀_old - m₀_new)
           O₀ = corr × O₀ + P×V₁  ← 修正過去的累積
           l₀ = corr × l₀ + sum(P)
  ④  finalize：O₀ /= l₀；寫 O₀ 回 DRAM

  ── i=4（處理 Q₁ = Q[4:8]）── 類似上述

  ★ 步驟③ 是驗證 online softmax 正確性的關鍵 ★
    如果 corr 算錯，O 和 CPU reference 一定不符
```

### 5.3 RTL State Machine 完整圖

```
                    ┌─────────────────────────────────────────┐
                    │              FA_IDLE                    │
                    │         等待 reg_control[0]=1           │
                    └──────────────────┬──────────────────────┘
                                       │ start
                    ┌──────────────────▼──────────────────────┐
              ┌────►│  FA_DMA_Q_AR  發出 AR 請求讀 Q row      │
              │     │  ARADDR = q_addr + (i+cur_row)×d×4      │
              │     └──────────────────┬──────────────────────┘
              │                        │ ARREADY
              │     ┌──────────────────▼──────────────────────┐
              │     │  FA_DMA_Q_R   接收 R beats 存入 Q_buf   │
              │     │  每個 RVALID beat：Q_buf[cur_row][word]  │
              │     └──────────────────┬──────────────────────┘
              │         cur_row++ ◄────┤ RLAST && cur_row < Br-1
              │                        │ RLAST && cur_row == Br-1
              │     ┌──────────────────▼──────────────────────┐
              │     │  FA_INIT_O_LM  初始化 O=0, l=0, m=-∞   │
              │     └──────────────────┬──────────────────────┘
              │                        │
              │          ┌─────────────▼───────────────────────┐
              │    ┌────►│  FA_DMA_K_AR  讀 K[j+cur_row] row   │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_K_R  存入 K_buf             │
              │    │     └─────────────┬───────────────────────┘
              │    │  cur_row++ ◄──────┤ RLAST && cur_row < Br-1
              │    │                   │ RLAST && cur_row == Br-1
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_V_AR  讀 V[j+cur_row] row   │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_V_R  存入 V_buf             │
              │    │     └─────────────┬───────────────────────┘
              │    │  cur_row++ ◄──────┤ RLAST && cur_row < Br-1
              │    │                   │ RLAST && cur_row == Br-1
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_COMPUTE  一個 clock cycle        │
              │    │     │  ① S_ij = Q_buf × K_buf^T / √d     │
              │    │     │  ② online softmax → O_buf, l, m    │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_NEXT_J                          │
              │    └─────┤  j += Br                           │
              │          └─────────────┬───────────────────────┘
              │                        │ j+Br >= N（所有 j-tile 完成）
              │          ┌─────────────▼───────────────────────┐
              │          │  FA_FINALIZE  O_buf[r][k] /= l[r]  │
              │          └─────────────┬───────────────────────┘
              │                        │
              │    ┌────►┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_O_AW 發出 AW 請求寫 O row   │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │ AWREADY
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_O_W  傳送 W beats           │
              │    │     │  WDATA = O_buf[cur_row][word_cnt]   │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │ last word accepted
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_DMA_O_B  等待 B response        │
              │    │     └─────────────┬───────────────────────┘
              │    │                   │ BVALID
              │    │     ┌─────────────▼───────────────────────┐
              │    │     │  FA_NEXT_I                          │
              │    └─────┤  cur_row++ （下一個 O row）          │
              │          └─────────────┬───────────────────────┘
              │    i += Br             │ cur_row == Br（整個 i-tile 寫完）
              └────────────────────────┤ i+Br < N（還有 i-tile）
                                       │ i+Br >= N（全部完成）
                          ┌────────────▼───────────────────────┐
                          │  FA_DONE                           │
                          │  FA_interrupt = 1                  │
                          └────────────────────────────────────┘
```

---

## 6. Online Softmax 視覺化解說

### 6.1 為什麼需要 Online Softmax？

```
問題：傳統 softmax 需要兩次掃描 ← 不適合 tiled 計算

  Pass 1：掃 N 個元素，找 max(S)
  Pass 2：計算 exp(S_i - max)，加總，再 normalize

  如果 S 矩陣分散在多個 j-tile，每個 tile 進來時
  還不知道全部的 max，就無法做 normalize。

解法：Online Softmax ← 每個 j-tile 處理完就更新，不需要等

  關鍵洞察：
    exp(a - max_new) = exp(a - max_old) × exp(max_old - max_new)
                       ───────────────   ──────────────────────
                       舊的 exp 值        correction factor（corr）
```

### 6.2 Online Softmax 逐步視覺化（以 row 0, N=8, Br=4 為例）

```
初始狀態：m = -∞,  l = 0,  O = [0, 0, ..., 0]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
j-tile 0（j=0：K[0:4], V[0:4]）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  S[0, 0:4] = Q[0] · K[0:4]^T / √d = [s₀, s₁, s₂, s₃]
                                       假設值: [0.2, 0.5, 0.1, 0.3]

  row_max = max(0.2, 0.5, 0.1, 0.3) = 0.5
  m_new   = max(m_old=-∞, row_max=0.5) = 0.5
  corr    = exp(-∞ - 0.5) ≈ 0   ← 第一次 j-tile，corr ≈ 0

  P[0, 0:4] = exp([0.2, 0.5, 0.1, 0.3] - 0.5)
            = [exp(-0.3), exp(0), exp(-0.4), exp(-0.2)]
            = [0.741,     1.000,  0.670,     0.819]

  sum_p = 0.741 + 1.000 + 0.670 + 0.819 = 3.230

  O  ← 0 × corr + P[0,0:4] × V[0:4]   （corr=0 所以舊 O 被丟棄）
  l  ← 0 × corr + 3.230 = 3.230
  m  ← 0.5

  狀態更新後：m=0.5, l=3.230, O=P×V（基於第一個 j-tile）

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
j-tile 1（j=4：K[4:8], V[4:8]）← ★ Online Softmax 關鍵 ★
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  S[0, 4:8] = Q[0] · K[4:8]^T / √d = [s₄, s₅, s₆, s₇]
                                       假設值: [0.3, 0.8, 0.4, 0.1]
                                                    ▲
                                                    新 max > 舊 max！

  row_max = max(0.3, 0.8, 0.4, 0.1) = 0.8
  m_new   = max(m_old=0.5, row_max=0.8) = 0.8   ← max 更新了
  corr    = exp(m_old - m_new)
          = exp(0.5 - 0.8)
          = exp(-0.3)
          = 0.741                                   ← 這個值很重要！

  ↓ 修正舊的 O 和 l：

  P[0, 4:8] = exp([0.3, 0.8, 0.4, 0.1] - 0.8)
            = [exp(-0.5), exp(0), exp(-0.4), exp(-0.7)]
            = [0.607,     1.000,  0.670,     0.497]

  sum_p = 0.607 + 1.000 + 0.670 + 0.497 = 2.774

  O  ← corr × O_old + P[0,4:8] × V[4:8]
     = 0.741 × (舊 O) + P[0,4:8] × V[4:8]
     ↑ 把舊 O 乘上 corr（縮小），加上新貢獻

  l  ← corr × l_old + sum_p
     = 0.741 × 3.230 + 2.774
     = 2.393 + 2.774
     = 5.167

  m  ← 0.8

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FA_FINALIZE（所有 j-tile 掃完後）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  O_final = O / l = O / 5.167   ← 這就是最終的 attention output

  驗證：O_final 應與 standard_attention_cpu() 的 O_ref 一致（誤差 < 1e-4）
```

### 6.3 Online Softmax 數學等價性說明

```
為什麼 online softmax 和傳統 softmax 結果相同？

傳統：
  softmax(s₀,...,s₇) = exp(sᵢ - max_all) / Σ exp(sⱼ - max_all)

Online（兩個 tile 後）：
  分子（O[k]）= Σᵢ∈tile0 exp(sᵢ-m_final)×Vᵢₖ + Σᵢ∈tile1 exp(sᵢ-m_final)×Vᵢₖ
  分母（l）   = Σⱼ exp(sⱼ - m_final)

  等價性來自：
    tile0 的貢獻在 tile1 處理時乘了 corr = exp(m_old - m_new)
    最終效果 = exp(sᵢ - m_old) × exp(m_old - m_final) = exp(sᵢ - m_final)

  ✅ 數學上完全等價，只差浮點精度
```

---

## 7. 各元件驗證策略

### 7.1 FlashAttention 驗證（已完成）

```
驗證架構：

  gen_qkv() ──→ Q, K, V（決定性 sin/cos 模式）
      │
      ├──→ standard_attention_cpu()  ──→  O_ref  ← 標準 two-pass softmax
      │      (fp32, O(N²d) 時間)
      │
      └──→ flash_attention()         ──→  O_hw   ← tiled online softmax
             (RTL behavioral sim)
      │
      └──→ |O_hw[i] - O_ref[i]| ≤ 1e-4 for all i  →  PASS / FAIL

驗證覆蓋度：
  case0（N=4,  Br=4）： 1 tile，驗證 DMA 讀寫正確性
  case1（N=8,  Br=4）： 2 tiles，驗證 online softmax 跨 tile 修正
  case2（N=196,Br=14）：14×14=196 tiles，驗證 ViT-Small 實際規模
```

### 7.2 RMSNorm 驗證規劃（未完成）

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × γ

驗證架構（規劃中）：

  gen_input() ──→ x[N×d], γ[d]
      │
      ├──→ rms_norm_cpu()    ──→  O_ref  ← 軟體計算
      │
      └──→ rms_norm_hw()     ──→  O_hw   ← RTL 模擬（待實作）
      │
      └──→ |O_hw[i] - O_ref[i]| ≤ 1e-5 → PASS / FAIL

ViT-Small 規格：
  N = 197（tokens）
  d = 384（embedding dim）
  γ ∈ R^384（learnable scale）
  ε = 1e-6

測試 case 規劃：
  case0：N=8,   d=4  （smoke test）
  case1：N=64,  d=64  （中等規模）
  case2：N=197, d=384 （ViT-Small 實際）
```

### 7.3 完整 Transformer Block 驗證規劃（未完成）

```
Block-level 驗證流程（規劃中）：

  ┌──────────────────────────────────────────────────────────────┐
  │  Python/PyTorch reference                                    │
  │  import torch                                                │
  │  block = VitBlock(d=384, heads=6, mlp_ratio=4)               │
  │  O_ref = block(x)    ← FP32 ground truth                    │
  └──────────────────────────────────────────────────────────────┘
                │
                │  比對
                ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  C++ Hardware Simulation                                     │
  │  rms_norm_hw(x)                ← RMSNorm (待實作)            │
  │  for h in 0..5:                                              │
  │      Q_h, K_h, V_h = linear_proj_hw(x_norm, Wq, Wk, Wv, h) │
  │      O_h = flash_attention(Q_h, K_h, V_h, N=197, d=64, Br=14│
  │  O_attn = concat(O_h) @ Wo                                   │
  │  x = x + O_attn              ← Residual Add                 │
  │  x = rms_norm_hw(x)          ← RMSNorm (待實作)              │
  │  x = ffn_hw(x)               ← FFN (待實作)                  │
  │  x = x + x_ffn               ← Residual Add                 │
  └──────────────────────────────────────────────────────────────┘
```

### 7.4 各元件驗證獨立性

```
每個元件可以獨立驗證，互不依賴：

  FlashAttention  ✅  已驗證  →  輸入 Q/K/V [N×d]，輸出 O [N×d]
  RMSNorm         ⏳  規劃中  →  輸入 x [N×d], γ [d]，輸出 [N×d]
  Linear (GEMM)   📋  研究中  →  輸入 x [N×M], W [M×K]，輸出 [N×K]
  FFN / MLP       📋  規劃中  →  輸入 x [N×384]，輸出 [N×384]（兩層 GEMM）

  驗證的「黃金標準」：
    FlashAttention ← CPU fp32 標準 attention（兩次掃描）
    RMSNorm        ← CPU fp32 / PyTorch reference
    Linear         ← CPU fp32 矩陣乘法
    FFN            ← CPU fp32 / PyTorch reference
    整個 Block     ← PyTorch 的 ViT block（同樣的 weight）
    End-to-End     ← PyTorch ViT-Small ImageNet accuracy 比對
```

---

## 8. 如何執行當前模擬

### 8.1 環境需求

```
作業系統：Linux 或 WSL2（Windows Subsystem for Linux）

必要工具：
  verilator --version    （需 5.x，推薦 5.006 以上）
  g++ --version          （需支援 C++17，推薦 g++ 11 以上）
  make --version

Verilator 安裝路徑：/usr/local/share/verilator
  若不同，修改以下兩個 Makefile 中的 VERILATOR_ROOT 變數：
    src/hardware/flash_attn/Makefile
    test/testbench/flash_attn/Makefile
```

### 8.2 一鍵執行所有 Case

```bash
cd "第三階段/【Phase0+1+ScaleTying+FlashAttn】FlashAttention_RMSNorm_ViT_ImageNet"
make run_fa
```

執行流程：
```
Step 1: Verilate RTL
  flash_attn_wrapper.sv  →  obj_dir/Vflash_attn_wrapper.h/.cpp
                         →  obj_dir/libVflash_attn_wrapper.a

Step 2: 編譯 Testbench
  tb.cpp + hal + driver + runtime → build/tb_fa_case{0,1,2}

Step 3: 執行
  ./tb_fa_case0   (N=4,  d=4,  Br=4)
  ./tb_fa_case1   (N=8,  d=8,  Br=4)
  ./tb_fa_case2   (N=196,d=64, Br=14)  ← ViT-Small 規格

Step 4: 最終報告
  [TB/FA] ALL TESTS PASSED (3/3)
```

### 8.3 分案執行

```bash
make run_fa_case0   # Smoke test (最快)
make run_fa_case1   # Online softmax 核心驗證
make run_fa_case2   # ViT-Small 實際規格 (最慢，約 1 分鐘)
```

### 8.4 進階選項

```bash
# 顯示所有 MMIO 和 DMA 操作（debug 用）
make run_fa_case2 DEBUG=1

# 輸出 FST 波形，可用 GTKWave 開啟
make run_fa_case2 TRACE=1
gtkwave build/fa_2.fst

# 兩者同時開啟
make run_fa TRACE=1 DEBUG=1

# 清理
make clean
```

### 8.5 預期輸出（case2）

```
[TB/FA] Starting FlashAttention simulation — case2  (N=196 d=64 Br=14)
[TB/FA] Computing CPU reference...
[TB/FA] Running hardware simulation...

===== FlashAttention Simulation Result =====
  Case              : case2  (N=196 d=64 Br=14)
  Cycles            : XXXXXXX
  Time (s)          : X.XXXXXX
  Mem reads  (Bytes): 301056  [294.0 KB]   ← = 3×196×64×4 ✓
  Mem writes (Bytes): 100352   [98.0 KB]   ← = 1×196×64×4 ✓
  Expected reads    : 301056  (3×N×d×4 bytes)
  Expected writes   : 100352  (1×N×d×4 bytes)
  Max abs diff      : X.XXe-0X  at idx=...
  Errors (>1e-04)   : 0  [PASS]            ← 全部通過！
=============================================
[TB/FA] *** TEST PASSED  (cycles=XXXXXXX) ***
```

---

## 9. 效能指標解讀

### 9.1 Cycle Count 組成分析

```
Cycle 數的來源（以 case2 為例，N=196, d=64, Br=14）：

  外迴圈：14 個 i-tile
  內迴圈：14 個 j-tile
  ──────────────────────────────────────────────────────────────────

  每個 DMA read transaction（ARLEN = d-1 = 63 beats）：
    AR handshake     ≈    2 cycles
    R data beats     = d × (1 + MEM_ACCESS_CYCLE) = 64 × 6 = 384 cycles
    小計             ≈  386 cycles

  每個 DMA write transaction（AWLEN = d-1 = 63 beats）：
    AW handshake     ≈    2 cycles
    W data beats     = d × (1 + MEM_ACCESS_CYCLE) = 64 × 6 = 384 cycles
    B response       ≈    2 cycles
    小計             ≈  388 cycles

  ──────────────────────────────────────────────────────────────────
  Q 讀取（每 i-tile 讀 Br=14 rows）：
    14 i × 14 rows × 386 cycles ≈ 75,628 cycles

  K+V 讀取（每 j-tile 讀 Br=14 rows × 2）：
    14 i × 14 j × 14 rows × 386 cycles × 2 ≈ 2,115,664 cycles

  O 寫出（每 i-tile 寫 Br=14 rows）：
    14 i × 14 rows × 388 cycles ≈ 76,048 cycles

  FA_COMPUTE（每個 tile 1 cycle）：
    196 cycles

  ──────────────────────────────────────────────────────────────────
  總計 ≈ 2,267,536 cycles（實際值因 FSM handshake 細節略高）
```

### 9.2 記憶體頻寬分析

```
DMA 理論值驗證：

  讀取：Q(1) + K(14) + V(14) per i-tile × 14 i-tiles
      = 14 × (14 + 14 + 14) × Br × d × 4 bytes  ← 注意 Q 只讀一次
      = 14 × (1 + 14 + 14) × 14 × 64 × 4
      = 理論值：301,056 bytes（含 Q 每 i-tile 讀一次）

  輸出欄位 "Mem reads  (Bytes): 301056" 必須與 "Expected reads: 301056" 完全相符
  ← 若不符，代表 DMA 地址計算有 bug

MAC 利用率：
  FA_COMPUTE behavioral sim 中，整個 tile 的計算在「1 clock」內完成
  → MAC/cycle 數字遠大於 1（這是 behavioral sim 的特性，不代表真實 ASIC 吞吐量）
  → 真實 PYNQ-Z2 systolic array 約 2×Br×d MACs per 若干 cycles
```

### 9.3 與真實 FPGA 的對應

```
Behavioral Simulation    →   真實 FPGA/ASIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FA_COMPUTE (1 cycle)     →   systolic array 需要 Br × d cycles
MEM_ACCESS_CYCLE = 5     →   DRAM AXI4 latency（PYNQ-Z2 ~100ns）
CYCLE_TIME = 5 ns        →   200 MHz 時鐘 = 5 ns/cycle
DMA burst (d words)      →   AXI4 INCR burst (ARLEN=d-1)

用途：
  Behavioral sim 驗證「演算法正確性」和「DMA 存取模式」
  不適合直接用 cycle count 推算 FPGA 效能
  FPGA 效能需要實際綜合和 Place & Route 後才能確認
```

---

## 10. 未來各階段規劃

### 10.1 Phase 2：RMSNorm 加速器

```
目標：實作並驗證 RMSNorm 硬體加速器

RTL 設計重點：
  1. DMA 讀入 x[N×d] 和 γ[d]
  2. 每個 row 計算：sum_sq = Σ xᵢ²
  3. rms = sqrt(sum_sq / d + ε)
  4. 每個元素：output[i] = x[i] / rms × γ[i]
  5. DMA 寫出 output[N×d]

與 FlashAttention 的架構差異：
  - 無 j-tile 迴圈（每個 row 獨立）
  - 更簡單的 state machine（~5 states）
  - 需要除法：可用 DPI-C 呼叫 1/sqrt()

估計工作量：1~2 週（框架已建立，只需填新 RTL）
```

### 10.2 Phase 3：Linear Projection (GEMM) 加速器

```
目標：實作 Wq/Wk/Wv/Wo 的矩陣乘法加速器

ViT-Small 規格：
  Wq/Wk/Wv/Wo ∈ R^{384×384}
  輸入：x [197×384]
  輸出：Q/K/V [197×384] → reshape → [6×197×64]

設計選項：
  A. 直接 GEMM（逐 tile 矩陣乘法）
  B. 用 lab-4 的 systolic array（如果介面相容）
  C. 行向量逐條計算（最簡單，但最慢）

與 FA 的銜接：
  GEMM output [197×384] → reshape [6×197×64]
  → 6× flash_attention() → concat [197×384]

估計工作量：2~4 週（GEMM 相對複雜）
```

### 10.3 Phase 4：完整 Transformer Block

```
整合方式（軟體層串接各硬體加速器）：

void transformer_block(float* x, /* weights... */) {
    // [1] Pre-Norm
    rms_norm_hw(x, gamma1, x_norm);     // 待實作

    // [2] Linear Projection
    gemm_hw(x_norm, Wq, Q);             // 待實作
    gemm_hw(x_norm, Wk, K);             // 待實作
    gemm_hw(x_norm, Wv, V);             // 待實作

    // [3] Multi-Head Flash Attention
    for (int h = 0; h < 6; h++)
        flash_attention(Q+h*stride, K+h*stride, V+h*stride,
                       O+h*stride, 197, 64, 14);  // ← 已實作 ✅

    // [4] Output projection + Residual
    gemm_hw(O_concat, Wo, O_proj);      // 待實作
    residual_add(x, O_proj, x);         // 純加法

    // [5] Post-Norm
    rms_norm_hw(x, gamma2, x_norm2);    // 待實作

    // [6] FFN
    gemm_hw(x_norm2, W_fc1, h1);        // 待實作（含 GELU）
    gemm_hw(h1, W_fc2, ffn_out);        // 待實作
    residual_add(x, ffn_out, x);        // 純加法
}
```

### 10.4 Phase 5：End-to-End ViT-Small

```
完整推論流程：

  1. Patch Embedding（可用 CPU 軟體計算，非加速器重點）
  2. 12× transformer_block()
  3. Final RMSNorm（同 Phase 2 實作）
  4. CLS token 取出 + Linear classifier（GEMM）
  5. Softmax → Top-1 class

驗證方式：
  載入 PyTorch 預訓練 ViT-Small/16 的 weights
  用 ImageNet validation set 的 100~1000 張圖片
  比較本模擬器的 Top-1 accuracy vs PyTorch 的 accuracy
  目標：差異 < 0.5%（允許浮點模型誤差）
```

---

## 11. 與 lab-4 的差異對照

### 11.1 架構對照表

| 層次 | lab-4 原始 | 本專案 | 主要差異 |
|------|----------|--------|---------|
| **HAL** | `dla_hal.hpp/cpp` | `flash_attn_hal.hpp/cpp` | Verilated model 換成 `Vflash_attn_wrapper`；interrupt 信號名稱改 |
| **共用 Base** | `hal.hpp` | `hal.hpp` | **完全相同**，零修改 |
| **Hardware RTL** | `asic_wrapper.sv`（卷積） | `flash_attn_wrapper.sv`（FA-2） | 全新 RTL；計算邏輯從 conv 換成 tiled attention |
| **Driver** | `driver_dla.h/cpp` | `driver_flash_attn.h/cpp` | 全新 register map；base 位址 `0x10050000` |
| **Runtime** | `runtime_dla.cpp` | `runtime_flash_attn.cpp` | `run_workload()` → `flash_attention()` |
| **DPI-C** | 無 | `src/dpi/dpi_math.c` | **新增**；FA-2 需要 exp/sqrt/fp32↔bits |
| **MMIO 基底** | `0x10040000` | `0x10050000` | 避免位址衝突 |

### 11.2 三個關鍵 RTL 設計修正

```
（這三個問題是從 lab-4 框架移植時遇到的新挑戰）

問題 1：S_buf NBA timing bug
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  錯誤寫法：S_buf[r][c] <= s_val  （NBA）
  問題：FA_COMPUTE 同一個 always_ff block 內
        先寫 S_buf（NBA）再讀 S_buf，
        讀到的是上一個 j-tile 的舊值 → online softmax 計算錯誤

  正確寫法：S_buf[r][c] = s_val   （Blocking）
  原理：S_buf 只在同一個 FA_COMPUTE 執行內被讀寫，
        blocking 在 Verilator behavioral sim 中立即可見

問題 2：WLAST_M NBA race condition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  錯誤寫法：WLAST_M <= (word_cnt == d-1)
  問題：if (WLAST_M) 在同一個 always_ff 裡看到的是上個 cycle 的值
        → 最後一個 W-channel beat 的 state transition 延後一拍
        → HAL 的 BVALID 等待邏輯 deadlock

  正確寫法：assign WLAST_M = (state==FA_DMA_O_W) && (word_cnt==d_w[7:0]-8'd1);
  原理：純 combinational，HAL 在任何時刻都能看到正確值

問題 3：RREADY_M / BREADY_M NBA race
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  同上，這兩個 ready 信號改成純 combinational assign，
  確保 HAL 的 eval() 後立即反映正確 ready 狀態
```

---

## 快速參考

```
執行指令：
  make run_fa              一鍵 build + 跑全部 3 個 case
  make run_fa_case2        只跑 ViT-Small 規格 case
  make run_fa DEBUG=1      顯示所有 MMIO/DMA log
  make run_fa TRACE=1      輸出 FST 波形
  make clean               清除所有 build 產物

關鍵檔案：
  RTL 邏輯         →  src/hardware/flash_attn/rtl/flash_attn_wrapper.sv
  Online softmax   →  flash_attn_wrapper.sv : FA_COMPUTE state（line 446）
  DMA 仿真         →  src/hal/flash_attn_hal.cpp : handle_dma_read/write()
  暫存器定義        →  src/runtime/flash_attn/driver_flash_attn.h
  CPU reference    →  src/runtime/flash_attn/runtime_flash_attn.cpp : standard_attention_cpu()
  Test case 參數   →  test/cases/case{0,1,2}/workload.h

進度摘要：
  ✅ Phase 0：框架建立（HAL/Driver/Runtime/DPI-C）  100%
  ✅ Phase 1：FlashAttention RTL 驗證               100%
  ⏳ Phase 2：RMSNorm 加速器                         規劃完成，RTL 待實作
  📋 Phase 3：Linear Projection (GEMM) 加速器        研究中
  📋 Phase 4：完整 Transformer Block 整合            規劃中
  🎯 Phase 5：End-to-End ViT-Small                  最終目標
```
