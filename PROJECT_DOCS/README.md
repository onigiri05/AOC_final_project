# FlashAttention-2 INT8 硬體加速 ViT-Small/16 — 專案文件總覽

> **版本**：Phase 0+1+ScaleTying+FlashAttn（含 CLS Fix、RMSNorm、Multi-Call HW Fix）  
> **最後更新**：2026-05-27  
> **平台**：Verilator 模擬 + Docker 容器 + Windows 主機

---

## 專案一句話摘要

以 **14×14 Systolic Array** 為核心實作 FlashAttention-2 INT8 硬體加速器，整合入 ViT-Small/16 完整推論管線（12 層 Transformer），並在 Verilator 模擬環境中完成端對端驗證。

---

## 文件導覽

| 文件 | 說明 | 適合讀者 |
|------|------|----------|
| **本文件 (README)** | 專案概覽、文件索引 | 所有人 |
| [01_系統架構.md](01_系統架構.md) | 硬體架構、模組拓撲、資料流圖 | 硬體 / 系統設計者 |
| [02_環境設定與工作流.md](02_環境設定與工作流.md) | Docker 建置、指令教學、完整 step-by-step | 新加入的開發者 |
| [03_模組說明.md](03_模組說明.md) | FA 核心、HAL、Driver、ViT Runtime 詳解 | 軟硬體整合者 |
| [04_實驗完整報告.md](04_實驗完整報告.md) | 所有測試結果、數據表格、分析 | 評審 / 報告撰寫 |
| [05_記憶體分配說明（含實驗修改）.md](05_記憶體分配說明（含實驗修改）.md) | 原始 BRAM 規劃 + 實驗中的軟體層修改 | 硬體實作者 |
| [06_修改紀錄與原因.md](06_修改紀錄與原因.md) | 每次重要修改的時間線與技術原因 | 所有人 |

---

## 快速指令索引

```bash
# ── 建置 ──────────────────────────────────────────────────
make -C test/testbench/vit clean && make -C test/testbench/vit

# ── 測試模式 ──────────────────────────────────────────────
./test/testbench/vit/build/tb_vit                              # Mode A：合成單塊
./test/testbench/vit/build/tb_vit weights/                    # Mode B：12層HW推論
./test/testbench/vit/build/tb_vit weights/ img.bin [class]    # Mode C：真實影像
./test/testbench/vit/build/tb_vit weights/ --batch batch_50/  # Mode D：批次SW準確率
./test/testbench/vit/build/tb_vit weights/ --batch batch_5/ --hw  # Mode D HW

# ── 建置旗標 ──────────────────────────────────────────────
make -C test/testbench/vit clean && make -C test/testbench/vit STATS_DEBUG=1
make -C test/testbench/vit clean && make -C test/testbench/vit SCALE_TYING=1
make -C test/testbench/vit clean && make -C test/testbench/vit RMSNORM=1

# ── 前處理工具 ────────────────────────────────────────────
python scripts/batch_preprocess.py \
    --ds "C:\...\IMAGENET_VAL_HF" --n 50 --out batch_50/
```

---

## 實驗結果速覽

| 實驗項目 | 結果 | 狀態 |
|----------|------|------|
| FA 硬體單次驗證 (case0/1/2 INT8) | Max diff 1.35e-4，0 errors | ✅ PASS |
| FA 多次連續 HW call (Mode A, 6 heads) | 6 × 747,497 cycles | ✅ PASS |
| ViT 12層完整 HW 推論 (Mode B) | 72 × 747,497 = 53,819,784 cycles | ✅ PASS |
| ViT SW 批次準確率 (50 張) | Top-1 72% / Top-5 92% | ✅ PASS |
| ScaleTying 比較 | tied = untied = 1.35e-4 | ✅ PASS |
| RMSNorm 編譯與執行 | 72% Top-1（LayerNorm 權重） | ✅ PASS |

---

## 專案目錄結構

```
【Phase0+1+ScaleTying+FlashAttn】FlashAttention_RMSNorm_ViT_ImageNet/
│
├── PROJECT_DOCS/              ← 本文件資料夾（你正在看的地方）
│
├── src/
│   ├── hal/                   ← FlashAttnHAL（C++驅動Verilator）
│   ├── hardware/flash_attn/   ← Verilog RTL + Verilator編譯
│   └── runtime/
│       ├── flash_attn/        ← flash_attention() API
│       └── vit/               ← ViT推論引擎
│
├── test/testbench/
│   ├── flash_attn/            ← FA單元測試 (case0/1/2)
│   └── vit/                   ← ViT端對端測試 (tb_vit)
│
├── scripts/
│   └── batch_preprocess.py    ← ImageNet批次前處理
│
├── weights/                   ← ViT-Small/16 預訓練權重
├── batch_50/                  ← 50張ImageNet測試影像
└── final project記憶體分配.md ← 硬體記憶體規劃原始文件
```
