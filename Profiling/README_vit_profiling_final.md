# ViT Profiling Package v3 — RTL-aware 8×8 Analytical Profiler

本專案用來比較 **Baseline-B FP32 hardware-aware tiled model** 與 **Optimized INT8 RTL-aware model** 在一個 ViT-Small/16 Transformer block 上的理論 profiling 結果。新版 profiler 對齊目前 FPGA RTL 設計：shared 8×8 systolic array、INT8 optimized dataflow、BRAM reuse、weight ping-pong、Softmax/RMSNorm/GELU LUT，以及 MLP GELU page streaming。

---

## 1. How to Run

### 1.1 建議環境

建議使用 Python 3.10 以上版本，並建立 virtual environment：

```bash
python -m venv myenv
source myenv/bin/activate      # Linux / WSL / macOS
# 或 Windows PowerShell:
# .\myenv\Scripts\Activate.ps1
```

### 1.2 安裝套件

基本執行與畫圖需要：

```bash
pip install numpy pandas matplotlib tabulate
```

若要從 timm model 自動解析 ViT shape，建議安裝：

```bash
pip install timm torch
```

如果沒有安裝 `timm` 或 checkpoint 無法解析，程式會 fallback 到預設 ViT-Small/16 config，但正式報告建議仍安裝 `timm` 並提供 optimized checkpoint。

### 1.3 檔案放置方式

建議專案結構如下：

```text
vit_profiling_package/
├── run_profile.py
└── vit_profiler/
    ├── __init__.py
    ├── config.py
    ├── main.py
    ├── model_parser.py
    ├── sections.py
    ├── profiler.py
    ├── roofline.py
    └── optimization_analysis.py
```

若你現在的檔案名稱含有括號，例如 `main(7).py`、`profiler(13).py`，建議整理回上面的標準檔名，並放進 `vit_profiler/` package 裡，否則 `from vit_profiler.main import main` 可能會找不到模組。

### 1.4 基本執行指令

使用 Baseline timm model，並指定 optimized checkpoint：

```bash
python run_profile.py \
  --baseline-model vit_small_patch16_224.augreg_in21k_ft_in1k \
  --optimized-checkpoint ./rms_qat_best.pt \
  -o outputs \
  --clock-mhz 25 \
  --dram-eff 0.50 \
  --bram-service-bpc 8 \
  --dsp-packing 2.0
```

若只是測試流程、沒有 optimized checkpoint，也可以先跑：

```bash
python run_profile.py -o outputs
```

若不想產生圖，只輸出 CSV / Markdown：

```bash
python run_profile.py -o outputs --no-plots
```

### 1.5 常用參數說明

| 參數 | 預設值 | 說明 |
|---|---:|---|
| `--baseline-model` | `vit_small_patch16_224.augreg_in21k_ft_in1k` | Baseline-B 使用的 timm ViT model name |
| `--baseline-pretrained` | off | 是否載入 timm pretrained weights；一般只解析 shape，不需要開啟 |
| `--optimized-checkpoint` | `None` | Optimized INT8 / QAT checkpoint path，例如 `rms_qat_best.pt` |
| `--output-dir`, `-o` | `outputs` | 輸出資料夾 |
| `--clock-mhz` | `25.0` | FPGA PL clock，單位 MHz |
| `--dram-eff` | `0.50` | DDR efficiency，相對 2.1 GB/s peak bandwidth |
| `--bram-kb` | `560.0` | 可用 BRAM capacity，單位 KB |
| `--dsp-packing` | `2.0` | INT8 DSP packing throughput multiplier |
| `--bram-service-bpc` | `8.0` | BRAM service bandwidth，單位 bytes/cycle |
| `--leakage-w` | `0.20` | leakage/static power 假設，單位 W |
| `--gelu-page-words` | `1024` | GELU page BRAM size，以 INT8 words 計 |
| `--optimistic-gelu-page` | off | 假設 FC2 hidden pages 只從 DDR load 一次，而不是每個 FC2 output tile reload |
| `--no-plots` | off | 只輸出表格，不產生 PNG 圖 |

---

## 2. File Structure and Function

### `run_profile.py`

最外層 entry point。內容很簡單，主要呼叫 `vit_profiler.main` 裡的 `main()`：

```text
from vit_profiler.main import main
```

使用者通常直接執行這個檔案。

---

### `vit_profiler/main.py`

主流程控制檔，負責：

1. 解析 command-line arguments。
2. 建立 baseline / optimized model spec。
3. 建立 hardware config 與 energy config。
4. 呼叫 profiler 產生 per-section profiling results。
5. 產生 group summary。
6. 輸出 CSV / Markdown / PNG plots。
7. 輸出 optimization impact 分析。

主要輸出都在這個檔案中指定。

---

### `vit_profiler/config.py`

定義 profiling 使用的 dataclass：

| Class | 功用 |
|---|---|
| `ViTModelSpec` | ViT model shape，例如 tokens、embed dim、heads、MLP dim、blocks |
| `HardwareConfig` | FPGA hardware 假設，例如 8×8 systolic、clock、DRAM bandwidth、BRAM capacity、DSP packing |
| `EnergyConfig` | analytical energy model，例如 FP32/INT8 MAC energy、BRAM/DRAM byte energy、LUT energy、leakage power |

預設硬體設定對齊目前 RTL：

```text
8×8 shared systolic
25 MHz clock
2.1 GB/s DDR peak bandwidth
50% DRAM efficiency
560 KB BRAM capacity
8 B/cycle BRAM service bandwidth
DSP packing factor = 2.0
```

---

### `vit_profiler/model_parser.py`

負責建立 model spec：

1. `parse_timm_model()`：從 timm model 解析 Baseline-B 的 ViT shape。
2. `parse_checkpoint()`：從 optimized checkpoint 的 state dict shape 解析 optimized model shape。
3. 若 timm 或 checkpoint 無法解析，會 fallback 到預設 ViT-Small/16 config。
4. `save_specs()`：輸出 `parsed_model_specs.json`。

---

### `vit_profiler/sections.py`

定義要 profiling 的 sections 與 groups。

Sections：

| Section | Kind |
|---|---|
| `Norm` | elementwise |
| `QKV Projection` | gemm |
| `Attention Score` | gemm |
| `Softmax` | softmax |
| `Attention Value` | gemm |
| `Output Projection` | gemm_elementwise |
| `MLP` | gemm_elementwise |

Groups：

| Group | Included sections |
|---|---|
| `Full MHSA` | QKV Projection + Attention Score + Softmax + Attention Value + Output Projection |
| `Full one block` | Norm + QKV Projection + Attention Score + Softmax + Attention Value + Output Projection + MLP |

---

### `vit_profiler/profiler.py`

核心 analytical profiler。負責針對每個 section 計算：

| Metric | 說明 |
|---|---|
| `math_macs` / `macs` | GEMM-like true MAC count |
| `operations` | non-GEMM modeled operations，例如 Norm / Softmax |
| `lut_accesses` | RMSNorm / Softmax / GELU LUT access count |
| `dram_read_bytes`, `dram_write_bytes`, `dram_total_bytes` | DRAM traffic |
| `bram_read_bytes`, `bram_write_bytes`, `bram_total_bytes` | BRAM traffic |
| `dram_usage_bytes`, `bram_usage_bytes`, `bram_usage_ramb36` | peak storage usage |
| `compute_cycles`, `dram_cycles`, `bram_cycles`, `cycles_total` | cycle breakdown |
| `latency_ms` | latency in ms |
| `performance_macs_per_cycle` | MACs / cycle |
| `operational_intensity` | MACs / DRAM byte |
| `total_memory_intensity` | MACs / (DRAM + BRAM byte) |
| `bound` | memory / compute / on-chip-memory / on-chip |
| `energy_*_uj` | analytical energy breakdown |
| `notes` | section-specific explanation |

新版 RTL-aware 設定包含：

1. Baseline-B：FP32 all-DRAM tiled model，使用相同 8×8 tile，但無 BRAM reuse。
2. Optimized：INT8 RTL-aware model，使用 BRAM reuse、weight ping-pong、DSP packing。
3. QKV Projection optimized 只 precompute K/V，不完整儲存 Q。
4. Attention Score optimized 包含 Q tile recomputation + QKᵀ。
5. Softmax true MACs = 0，以 operations 與 cycle floor 表示 row-wise FSM latency。
6. MLP 使用 GELU page cache + DDR streaming，而不是完整 hidden_gelu on-chip buffer。

---

### `vit_profiler/roofline.py`

負責產生所有圖表。主要功能包含：

| Function | 輸出 / 功用 |
|---|---|
| `plot_roofline()` | DRAM roofline plot，可畫 section-level 或 group-level |
| `plot_bar_compare()` | Baseline vs Optimized bar chart |
| `plot_memory_access_compare()` | DRAM / BRAM access comparison |
| `plot_memory_usage_compare()` | DRAM / BRAM peak usage comparison |
| `plot_norm_operations_compare()` | Norm modeled operations comparison |
| `plot_softmax_operations_compare()` | Softmax modeled operations comparison |
| `plot_norm_softmax_macs_compare()` | Norm / Softmax cycles 或指定 metric comparison |

注意：`plot_roofline()` 預設會排除 `Norm` 和 `Softmax`，因為它們不是 GEMM section，用 MAC-based DRAM roofline 會比較容易誤解。

---

### `vit_profiler/optimization_analysis.py`

負責整理六項優化對 performance metrics 的影響。包含：

| Optimization | 主要影響 |
|---|---|
| INT8 QAT | DRAM/BRAM bytes、energy |
| LayerNorm to Streaming RMSNorm + LUT | Norm operations、cycles、LUT accesses |
| Softmax LUT | Softmax cycles、DRAM reduction、LUT accesses |
| Operator Fusion / page streaming | MLP / Output Projection intermediate materialization |
| Ping-Pong Buffer | GEMM-like section latency cycles |
| DSP Data Packing | effective MACs/cycle、compute cycles |

會輸出 `optimization_impact.csv`、`optimization_impact.md` 和 `optimization_impact.png`。

---

## 3. Output Files

執行後，所有輸出會放在 `--output-dir` 指定的資料夾，例如：

```text
outputs/
```

### 3.1 Model / Config Outputs

| Output file | 說明 |
|---|---|
| `parsed_model_specs.json` | Baseline-B 和 Optimized model 的 parsed shape / spec |
| `hardware_config.json` | 本次 profiling 使用的 hardware config 與 energy config |

---

### 3.2 Profiling Table Outputs

| Output file | 說明 |
|---|---|
| `profiling_results.csv` | 每個 section 的完整 profiling metrics，適合後續用 Excel / pandas 分析 |
| `profiling_results.md` | 和 CSV 相同內容，但輸出成 Markdown table，適合放進 HackMD 報告 |
| `group_summary.csv` | Full MHSA / Full one block 的 group-level summary |
| `group_summary.md` | group summary 的 Markdown table |
| `optimization_impact.csv` | 六項優化對 DRAM、cycles、energy 等 metrics 的影響整理 |
| `optimization_impact.md` | optimization impact 的 Markdown table |

---

### 3.3 Plot Outputs

若沒有使用 `--no-plots`，會產生以下 PNG 圖：

| Output file | 說明 |
|---|---|
| `roofline_sections.png` | Section-level DRAM roofline |
| `roofline_groups.png` | Full MHSA / Full one block group-level DRAM roofline |
| `macs_by_section.png` | True MACs by section；Norm / Softmax 會被排除 |
| `dram_bytes_by_section.png` | DRAM access bytes by section |
| `bram_bytes_by_section.png` | BRAM access bytes by section |
| `dram_usage_by_section.png` | DRAM peak usage by section |
| `bram_usage_by_section.png` | BRAM peak usage by section |
| `cycles_by_section.png` | Cycles by section |
| `cycles_norm_softmax.png` | Norm / Softmax cycles comparison |
| `energy_by_section.png` | Energy by section |
| `dram_bram_access_by_section.png` | DRAM vs BRAM access comparison |
| `dram_bram_usage_by_section.png` | DRAM vs BRAM peak usage comparison |
| `norm_operations_by_section.png` | Norm modeled operations comparison |
| `softmax_operations_by_section.png` | Softmax modeled operations comparison |
| `optimization_impact.png` | Optimization impact summary plot |

---

## 4. Notes for Interpreting Results

1. **Baseline-B is all-DRAM.**  
   Baseline-B 使用與 RTL 相同的 8×8 tiling，但假設沒有 BRAM/local buffer reuse，因此 activation / weight / output tile traffic 都視為 DRAM access。

2. **Optimized is RTL-aware.**  
   Optimized 使用 INT8 activation / weight、BRAM reuse、weight ping-pong、DSP packing，以及 LUT-based RMSNorm / Softmax / GELU。

3. **High DRAM OI does not always mean compute-bound.**  
   DRAM roofline 只看 DRAM bandwidth。如果 optimized 的 DRAM traffic 已經很低，但 BRAM access 或 BRAM service cycles 很高，performance 仍可能遠低於 128 MAC/cycle peak。

4. **MLP cycles may still be high.**  
   MLP 的 FC1 / FC2 是一個 block 中最大的 GEMM workload，而且新版 RTL 使用 1K-word GELU page cache + DDR streaming，因此 optimized MLP cycles 不會單純因為 INT8 4× traffic reduction 而等比例下降。

5. **Energy is analytical estimate.**  
   Energy 使用固定 pJ/op 或 pJ/byte 假設，適合比較趨勢，但不代表實際板上功耗。正式功耗仍需搭配 Vivado power report 或 PYNQ 實測。

---

## 5. Example Command

```bash
python run_profile.py \
  --baseline-model vit_small_patch16_224.augreg_in21k_ft_in1k \
  --optimized-checkpoint ./rms_qat_best.pt \
  -o outputs_v3 \
  --clock-mhz 25 \
  --dram-eff 0.50 \
  --bram-service-bpc 8 \
  --dsp-packing 2.0
```

完成後會看到：

```text
Done. Results saved to: <output directory>
Parsed model specs:
...
```
