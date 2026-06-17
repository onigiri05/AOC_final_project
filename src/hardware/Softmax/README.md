# Softmax Unit for MHSA

本資料夾實作一個用於 Vision Transformer Multi-Head Self-Attention（MHSA）的硬體 Softmax 單元，並附有可在 Vivado 執行的 self-checking testbench。

Softmax Unit 接收一列共 208 個 `INT32` attention scores。輸入為 `Q_INT8 × K_INT8` 的累加結果，模組會先在原始 INT32 domain 找出 row maximum，再執行 max subtraction、power-of-two scaling、exponential LUT 與 normalization，最後輸出 signed INT8 Q0.7 attention weights。

## 檔案說明

| 檔案 | 說明 |
|---|---|
| `Softmax_Unit_0616_2016.sv` | Softmax Unit RTL 實作 |
| `tb_Softmax_Unit_Q57.sv` | Vivado behavioral simulation testbench |

## Module I/O

```systemverilog
module Softmax_Unit (
    input  logic clk,
    input  logic rst_n,
    input  logic start,

    input  logic signed [31:0] score_row [0:207],
    input  logic [5:0] q_shift,
    input  logic [5:0] k_shift,
    input  logic [207:0] mask,

    output logic signed [7:0] attention_row [0:207],
    output logic done
);
```

### 輸入

- `score_row[0:207]`：`Q_INT8 × K_INT8` 的 signed INT32 累加結果。
- `q_shift`：Q 的 power-of-two quantization shift，`q_scale = 2^(-q_shift)`。
- `k_shift`：K 的 power-of-two quantization shift，`k_scale = 2^(-k_shift)`。
- `mask[207:0]`：
  - `1`：有效 token。
  - `0`：padding token，不參與 max、exp sum 與 normalization，輸出固定為 0。
- `start`：拉高一個 clock cycle，啟動一次 Softmax 運算。
- `rst_n`：active-low reset。

### 輸出

- `attention_row[0:207]`：Softmax 結果，格式為 signed INT8 Q0.7。
- `done`：完整一列運算完成時拉高一個 clock cycle。

## 運算流程

本設計先在原始 INT32 score domain 找出有效 token 的 row maximum：

```text
row_max = max(score_row[i]), where mask[i] = 1
```

接著先減去 row maximum，再做 scaling：

```text
score_delta = score_row - row_max
```

實數 attention score difference 為：

```text
score_real = score_delta / 2^(q_shift + k_shift + 3)
```

其中 `+3` 對應 MHSA 的：

```text
1 / sqrt(64) = 1 / 8 = 2^-3
```

為了保留 exponential LUT 的小數精度，score difference 轉為 signed Q5.7：

```text
score_q57
    = score_delta × 2^7 / 2^(q_shift + k_shift + 3)
```

若 `q_shift = 4`、`k_shift = 4`：

```text
score_q57 = score_delta >>> 4
```

完整流程如下：

```text
Q_INT8 × K_INT8 accumulation
        ↓
Find raw INT32 row maximum
        ↓
score_delta = score - row_max
        ↓
Convert to signed Q5.7
        ↓
Clamp to [-12, 0]
        ↓
Exponential LUT
        ↓
Accumulate exp_sum
        ↓
Normalize to signed INT8 Q0.7
```

## Fixed-point formats

| 資料 | 格式 | 說明 |
|---|---|---|
| `score_row` | signed INT32 | Q×K 累加結果 |
| `scaled_score` | signed Q5.7, 12-bit | LUT input，範圍限制在 `[-12, 0]` |
| `exp_value` | unsigned UQ1.15, 16-bit | `round(exp(x) × 32768)` |
| `attention_row` | signed Q0.7, 8-bit | Softmax probability × 128 |

當 Softmax probability 等於 1 時，signed INT8 無法表示 128，因此輸出會飽和為 127。

## Exponential LUT

LUT input 為 signed Q5.7，step 為：

```text
2^-7 = 1/128 = 0.0078125
```

有效輸入範圍：

```text
[-12, 0]
```

對應 address：

```text
address 0    -> x = 0
address 128  -> x = -1
address 256  -> x = -2
...
address 1536 -> x = -12
```

LUT output 定義為：

```text
exp_lut_q57[address]
    = round(exp(-address / 128) × 32768)
```

例如：

```text
x =  0 -> 32768
x = -1 -> 約 12055
x = -2 -> 約 4435
x = -3 -> 約 1632
```

低於 `-12` 的輸入會先被 clamp 成 `-12`。

## FSM 狀態

| State | 功能 |
|---:|---|
| 0 | 等待 `start` |
| 1 | 在有效 raw INT32 scores 中尋找 row maximum |
| 2 | 計算 `score-row_max`、套用 scale 並轉成 signed Q5.7 |
| 3 | 查詢 exponential LUT 並累加 `exp_sum` |
| 4 | 執行 normalization，輸出 signed INT8 Q0.7 |
| 5 | 拉高 `done` 一個 clock cycle |

State 1 到 State 4 每個 clock 處理一個元素，因此一列 208 個 scores 的主要運算時間約為：

```text
208 × 4 + 1 = 833 cycles
```

不包含 `start` 前後的控制週期。

## Testbench 測試內容

`tb_Softmax_Unit_Q57.sv` 為 self-checking testbench，預設 clock period 為 10 ns，也就是 100 MHz。

包含以下測試：

1. **四個相同 score**
   - 預期 Softmax 為 `[0.25, 0.25, 0.25, 0.25]`。
   - Q0.7 輸出預期為 `[32, 32, 32, 32]`。

2. **已知 score differences**
   - 使用實數差值 `[0, -1, -2, -3]`。
   - 驗證 Q5.7 code 與 exponential normalization。
   - 預期 Q0.7 輸出約為 `[82, 30, 11, 4]`。

3. **Mask 測試**
   - masked position 即使具有很大的 score，也不得影響 row maximum 或 denominator。
   - 兩個相同有效 score 的輸出預期為 `[64, 64]`。

4. **全部 masked**
   - 驗證不會發生除以 0。
   - 所有輸出應為 0。

測試成功時 Tcl Console 會顯示：

```text
ALL TESTS PASSED
```

## 注意事項

- 本版本固定支援一列 208 個 scores。
- Q、K scale 必須是 power-of-two 格式，並由 `q_shift`、`k_shift` 表示。
- Q5.7 conversion 目前使用算術 shift，屬於 truncation，尚未加入 round-to-nearest。
- exponential LUT 目前以大型 combinational `case` 實作；behavioral simulation 可正常使用，但 synthesis 後可能被實作成大量 LUT logic，而非 BRAM。
- normalization 中使用 `/` 除法，synthesis 時可能產生較大的 divider。
- 此版本主要用於功能驗證；後續可將 exponential LUT 改為 ROM/BRAM，並以 reciprocal LUT 加乘法取代除法。
