# Softmax Unit for MHSA

本資料夾實作一個可用於 Vision Transformer MHSA 的簡易硬體 Softmax 單元。

Softmax 單元接收一列共 208 個 `INT32` attention scores，完成縮放、mask、最大值搜尋、指數近似與 normalization，最後輸出 `INT8 Q0.7` 格式的 attention weights。

## 檔案說明

| 檔案                         | 說明                   |
| ---------------------------- | ---------------------- |
| `softmax_unit_3.sv`          | Softmax Unit RTL 實作  |
| `tb_softmax_unit_current.sv` | Softmax Unit testbench |

## I/O port

```text
module Softmax_Unit (
    input  logic clk,
    input  logic rst_n,
    input  logic start, // Start softmax signal

    input  logic signed [31:0] score_row [0:207], // Q_INT8 × K_INT8
    input  logic [5:0] q_shift, // q_scale = 2^(-q_shift)
    input  logic [5:0] k_shift, // k_scale = 2^(-k_shift)
    input  logic [207:0] mask, // 1: valid token, 0: padded token, output = 0

    output logic signed [7:0] attention_row [0:207], // Signed INT8 Q0.7

    output logic done
);
```

### 輸入

- `score_row[0:207]`：`Q_INT8 × K_INT8` 的 INT32 累加結果
- `q_shift`：Q 的 power-of-two quantization shift
- `k_shift`：K 的 power-of-two quantization shift
- `mask[207:0]`：有效 token mask
  - `1`：有效 token
  - `0`：padding token，輸出固定為 0
- `start`：拉高一個 clock cycle，啟動一次 Softmax 運算

### 輸出

- `attention_row[0:207]`：Softmax 結果，格式為 signed INT8 Q0.7
- `done`：運算完成時拉高一個 clock cycle

## 運算流程

輸入 score 先進行縮放：

```text
total_shift = q_shift + k_shift + 3
scaled_score = score_row >>> total_shift
```

其中 `+3` 對應 MHSA 中的：

```text
1 / sqrt(64) = 1 / 8
```

接著依序執行：

1. 對所有 score 做算術右移縮放
2. 在有效 token 中找出最大值
3. 計算 `exp(scaled_score - max_score)`
4. 累加所有 exponential values
5. 計算 Softmax probability
6. 轉換成 INT8 Q0.7

Softmax normalization 為：

```text
attention_int8 = round(exp_value / exp_sum × 128)
```

由於 signed INT8 最大值為 127，因此 Softmax 等於 1 時會飽和為 127。

## Exponential LUT

目前使用簡化的整數 LUT：

```text
x =  0  -> exp(x) ≈ 32767
x = -1  -> exp(x) ≈ 12055
x = -2  -> exp(x) ≈ 4435
...
x = -7  -> exp(x) ≈ 30
x <= -8 -> 0
```

LUT 輸出格式為 unsigned Q0.15。

目前 `scaled_score` 為整數，因此小數部分會在算術右移時被捨去。若之後希望提高 Softmax 精度，可將 scaled score 改為 fixed-point，並使用具有 fractional address bits 的 exponential LUT。

## FSM 狀態

| State | 功能                                  |
| ----: | ------------------------------------- |
|     0 | 等待 `start`                          |
|     1 | 計算 scaled score                     |
|     2 | 尋找有效 score 最大值                 |
|     3 | 查詢 exponential LUT 並累加 `exp_sum` |
|     4 | Normalization 並輸出 Q0.7 attention   |
|     5 | 拉高 `done`                           |

每個處理階段一次處理一個元素，因此一列 208 個 score 的完整運算大約需要：

```text
208 × 4 + 1 = 833 cycles
```

不包含啟動前後的額外控制週期。

## Testbench 測試項目

Testbench 包含以下測試：

1. 四個相同 score
2. 不同整數 score 的 exponential mapping
3. Mask 排除大型 score
4. 所有 token 都被 mask
5. `total_shift >= 32` 的特殊情況
6. 單一有效 token與 Q0.7 飽和

模擬成功時會顯示：

```text
PASS: all Softmax_Unit test cases passed.
```

## 注意事項

- 固定支援一列 208 個 score。
- 目前 exponential LUT 只支援整數輸入。
- `/` 除法運算在 synthesis 時可能產生較大的組合除法器。
- 此版本適合功能驗證；後續硬體最佳化可改用 reciprocal LUT、乘法與 pipeline。
