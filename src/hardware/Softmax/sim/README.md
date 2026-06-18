---
title: Softmax FPGA test
---

# Softmax RTL Golden Package

本文件對應目前 FPGA 驗證使用的版本：

- RTL：`Softmax_Unit.sv`
- Golden package：`softmax_pow2lut12_validation_package`
- LUT：`exp_lut_10bit_Q1_15_range12.hex`
- Softmax 輸出：signed INT8 Q0.7
- LUT 輸出：unsigned UQ1.15，16 bits

---

## 1. 設定

| 項目                    |               值 |
| ----------------------- | ---------------: |
| Heads                   |                6 |
| Tokens                  |              197 |
| Head dimension          |               64 |
| RTL row size            |              208 |
| Valid key positions     |              197 |
| Padding key positions   |               11 |
| RTL `q_shift`           |                4 |
| RTL `k_shift`           |                4 |
| Score shift             |                3 |
| LUT entries             |             1024 |
| LUT input range         |       `[-12, 0]` |
| LUT output format       |  unsigned UQ1.15 |
| Attention output format | signed INT8 Q0.7 |

目前 golden data 是以：

```text
q_shift = 4
k_shift = 4
```

產生。

---

## 2. RTL bit-accurate 流程

```text
Q_INT8 × K_INT8^T
        ↓
signed INT32 accumulator
        ↓
scores_shifted = score_int32 >>> 3
        ↓
在 shifted INT32 domain 找 row maximum
        ↓
int_diff = scores_shifted - max_scores_shifted
        ↓
diff_magnitude = -int_diff
        ↓
lut_idx = clamp(
    (diff_magnitude × 341)
    >> (q_shift + k_shift + 2),
    0,
    1023
)
        ↓
查詢 1024-entry exp LUT
        ↓
unsigned UQ1.15 exponential values
        ↓
integer exp_sum
        ↓
integer normalization
        ↓
signed INT8 Q0.7 attention
```

---

## 3. 測試層級

### Level 1：單一 score row

測試範圍：

```text
head 0
query row 0
208 個 key positions
```

資料 shape：

```text
[208]
```

可檢查項目：

```text
score input
scores_shifted
row max
int_diff
LUT index
exp output
exp sum
attention Q0.7
```

---

### Level 2：完整單一 head

測試範圍：

```text
head 0
197 個 query rows
每 row 208 positions
```

資料 shape：

```text
[197, 208]
```

總共執行 Softmax：

```text
197 次
```

---

### Level 3：六個 heads

測試範圍：

```text
6 heads × 197 query rows
```

資料 shape：

```text
[6, 197, 208]
```

總共執行 Softmax：

```text
6 × 197 = 1182 次
```

```text
雖然實際上 Q * K 完含 padding 是 `[208, 208]`，但測試時先不考慮 padding 的 row，因為這可靠 controller 判斷 address 要不要處理。
```

---

## 4. Padding 規格

實際有效 attention matrix：

```text
[197, 197]
```

RTL row 使用：

```text
208 positions
```

因此：

```text
key 0～196   ：valid
key 197～207 ：padding
```

Padding 規則：

```text
mask = 0
exp_value = 0
attention_q07 = 0
```

Level 1、2、3 的 padded golden 都已將最後 11 個位置設為 0。

---

## 5. Level 1 Golden Data

資料夾：

```text
level1_head0_row0/
```

| 檔案                              |   Shape |       Word width | RTL 對應                  | 意義                |
| --------------------------------- | ------: | ---------------: | ------------------------- | ------------------- |
| `score_int32_padded.hex`          | `[208]` |           32-bit | `score_row[index]`        | 原始 QK INT32 input |
| `scores_shifted_int32_padded.hex` | `[208]` |           32-bit | `dut.scaled_score[index]` | `score >>> 3`       |
| `mask.hex`                        | `[208]` |            8-bit | `mask[index]`             | 1=valid，0=padding  |
| `max_score_shifted_int32.hex`     |  scalar |           32-bit | `dut.max_score`           | shifted row maximum |
| `int_diff_int32_padded.hex`       | `[208]` |           32-bit | subtraction intermediate  | `shifted-max`       |
| `lut_index_uint10_padded.hex`     | `[208]` | 16-bit container | LUT address               | 實際有效為 10-bit   |
| `exp_uq15_padded.hex`             | `[208]` |           16-bit | `dut.exp_value[index]`    | unsigned UQ1.15 exp |
| `exp_sum_uint32.hex`              |  scalar |           32-bit | `dut.exp_sum`             | row denominator     |
| `attention_q07_padded.hex`        | `[208]` |            8-bit | `attention_row[index]`    | signed Q0.7 output  |

---

## 6. Level 2 Golden Data

資料夾：

```text
level2_head0/
```

| 檔案 stem                     |       Shape | 意義                    |
| ----------------------------- | ----------: | ----------------------- |
| `score_int32_padded`          | `[197,208]` | 原始 QK input           |
| `scores_shifted_int32_padded` | `[197,208]` | `score >>> 3`           |
| `mask_padded`                 | `[197,208]` | valid/padding mask      |
| `max_score_shifted_int32`     |     `[197]` | 每 row maximum          |
| `int_diff_int32_padded`       | `[197,208]` | shifted score-minus-max |
| `lut_index_uint10_padded`     | `[197,208]` | LUT addresses           |
| `exp_uq15_padded`             | `[197,208]` | LUT output              |
| `exp_sum_uint32`              |     `[197]` | 每 row exp sum          |
| `attention_q07_padded`        | `[197,208]` | Q0.7 padded output      |
| `attention_q07_valid`         | `[197,197]` | 有效 attention matrix   |

---

## 7. Level 3 Golden Data

資料夾：

```text
level3_all_heads/
```

| 檔案 stem                     |         Shape | 意義                    |
| ----------------------------- | ------------: | ----------------------- |
| `score_int32_padded`          | `[6,197,208]` | 原始 QK input           |
| `scores_shifted_int32_padded` | `[6,197,208]` | `score >>> 3`           |
| `mask_padded`                 | `[6,197,208]` | valid/padding mask      |
| `max_score_shifted_int32`     |     `[6,197]` | 每 head、每 row maximum |
| `int_diff_int32_padded`       | `[6,197,208]` | shifted score-minus-max |
| `lut_index_uint10_padded`     | `[6,197,208]` | LUT addresses           |
| `exp_uq15_padded`             | `[6,197,208]` | LUT output              |
| `exp_sum_uint32`              |     `[6,197]` | 每 row exp sum          |
| `attention_q07_padded`        | `[6,197,208]` | padded Q0.7 output      |
| `attention_q07_valid`         | `[6,197,197]` | 有效 attention matrix   |

Level 3 flatten 順序：

```text
[head, query, key]
```

---

## 8. Memory 資源建議

| 資料                                | 建議資源 |
| ----------------------------------- | -------- |
| Input score row `[208] × INT32`     | BRAM     |
| Output attention row `[208] × INT8` | BRAM     |
| Exp LUT `1024 × 16`                 | BRAM     |
| `scaled_score[208]`                 | LUTRAM   |
| `exp_value[208]`                    | LUTRAM   |
| `max_score`                         | register |
| `exp_sum`                           | register |
| `mask[208]`                         | LUTRAM   |

LUT 大小：

```text
1024 × 16 = 16384 bits
```

---
