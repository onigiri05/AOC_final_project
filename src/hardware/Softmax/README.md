# Softmax Unit 與 RTL 驗證說明

本資料夾包含一個 Softmax RTL 模組，以及 Level 1～Level 3 的自動比對 testbench。設計主要用於 ViT / MHSA 中單一 attention score row 的 softmax 計算與 FPGA 前期 RTL 驗證。

## 檔案說明

### `softmax_unit.sv`

主要流程：

1. 將輸入的 INT32 score 做算術右移 3 bit，實作除以 `sqrt(64) = 8`。
2. 在有效 token 中尋找 row maximum。
3. 計算 `int_diff = shifted_score - row_max`。
4. 將差值映射至 1024-entry exponential LUT。
5. 累加所有 exponential value，得到 softmax denominator。
6. 正規化並輸出 signed INT8 Q0.7 attention probability。

主要介面：

| 訊號                   | 方向   | 說明                                    |
| ---------------------- | ------ | --------------------------------------- |
| `clk`                  | input  | 時脈                                    |
| `rst_n`                | input  | Active-low reset                        |
| `start`                | input  | 拉高一個 clock 啟動一次 row softmax     |
| `score_row[0:207]`     | input  | 208 個 signed INT32 attention scores    |
| `q_shift`              | input  | Q 的 power-of-two scale shift，預設為 4 |
| `k_shift`              | input  | K 的 power-of-two scale shift，預設為 4 |
| `mask[207:0]`          | input  | `1` 表示有效 token，`0` 表示 padding    |
| `attention_row[0:207]` | output | signed INT8 Q0.7 softmax 結果           |
| `done`                 | output | 結果完成脈衝，維持一個 clock            |

模組需要下列 exponential LUT：

```text
exp_lut_10bit_Q1_15_range12.hex
```

LUT 規格：

- 深度：1024 entries
- 位寬：16 bit
- 輸入範圍：`[-12, 0]`
- 輸出格式：unsigned Q1.15
- 建議公式：`round(exp(-index / 85.25) * 2^15)`

---

### `tb_Softmax_Unit_Level1.sv`

Level 1 單列測試。

測試範圍：

- 單一 attention head
- 單一 query row
- 208 個 key positions

此 testbench 會比較：

- Shifted score
- Row maximum
- Exponential LUT output
- Exponential sum
- 最終 Q0.7 attention output

---

### `tb_Softmax_Unit_Level2.sv`

Level 2 單一 head 完整測試。

測試範圍：

- 1 個 attention head
- 197 個 query rows
- 每個 row padding 至 208 個 key positions

此 testbench 逐列載入 golden data，並比較：

- Shifted score
- Row maximum
- Exponential LUT output
- Exponential sum
- 最終 attention matrix

測試完成後會輸出各階段 mismatch 統計。

---

### `tb_Softmax_Unit_Level3.sv`

Level 3 六個 head 完整測試。

測試範圍：

- 6 個 attention heads
- 每個 head 197 個 query rows
- 總列數：`6 × 197 = 1182`
- 每列 208 個 key positions

此 testbench 主要比較：

- Row maximum
- Exponential sum
- 最終 Q0.7 attention output

## Golden HEX 檔案

依 testbench 層級，模擬時需要下列檔案：

| 檔案                              | 內容                                        |
| --------------------------------- | ------------------------------------------- |
| `score_int32_padded.hex`          | 原始 INT32 score，每列 padding 至 208       |
| `mask.hex`                        | Level 1 使用的 mask                         |
| `mask_padded.hex`                 | Level 2 / Level 3 使用的 flattened mask     |
| `scores_shifted_int32_padded.hex` | `score >>> 3` 的結果                        |
| `max_score_shifted_int32.hex`     | 每列 shifted score 的最大值                 |
| `int_diff_int32_padded.hex`       | `shifted_score - row_max`                   |
| `lut_index_uint10_padded.hex`     | 10-bit exponential LUT address              |
| `exp_uq15_padded.hex`             | LUT 輸出的 unsigned Q1.15 exponential value |
| `exp_sum_uint32.hex`              | 每列 exponential value 的總和               |
| `attention_q07_padded.hex`        | 最終 signed INT8 Q0.7 attention output      |
| `exp_lut_10bit_Q1_15_range12.hex` | RTL 使用的 exponential ROM                  |

```text
跑simulation時記得改 `$readmemh()` 的檔案名稱或改成絕對路徑。
```

## testbench 預期輸出

成功時會看到類似：

```text
LEVEL 1 PASS: all 208 attention values match.
```

```text
LEVEL 2 PASS: all 197 rows and all internal stages match.
```

```text
LEVEL 3 PASS: all 1182 rows matched.
```

若 golden data 與 RTL 不一致，testbench 會顯示 row、key index、expected value 與 actual value。
