# Streaming RMSNorm Unit README

## 1. 功能說明

`Streaming_RMSNorm_Unit` 負責把 RMSNorm input activation 做正規化，輸出下一級 Systolic Array 可使用的 INT8 activation stream。

計算式：

```text
y[t,c] = x[t,c] × inv_rms[t] × gamma[c]
```

目前資料規格：

```text
TOKEN_NUM   = 197
CHANNEL_NUM = 384

x_in        = unsigned INT8 zero-point 128
inv_rms     = 16-bit fixed-point
gamma       = 16-bit fixed-point
y_out       = unsigned INT8 zero-point 128
```

`Streaming_RMSNorm_Unit` 外部 input / output 使用 unsigned INT8 zero-point 128，內部計算時會轉成 signed INT8 centered value。

```text
external x_in uint8 zero-point 128
→ internal signed INT8
→ RMSNorm calculation
→ external y_out uint8 zero-point 128
```

`Streaming_RMSNorm_Unit` 每次輸出：

```text
1 個 INT8 y_out
```

但 Systolic Array 的 activation input 是：

```text
act_bram_row[127:0] = 16 個 INT8
```

所以中間需要：

```text
Streaming_RMSNorm_Unit
→ Streaming_RMSNorm_RowPacker
→ Activation BRAM
→ Systolic.v
```

---

## 2. 整體連接關係

```text
Activation-Residual Buffer / GLB
        ↓
        x_in[7:0], x_valid, x_ready

Streaming_RMSNorm_Unit
        ↓
        y_out[7:0], y_valid, y_ready, y_last

Streaming_RMSNorm_RowPacker
        ↓
        act_wr_row[127:0], act_wr_addr, act_wr_valid

Activation BRAM
        ↓
        act_bram_row[127:0]

Systolic.v
```

另外 `Streaming_RMSNorm_Unit` 會讀兩個 buffer：

```text
Token Stat SRAM  → 提供 inv_rms[t]
Gamma Buffer     → 提供 gamma[c]
```

---

## 3. Streaming_RMSNorm_Unit IO Port

### Control

| Signal  | 方向     | 說明                                    |
| ------- | ------ | ------------------------------------- |
| `clk`   | input  | clock                                 |
| `rst_n` | input  | active-low reset                      |
| `start` | input  | controller 拉高 1 cycle，開始處理一整個 RMSNorm |
| `busy`  | output | RMSNorm 正在運作                          |
| `done`  | output | 最後一筆 output 被接收後 pulse 1 cycle        |

---

### Input Activation Stream

| Signal    | 方向     | 寬度 | 連接對象                    | 說明                                       |
| --------- | ------ | -: | ----------------------- | ---------------------------------------- |
| `x_valid` | input  |  1 | GLB / Activation Buffer | input activation 有效                      |
| `x_ready` | output |  1 | GLB / Activation Buffer | RMSNorm 可以接收資料                           |
| `x_in`    | input  |  8 | GLB / Activation Buffer | unsigned INT8 activation, zero-point 128 |

資料順序必須是 token-major：

```text
token0 ch0
token0 ch1
...
token0 ch383
token1 ch0
...
token196 ch383
```

每當：

```text
x_valid && x_ready
```

代表 RMSNorm 接收一筆 `x_in`。

---

### Token Stat SRAM Interface

| Signal         | 方向     |         寬度 | 連接對象            | 說明                   |
| -------------- | ------ | ---------: | --------------- | -------------------- |
| `inv_rms_addr` | output | `TOKEN_AW` | Token Stat SRAM | 讀取 inv_rms 的 address |
| `inv_rms_data` | input  |         16 | Token Stat SRAM | `inv_rms[t]`         |

address 對應：

```text
inv_rms_addr = token_cnt
```

也就是同一個 token 的 384 個 channel 共用同一個 `inv_rms[t]`。

---

### Gamma Buffer Interface

| Signal       | 方向     |           寬度 | 連接對象         | 說明                 |
| ------------ | ------ | -----------: | ------------ | ------------------ |
| `gamma_addr` | output | `CHANNEL_AW` | Gamma Buffer | 讀取 gamma 的 address |
| `gamma_data` | input  |           16 | Gamma Buffer | `gamma[c]`         |

address 對應：

```text
gamma_addr = channel_cnt
```

也就是每個 channel 讀自己的 `gamma[c]`。

---

### Output Stream

| Signal    | 方向     | 寬度 | 連接對象      | 說明                                                  |
| --------- | ------ | -: | --------- | --------------------------------------------------- |
| `y_valid` | output |  1 | RowPacker | output 有效                                           |
| `y_ready` | input  |  1 | RowPacker | RowPacker 可以接收資料                                    |
| `y_last`  | output |  1 | RowPacker | 最後一筆 RMSNorm output                                 |
| `y_out`   | output |  8 | RowPacker | unsigned INT8 normalized activation, zero-point 128 |

每當：

```text
y_valid && y_ready
```

代表一筆 `y_out` 被 RowPacker 接收。

---

## 4. RowPacker IO Port

RowPacker 的功能是把 RMSNorm 的 8-bit stream pack 成 Systolic 需要的 128-bit row。

```text
16 筆 y_out[7:0]
→ 1 筆 act_wr_row[127:0]
```

### Input from Streaming_RMSNorm_Unit

| Signal      | 方向     | 寬度 | 連接來源      |
| ----------- | ------ | -: | --------- |
| `s_data_i`  | input  |  8 | `y_out`   |
| `s_valid_i` | input  |  1 | `y_valid` |
| `s_ready_o` | output |  1 | `y_ready` |
| `s_last_i`  | input  |  1 | `y_last`  |

連接方式：

```systemverilog
assign packer_s_data_i  = rms_y_out;
assign packer_s_valid_i = rms_y_valid;
assign rms_y_ready      = packer_s_ready_o;
assign packer_s_last_i  = rms_y_last;
```

---

### Output to Activation BRAM

| Signal           | 方向     |       寬度 | 連接對象                  | 說明                        |
| ---------------- | ------ | -------: | --------------------- | ------------------------- |
| `act_wr_valid_o` | output |        1 | Activation BRAM       | write data valid          |
| `act_wr_ready_i` | input  |        1 | Activation BRAM       | BRAM 可以接收 write           |
| `act_wr_addr_o`  | output | `ADDR_W` | Activation BRAM       | write address             |
| `act_wr_row_o`   | output |      128 | Activation BRAM       | packed 16 INT8 activation |
| `act_wr_last_o`  | output |        1 | Controller / optional | 最後一個 packed row           |

packed row 格式：

```text
act_wr_row[7:0]       = 第 0 個 activation
act_wr_row[15:8]      = 第 1 個 activation
...
act_wr_row[127:120]   = 第 15 個 activation
```

這個格式要對上 `Act_fifo.v`：

```verilog
act_r[15][i] <= act_row_in[8*i +: 8];
```

---

## 5. Activation BRAM 和 Systolic.v 連接

RowPacker 寫入 Activation BRAM，Systolic 再從 Activation BRAM 讀出。

### Activation BRAM Write Side

來源：RowPacker

```text
act_wr_valid_o
act_wr_ready_i
act_wr_addr_o
act_wr_row_o[127:0]
```

---

### Activation BRAM Read Side

來源：Systolic.v

| Signal           | 方向              |  寬度 | 說明                              |
| ---------------- | --------------- | --: | ------------------------------- |
| `act_bram_addr`  | Systolic output |  17 | Systolic 要讀的 activation address |
| `act_bram_valid` | Systolic input  |   1 | BRAM read data valid            |
| `act_bram_row`   | Systolic input  | 128 | 16 個 INT8 activation            |

連接方式：

```text
Systolic.act_bram_addr → Activation BRAM read address
Activation BRAM rdata  → Systolic.act_bram_row[127:0]
Activation BRAM valid  → Systolic.act_bram_valid
```

---

## 6. Activation BRAM Address Layout

因為 `Systolic.v` 在一個 tile 裡會連續讀 16 row：

```text
act_base_addr + 0
act_base_addr + 1
...
act_base_addr + 15
```

所以 RowPacker 寫入時要讓同一個 `(m_tile, k_tile)` 的 16 個 token row 放在連續 address。

定義：

```text
m_tile  = token_idx / 16
m_inner = token_idx % 16
k_tile  = channel_idx / 16
```

對 ViT-Small/16：

```text
CHANNEL_NUM = 384
CHANNEL_TILE = 16
K_TILE_NUM = 384 / 16 = 24
```

write address：

```text
addr = base_addr
     + m_tile × 384
     + k_tile × 16
     + m_inner
```

這樣 Systolic 讀取時會得到：

```text
act_base_addr + 0  → token m+0, channel k~k+15
act_base_addr + 1  → token m+1, channel k~k+15
...
act_base_addr + 15 → token m+15, channel k~k+15
```

---

## 7. Controller 需要提供的訊號

### 給 Streaming_RMSNorm_Unit

```text
rms_start
x_valid
x_in
inv_rms_data
gamma_data
y_ready
```

並接收：

```text
rms_busy
rms_done
x_ready
inv_rms_addr
gamma_addr
y_valid
y_out
y_last
```

---

### 給 RowPacker

```text
packer_start
act_write_base_addr
act_wr_ready
```

並接收：

```text
act_wr_valid
act_wr_addr
act_wr_row[127:0]
act_wr_last
```

---

### 給 Systolic.v

Controller 要等 RowPacker 把需要的 activation tile 寫好後，再啟動 Systolic。

需要設定：

```text
systolic_en
act_base_addr
w_base_addr
k_tile_cnt
```

Systolic 回傳：

```text
module_ready
opsum_valid
opsum[31:0]
```
