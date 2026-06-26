# Streaming RMSNorm Unit README

## 1. 功能說明

`Streaming_RMSNorm_Unit` 負責把 RMSNorm input activation 做正規化，輸出下一級 Systolic Array 可使用的 INT8 activation stream。

計算式：

```text
y[t,c] = clamp_int8((x[t,c] × inv_rms[t] × gamma[c]) >>> (2 × FRAC + OUT_SHIFT))
```

目前資料規格：

```text
TOKEN_NUM   = 197
CHANNEL_NUM = 384
TOTAL_ELEMS = 197 × 384 = 75648

x_in        = signed INT8
inv_rms     = unsigned 16-bit fixed-point, FRAC=14
gamma       = signed 16-bit fixed-point, FRAC=14
y_out       = signed INT8
```

`Streaming_RMSNorm_Unit` 目前這份 standalone 版本的外部 input / output 都是 signed INT8 bit pattern，不做 zero-point 128 轉換。

```text
signed INT8 x_in
→ RMSNorm calculation
→ signed INT8 y_out
```

`Streaming_RMSNorm_Unit` 每次輸出：

```text
1 個 signed INT8 y_out
```

若後面要接 Activation BRAM / Systolic，會透過 RowPacker 先 pack 成 32-bit BRAM word：

```text
act_wr_data_o[31:0] = 4 個 INT8
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
        act_wr_data_o[31:0], act_wr_byte_en_o[3:0], act_wr_addr_o, act_wr_valid_o

Activation BRAM
        ↓
        act_bram_data[31:0]

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
| `x_in`    | input  |  8 | GLB / Activation Buffer | signed INT8 activation |

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
| `y_out`   | output |  8 | RowPacker | signed INT8 normalized activation |

每當：

```text
y_valid && y_ready
```

代表一筆 `y_out` 被 RowPacker 接收。

---

## 4. RowPacker IO Port

RowPacker 的功能是把 RMSNorm 的 8-bit stream pack 成 Activation BRAM 使用的 32-bit word。

```text
4 筆 y_out[7:0]
→ 1 筆 act_wr_data_o[31:0]
```

RowPacker 內有 `SIGNED_TO_ZP128` 參數：

```text
SIGNED_TO_ZP128 = 1: signed INT8 轉成 uint8 zero-point 128 後 pack
SIGNED_TO_ZP128 = 0: 保留 signed INT8 bit pattern 直接 pack
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
| `act_wr_addr_o`    | output | `ADDR_W` | Activation BRAM       | 32-bit word write address |
| `act_wr_data_o`    | output |       32 | Activation BRAM       | packed 4 INT8 activation  |
| `act_wr_byte_en_o` | output |        4 | Activation BRAM       | byte enable               |
| `act_wr_last_o`    | output |        1 | Controller / optional | 最後一個 packed word      |

packed word 格式：

```text
act_wr_data_o[7:0]    = 第 0 個 activation
act_wr_data_o[15:8]   = 第 1 個 activation
act_wr_data_o[23:16]  = 第 2 個 activation
act_wr_data_o[31:24]  = 第 3 個 activation
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
act_wr_data_o[31:0]
act_wr_byte_en_o[3:0]
```

---

### Activation BRAM Read Side

來源：Systolic.v

| Signal           | 方向              |  寬度 | 說明                              |
| ---------------- | --------------- | --: | ------------------------------- |
| `act_bram_addr`  | Systolic output | 17 | Systolic 要讀的 activation word address |
| `act_bram_valid` | Systolic input  |  1 | BRAM read data valid                    |
| `act_bram_data`  | Systolic input  | 32 | 4 個 INT8 activation                    |

連接方式：

```text
Systolic.act_bram_addr → Activation BRAM read address
Activation BRAM rdata  → Systolic.act_bram_data[31:0]
Activation BRAM valid  → Systolic.act_bram_valid
```

---

## 6. Activation BRAM Address Layout

RowPacker 的 `base_addr_i` 是 32-bit word address。因為每 16 個 channels 會拆成 4 個 32-bit words，所以同一個 `(m_tile, k_tile)` 的 16 個 token row 會各自對應 4 個 word address。

定義：

```text
m_tile   = token_idx / 16
m_inner  = token_idx % 16
k_tile   = channel_idx / 16
word_sel = (channel_idx % 16) / 4
```

對 ViT-Small/16：

```text
CHANNEL_NUM  = 384
CHANNEL_TILE = 16
TOKEN_TILE   = 16
K_TILE_NUM   = 384 / 16 = 24
```

write address：

```text
row_addr  = m_tile × (K_TILE_NUM × TOKEN_TILE)
          + k_tile × TOKEN_TILE
          + m_inner

word_addr = base_addr + row_addr × 4 + word_sel
```

這樣同一個 `(m_tile, k_tile)` 的 16 channels 會拆成 4 個 32-bit words，並依照 systolic tile 需要的順序排列。

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
act_wr_data[31:0]
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
