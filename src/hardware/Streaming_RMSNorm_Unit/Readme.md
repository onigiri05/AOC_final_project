# Streaming RMSNorm Unit README

## 1. Overview

`Streaming_RMSNorm_Unit` 是 ViT Accelerator 中用來執行 RMSNorm 的硬體模組。

此 module 的目標是將輸入 activation stream 逐筆正規化，並輸出 INT8 normalized activation stream 給下一級 PE Array 前的 Activation FIFO。

本設計對應 ViT-Small/16 的 activation shape：

```text
TOKEN_NUM   = 197
CHANNEL_NUM = 384
```

也就是輸入資料順序為：

```text
token 0, channel 0 ~ 383
token 1, channel 0 ~ 383
...
token 196, channel 0 ~ 383
```

RMSNorm 計算公式為：

```text
y[t,c] = clamp_int8((x[t,c] * inv_rms[t] * gamma[c]) >>> SHIFT)
```

其中：

```text
SHIFT = 2 * FRAC + OUT_SHIFT
```

---

## 2. Module Hierarchy

本檔案包含兩個主要 module：

```text
Streaming_RMSNorm_Unit
└── Streaming_RMSNorm_Core
```

### 2.1 Streaming_RMSNorm_Core

`Streaming_RMSNorm_Core` 負責單一 activation element 的 RMSNorm 運算。

輸入：

```text
x_in        : signed INT8 activation
inv_rms_in  : unsigned 16-bit fixed-point
gamma_in    : signed 16-bit fixed-point
```

輸出：

```text
y_out       : signed INT8 normalized activation
```

Core 內部是 3-stage pipeline：

```text
Stage 1: x_in * inv_rms_in
Stage 2: result * gamma_in
Stage 3: arithmetic shift + clamp to INT8
```

---

### 2.2 Streaming_RMSNorm_Unit

`Streaming_RMSNorm_Unit` 是外層控制 module，負責：

```text
1. start / busy / done 控制
2. token counter
3. channel counter
4. Token Stat SRAM read address generation
5. Gamma Buffer read address generation
6. input valid/ready handshake
7. output valid/ready handshake
8. y_last generation
```

address 對應方式：

```text
inv_rms_addr = token_cnt
gamma_addr   = channel_cnt
```

---

## 3. Fixed-Point Format

目前暫定格式如下：

```text
x_in       : signed INT8
inv_rms    : unsigned 16-bit fixed-point
gamma      : signed 16-bit fixed-point
FRAC       : 14
OUT_SHIFT  : 0
```

因此目前預設：

```text
FRAC = 14
SHIFT = 2 * FRAC + OUT_SHIFT = 28
```

計算方式：

```text
prod1 = x_in * inv_rms
prod2 = prod1 * gamma
scaled = prod2 >>> SHIFT
y_out = clamp_int8(scaled)
```

注意：`FRAC=14` 目前只是暫定值，最後需要根據 software calibration 決定。若 software 最後選擇不同 Q format，只需要調整：

```systemverilog
`define RMS_FRAC
`define RMS_OUT_SHIFT
```

---

## 4. Default Parameters

目前預設參數如下：

```systemverilog
`define RMS_TOKEN_NUM       197
`define RMS_CHANNEL_NUM     384
`define RMS_X_W             8
`define RMS_SCALE_W         16
`define RMS_FRAC            14
`define RMS_OUT_SHIFT       0
```

| Parameter     | Description                            |
| ------------- | -------------------------------------- |
| `TOKEN_NUM`   | token 數量，ViT-Small/16 為 197            |
| `CHANNEL_NUM` | embedding dimension，ViT-Small/16 為 384 |
| `X_W`         | activation bitwidth，目前為 INT8           |
| `SCALE_W`     | inv_rms / gamma bitwidth，目前為 16-bit    |
| `FRAC`        | fixed-point 小數位數，目前暫定 14               |
| `OUT_SHIFT`   | 額外 output scaling shift，目前暫定 0         |

---

## 5. Input / Output Interface

### 5.1 Control Signals

| Signal  | Direction | Description                     |
| ------- | --------- | ------------------------------- |
| `clk`   | input     | clock                           |
| `rst_n` | input     | active-low reset                |
| `start` | input     | 開始處理一整個 RMSNorm stream          |
| `busy`  | output    | module 正在運作                     |
| `done`  | output    | 最後一筆 output 被接收後 pulse 一個 cycle |

---

### 5.2 Input Activation Stream

| Signal    | Direction | Description            |
| --------- | --------- | ---------------------- |
| `x_valid` | input     | 上游資料有效                 |
| `x_ready` | output    | module 可以接收資料          |
| `x_in`    | input     | signed INT8 activation |

input activation 來源通常是：

```text
Activation-Residual Buffer / Global Buffer
```

資料順序必須是 token-major：

```text
token 0 channel 0
token 0 channel 1
...
token 0 channel 383
token 1 channel 0
...
token 196 channel 383
```

---

### 5.3 Token Stat SRAM Interface

| Signal         | Direction | Description             |
| -------------- | --------- | ----------------------- |
| `inv_rms_addr` | output    | Token Stat SRAM address |
| `inv_rms_data` | input     | inv_rms[t]              |

`inv_rms_addr` 由 `token_cnt` 產生：

```text
inv_rms_addr = token_cnt
```

代表同一個 token 的 384 個 channels 都共用同一個 `inv_rms[t]`。

---

### 5.4 Gamma Buffer Interface

| Signal       | Direction | Description          |
| ------------ | --------- | -------------------- |
| `gamma_addr` | output    | Gamma Buffer address |
| `gamma_data` | input     | gamma[c]             |

`gamma_addr` 由 `channel_cnt` 產生：

```text
gamma_addr = channel_cnt
```

代表不同 channel 使用不同的 learnable scale parameter `gamma[c]`。

---

### 5.5 Output Stream

| Signal    | Direction | Description                       |
| --------- | --------- | --------------------------------- |
| `y_valid` | output    | output data valid                 |
| `y_ready` | input     | 下游可以接收資料                          |
| `y_last`  | output    | 最後一筆 output                       |
| `y_out`   | output    | signed INT8 normalized activation |

output 目的地通常是：

```text
Activation FIFO / PE input buffer
```

本 module 不需要把 normalized activation 完整寫回大的 GLB，因為 Streaming RMSNorm 的目的就是讓 RMSNorm output 直接 stream 到下一級 GEMM，以減少中間資料搬移。

---

## 6. Dataflow

整體資料流如下：

```text
Activation-Residual Buffer / GLB
        |
        | x[t,c]
        v
Streaming_RMSNorm_Unit
        |
        | read inv_rms[t] from Token Stat SRAM
        | read gamma[c] from Gamma Buffer
        v
Streaming_RMSNorm_Core
        |
        | x[t,c] * inv_rms[t] * gamma[c]
        | shift + clamp
        v
INT8 normalized activation stream
        |
        v
Activation FIFO / PE Array input
```

---

## 7. Handshake Behavior

本 module 使用 valid/ready protocol。

### Input side

一筆 input 真的被接收的條件是：

```text
x_valid && x_ready
```

只有在這個條件成立時，`token_cnt` 和 `channel_cnt` 才會前進。

### Output side

一筆 output 真的被下游接收的條件是：

```text
y_valid && y_ready
```

如果 `y_ready = 0`，pipeline 會 stall，避免 output data 被覆蓋或遺失。

### Last output

最後一筆 output 對應：

```text
token_cnt   = TOKEN_NUM - 1
channel_cnt = CHANNEL_NUM - 1
```

當最後一筆 output 被接收時：

```text
y_valid && y_ready && y_last
```

module 會：

```text
busy <= 0
done <= 1 for one cycle
```

---

## 8. Pipeline Latency

`Streaming_RMSNorm_Core` 是 3-stage pipeline：

```text
Stage 1: x * inv_rms
Stage 2: result * gamma
Stage 3: shift + clamp
```

因此在沒有 stall 的情況下，第一筆 input 進入後，需要約 3 個 pipeline stages 後才會看到第一筆 output。

steady-state 情況下，若 `x_valid = 1` 且 `y_ready = 1`，module 可以做到：

```text
1 input element / cycle
1 output element / cycle
```

---

## 12. Future Work

後續需要確認或補強的項目：

```text
1. 由 software calibration 決定最終 FRAC / Q format
2. 由 quantization flow 決定 OUT_SHIFT
3. 補上 synchronous BRAM read latency alignment wrapper
4. 使用真實 ViT activation / gamma / inv_rms 測完整 197 × 384 case
5. 和下一級 Activation FIFO / PE Array 做 integration test
6. 和 RMS Stat Accumulator + Inv-Sqrt LUT + Token Stat SRAM 做 fused path test
```


