# PPU Residual + RMS Tile Units

本資料夾包含三個 PPU 後半段模組，配合目前專題的 **16×16 systolic array tile dataflow**。

檔案：

```text
Residual_Add_Unit.sv
RMS_Stat_Accumulator.sv
PPU_Residual_RMS_Tail.sv
```

---

## 1. 整體資料流

PPU tail 接在 `Requant_Unit` 後面。

```text
Requant output uint8 tile
        ↓
Residual_Add_Unit
        ↓
X_mid / X_out tile
        ↓
RMS_Stat_Accumulator
        ↓
sum_sq[token]
```

目前 tile 格式為：

```text
16 tokens × 16 channels
```

也就是每次處理一個 `16×16` INT8 activation tile。

---

## 2. Residual_Add_Unit.sv

### 功能

做 shortcut / residual add。

使用在兩個地方：

```text
Attention 後：
X_mid = X + O

FC2 後：
X_out = X_mid + MLP_out
```

### 量化格式

因為前面的 `Requant_Unit` 輸出是：

```text
uint8, zero point = 128
```

所以 residual add 公式為：

```text
q_out = clamp(q_main + q_residual - 128, 0, 255)
```

### 主要 I/O

```verilog
input  logic [16*16*8-1:0] main_tile_i;
input  logic [16*16*8-1:0] residual_tile_i;
output logic [16*16*8-1:0] data_tile_o;
```

說明：

```text
main_tile_i:
    Requant 後的 main branch tile
    可能是 Attention output O 或 FC2 output MLP_out

residual_tile_i:
    shortcut branch tile
    Attention 後是 X
    FC2 後是 X_mid

data_tile_o:
    residual add 後的結果
    Attention 後是 X_mid
    FC2 後是 X_out
```

這個 module 是 combinational，不含 valid / ready。

---

## 3. RMS_Stat_Accumulator.sv

### 功能

在產生 `X_mid` 或 `X_out` 時，同步累加 RMSNorm 下一階段需要的統計量：

```text
sum_sq[token] = Σ x[token, channel]^2
```

因為 activation 是 `uint8 zero point = 128`，所以實際累加：

```text
sum_sq[token] += Σ(q_out - 128)^2
```

### 為什麼需要 partial_sum_mem？

目前資料流是：

```text
Token tile = 16 tokens
Output channel tile = 16 channels
D = 384 channels
```

所以一個 token 的 384 channels 會被切成：

```text
384 / 16 = 24 個 channel tiles
```

因此 `RMS_Stat_Accumulator` 會用：

```text
partial_sum_mem[token]
```

跨 24 個 channel tiles 累加。

當 `channel_tile_idx_i == 23` 時，代表該 token 的 384 channels 都累加完成，輸出完整 `sum_sq`。

### 主要 I/O

```verilog
input  logic                 tile_valid_i;
output logic                 tile_ready_o;

input  logic                 acc_en_i;
input  logic [16*16*8-1:0]   data_tile_i;
input  logic [7:0]           base_token_idx_i;
input  logic [4:0]           channel_tile_idx_i;
input  logic [15:0]          token_valid_mask_i;

output logic                 stat_valid_o;
input  logic                 stat_ready_i;
output logic [7:0]           stat_token_idx_o;
output logic [31:0]          sum_sq_o;
```

說明：

```text
tile_valid_i:
    上游送來的 16×16 tile 有效

tile_ready_o:
    本 module 可以接收 tile

acc_en_i:
    是否啟用 RMS statistic 累加
    Attention 後與 FC2 後啟用
    FC1 後關閉

data_tile_i:
    X_mid 或 X_out 的 16×16 tile

base_token_idx_i:
    目前 tile 的第一個 token index
    例如 m_tile = 0 時是 0
    m_tile = 1 時是 16

channel_tile_idx_i:
    目前是第幾個 channel tile
    範圍 0~23

token_valid_mask_i:
    16 個 token row 哪些有效
    最後一個 token tile 可能有 padding，因此需要 mask

stat_valid_o:
    sum_sq_o 有效

stat_ready_i:
    下游 Token Stat SRAM / LUT 接收端 ready

stat_token_idx_o:
    目前輸出的 token index

sum_sq_o:
    該 token 的完整 Σ(q-128)^2
```

---

## 4. PPU_Residual_RMS_Tail.sv

### 功能

整合：

```text
Residual_Add_Unit
RMS_Stat_Accumulator
```

根據 `ppu_mode_i` 決定資料路徑。

### Mode 說明

```text
ppu_mode_i = 2'b00：Attention output
    main_tile_i     = Attention output O
    residual_tile_i = X
    output          = X_mid = X + O
    啟用 RMS stat

ppu_mode_i = 2'b01：FC1 output
    main_tile_i     = GELU + Requant 後的 FC1 output
    residual_tile_i = don't care
    output          = main_tile_i
    不做 residual add
    不啟用 RMS stat

ppu_mode_i = 2'b10：FC2 output
    main_tile_i     = MLP_out
    residual_tile_i = X_mid
    output          = X_out = X_mid + MLP_out
    啟用 RMS stat
```

### 主要 I/O

```verilog
input  logic                 clk;
input  logic                 rst_n;

input  logic [1:0]           ppu_mode_i;

input  logic                 tile_valid_i;
output logic                 tile_ready_o;

input  logic [16*16*8-1:0]   main_tile_i;
input  logic [16*16*8-1:0]   residual_tile_i;

input  logic [7:0]           base_token_idx_i;
input  logic [4:0]           channel_tile_idx_i;
input  logic [15:0]          token_valid_mask_i;

output logic                 data_tile_valid_o;
input  logic                 data_tile_ready_i;
output logic [16*16*8-1:0]   data_tile_o;

output logic                 stat_valid_o;
input  logic                 stat_ready_i;
output logic [7:0]           stat_token_idx_o;
output logic [31:0]          sum_sq_o;
```

---

## 5. 符合目前 project dataflow 的原因

目前 systolic array 每次輸出：

```text
16 tokens × 16 output channels
```

而 PPU tail 也是吃：

```text
16×16 INT8 tile
```

所以可以直接接在 systolic array + requant 後面。

在 FC2 dataflow 中，PPU 階段會吃：

```text
PPU_out[16][16]
res[16][16]
```

然後寫回：

```text
BRAM_ACT_OUT[m tile][n tile]
```

這正好對應：

```text
main_tile_i     = PPU_out / Requant output
residual_tile_i = res
data_tile_o     = X_mid 或 X_out
```

---

## 6. 驗證方式

目前使用 SystemVerilog testbench 驗證三個 module。

測試檔：

```text
tb_Residual_Add_Unit.sv
tb_RMS_Stat_Accumulator.sv
tb_PPU_Residual_RMS_Tail.sv
```

golden data 在 testbench 裡直接用相同數學公式產生，不需要額外 Python 檔。

### Residual_Add_Unit 驗證

檢查公式：

```text
q_out = clamp(q_main + q_residual - 128, 0, 255)
```

包含：

```text
正常加法
上溢 clamp 到 255
下溢 clamp 到 0
zero point = 128 的 case
```

### RMS_Stat_Accumulator 驗證

檢查：

```text
partial_sum_mem[token] 跨 24 個 channel tiles 累加
channel_tile_idx_i == 23 時輸出完整 sum_sq
token_valid_mask_i 可以處理最後 padding token
acc_en_i = 0 時不累加
stat_ready_i = 0 時會 stall
```

### PPU_Residual_RMS_Tail 驗證

檢查三種 mode：

```text
00 Attention：
    做 residual add
    啟用 RMS stat

01 FC1：
    bypass main_tile
    不做 residual add
    不啟用 RMS stat

10 FC2：
    做 residual add
    啟用 RMS stat
```

---

## 7. 執行指令

一次測三個 module：

```bash
make vcs
```

分開測：

```bash
make test_residual
make test_rms
make test_tail
```

清除模擬檔案：

```bash
make clean
```

通過時會看到類似：

```text
TEST PASSED: Residual_Add_Unit
TEST PASSED: RMS_Stat_Accumulator
TEST PASSED: PPU_Residual_RMS_Tail
```

---

## 8. 注意事項

`sum_sq_o` 是 32-bit raw statistic。

如果之後 Token Stat SRAM 只想存 8-bit，不能直接截斷 `sum_sq_o`，應該再接：

```text
inv-sqrt LUT
或
sum_sq quantization / compression unit
```

目前這三個 module 只負責：

```text
Residual Add
產生 X_mid / X_out
累加 RMSNorm 需要的 Σx²
```
