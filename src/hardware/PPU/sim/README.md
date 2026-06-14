---
title: PPU Top-Level Module

---

# PPU Top-Level Module 

## 1. 模組概述 (Module Overview)
`PPU` (Post-Processing Unit) 是 Vision Transformer (ViT-Small/16) 硬體加速晶片中的關鍵後處理單元。它緊接在脈動陣列 (Systolic Array) 之後，負責接收高精度的 32-bit 部分和 (Partial Sum / Psum)，並在硬體流水線中平行處理量化、非線性激活函數、殘差相加 (Residual Add) 以及層歸一化統計量 (RMSNorm Statistics) 的計算。

本模組整合了四大核心運算子單元：
- **GELU_Unit**: 1-cycle 延遲的非線性激活函數查表 ROM。
- **Requant_Unit**: 高效率的 Power-of-two 算術右移與飽和截斷電路，將 INT32 轉回以 128 為零點的 uint8 格式。
- **Residual_Add_Unit**: $16 \times 16$ Tile 規模的特徵圖殘差相加器。
- **RMS_Stat_Accumulator**: 跨通道 (Channel-tile) 的平方和累加器，為後續的 RMSNorm 提供硬體統計量。
```
Systolic Array 輸出 INT32 Tile (16x16)
        ↓
[Stage 1] GELU Unit (查表) & Bypass 邏輯
        ↓
[Stage 1] Requant Unit (算術右移 + 飽和截斷至 uint8)
        ↓
[Stage 2] Residual Add Unit (+ 來自 SRAM 的 Residual Tile)
        ↓
[Stage 2] RMS Stat Accumulator (計算 Σ(x-128)²)
        ↓
PPU 輸出 uint8 Tile (至 GLB) & 32-bit RMS 統計量 (至 Stat SRAM)
```
---

## 2. 硬體參數定義 (Parameters)
模組支援高度參數化配置，預設配置完全對齊 ViT-Small/16 模型規模：

| 參數名稱 | 預設值 | 資料型態 | 說明 |
| :--- | :---: | :---: | :--- |
| `TOKEN_NUM` | `197` | `int` | 輸入 Sequence 的總 Token 數 (14x14 patches + 1 class token) |
| `CHANNEL_NUM` | `384` | `int` | 特徵圖通道維度 (Embedding Dimension $D$) |
| `TOKEN_TILE` | `16` | `int` | 脈動陣列與 PPU 單次處理的 Token 區塊大小 (Row 規模) |
| `CHANNEL_TILE` | `16` | `int` | 脈動陣列與 PPU 單次處理的 Channel 區塊大小 (Column 規模) |
| `DATA_W` | `8` | `int` | 激活值與特徵圖的 INT8/uint8 位元寬度 |
| `SUM_W` | `32` | `int` | RMS 統計累加器的位元寬度 |
| `TOKEN_W` | `8` | `int` | Token 索引訊號的位元寬度 |
| `CHANNEL_TILE_W` | `5` | `int` | Channel Tile 索引的位元寬度 (384/16 = 24 區塊，需 5-bit) |
| `ZERO_POINT` | `8'd128` | `logic [7:0]` | 量化 uint8 的非零偏置起點 (Zero Point) |

---

## 3. 輸出入埠介紹 (I/O Ports)

PPU 頂層模組內部埠依照功能可劃分為 **系統控制**、**輸入資料交握**、**輸出特徵圖交握** 以及 **統計量交握** 四大類：

### A. 系統與全域控制訊號 (System & Global Control)
| 訊號名稱 | 方向 | 位元寬度 | 說明 |
| :--- | :---: | :---: | :--- |
| `clk` | Input | `1` | 全域同步時脈訊號 (100MHz) |
| `rst` | Input | `1` | 高電位非同步/同步重置訊號 |
| `ppu_mode_i` | Input | `2` | **PPU 工作模式選擇：**<br>• `2'b00`: Attention Output 階段 (啟用殘差與 RMS)<br>• `2'b01`: FFN FC1 階段 (啟用 GELU 激活函數，旁路殘差)<br>• `2'b10`: FFN FC2 階段 (旁路 GELU，啟用殘差與 RMS) |
| `scaling_factor_i` | Input | `6` | 量化右移位數 $n$ (代表除以 $2^n$)，由 Control Path 暫存器配置 |

### B. 輸入端 Tile 資料與交握介面 (Input Tile & Handshake)
| 訊號名稱 | 方向 | 位元寬度 | 說明 |
| :--- | :---: | :---: | :--- |
| `tile_valid_i` | Input | `1` | 輸入資料有效訊號 (來自 Systolic Array 或輸入 Buffer) |
| `tile_ready_o` | Output | `1` | PPU 準備好接收新資料。當管線阻塞或 Pending Queue 滿時拉低 |
| `psum_tile_i` | Input | `TOKEN_TILE * CHANNEL_TILE * 32` <br> (8192 bits) | 來自脈動陣列的 $16\times16$ INT32 部分和資料矩陣 |
| `residual_tile_i` | Input | `TOKEN_TILE * CHANNEL_TILE * DATA_W` <br> (2048 bits) | 來自 Shortcut Buffer 的 uint8 殘差資料矩陣 |
| `base_token_idx_i` | Input | `TOKEN_W` | 當前 Tile 內第 0 行對應的真實 Token 絕對索引值 |
| `channel_tile_idx_i` | Input | `CHANNEL_TILE_W` | 當前處理的 Channel 區塊索引 (範圍 0~23) |
| `token_valid_mask_i` | Input | `TOKEN_TILE` | Token 有效遮罩。處理邊界 Token (如 192~196) 時遮蔽無效 Row |

### C. 輸出端特徵圖與交握介面 (Output Tile & Handshake)
| 訊號名稱 | 方向 | 位元寬度 | 說明 |
| :--- | :---: | :---: | :--- |
| `data_tile_valid_o` | Output | `1` | PPU 輸出特徵圖有效訊號，送往 Global Buffer (GLB) |
| `data_tile_ready_i` | Input | `1` | 後級 Buffer 準備接收訊號 (Backpressure 反向壓力來源) |
| `data_tile_o` | Output | `TOKEN_TILE * CHANNEL_TILE * DATA_W` <br> (2048 bits) | 處理完成的 $16\times16$ uint8 (ZP=128) 輸出特徵圖矩陣 |

### D. 統計量輸出與交握介面 (Statistic Output & Handshake)
| 訊號名稱 | 方向 | 位元寬度 | 說明 |
| :--- | :---: | :---: | :--- |
| `stat_valid_o` | Output | `1` | RMSNorm 統計量有效訊號 (當看滿 24 個 Channel Tile 後拉高) |
| `stat_ready_i` | Input | `1` | 後級 Token Stat SRAM 或 Inv-Sqrt 單元準備好接收訊號 |
| `stat_token_idx_o` | Output | `TOKEN_W` | 當前輸出的統計量所屬的 Token 絕對索引值 |
| `sum_sq_o` | Output | `SUM_W` | 該 Token 完整的 384 通道平方和結果 $\sum (x - 128)^2$ |

---

## 4. 內部管線結構與資料流 (Pipeline & Dataflow)

PPU 頂層模組將內部運算切分為兩大硬體流水線階段 (Pipeline Stages) 以優化時序：
### Stage 1: Requantization & Non-linear Activation

- 輸入的 `psum_tile_i` 會同時進入 `GELU_Unit` 的輸入埠與一組內部的**旁路暫存器陣列** (`stg1_psum_bypass`)。
- `GELU_Unit` 內含多組 256x8-bit ROM，會花費 **1 個時脈週期** 完成查表。
- 在下一個時脈週期，透過多路選擇器 (Mux) 判斷：
  - 若 `ppu_mode_i == 2'b01` (FC1 階段)，資料流選擇 `lane_gelu_out`。
  - 其他工作模式，則旁路選擇 `stg1_psum_bypass`。
- 資料隨後進入純組合邏輯的 `Requant_Unit` 進行算術右移 (`>>> scaling_factor`)、溢位偵測及飽和截斷，轉換成 **uint8**。

---

### Stage 2 & 3: Tail Processing & Statistics Accumulation

經過量化後的 16×16 uint8 資料矩陣進入 `PPU_Residual_RMS_Tail` 封裝模組。

- **Residual Add (殘差相加)**: 呼叫 `Residual_Add_Unit`，依據以下公式執行非對稱量化空間下的殘差加法：
  $q_{out} = \text{clamp}(q_{main} + q_{res} - 128, 0, 255)$ 
  *(註：若在 FC1 模式則直接旁路不處理。)*
- **RMS Accumulator (統計量累加)**: 將資料送入 `RMS_Stat_Accumulator`。由於 ViT-Small 的通道數為 384 (即 24×16)，累加器內部包含記憶體陣列，會暫存非連續週期進來的同個 Token 中途累加值。當最後一個區塊 (`channel_tile_idx_i == 23`) 運算完成時，完整的 32-bit 平方和會被推入 Pending Queue，並透過 `stat_valid_o` 逐筆同步送出。


```text
                    +----------------------------------------+
                    |           PPU Pipeline Top             |
                    +----------------------------------------+
                                        |
     [Stage 1: Requant & GELU]          v
                                  +------------+
                     Psum_i ----> | GELU Unit  | ----+
                                  +------------+     |
                                        |            v
                                        |     [ppu_mode_i == 2'b01]
                                        +---->  / Mux
                                               /  \
                                               +--+
                                                  |
                                                  v
                                          +--------------+
                                          | Requant Unit |
                                          +--------------+
                                                  |
     [Stage 2: Tail & Accumulator]                v  (stg1_main_tile)
                                  +------------------------------+
                  Residual_i ---->|    PPU_Residual_RMS_Tail     |
                                  +------------------------------+
                                    |                          |
                                    v (Handshake)              v (Handshake)
                               [data_tile_o]              [sum_sq_o]