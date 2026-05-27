---
title: final project記憶體分配
tags: [AOC]

---

## 修改記錄

> 更新日期：2026-05-26
> 修改原因：FA team 確認 Systolic Array 最終尺寸為 **14×14**，Br=Bc=14（對稱切割）。
> S_tile = 14×14 = SA out_mat 大小，不需要額外 Score Buffer。
> 所有含 16（SA 相關尺寸）的欄位依此連帶更新。

| # | 修改位置 | 原值 | 新值 | 說明 |
|---|---------|------|------|------|
| 1 | Tile setting / Systolic array | 16×16 | **14×14** | FA team 決定，Br=Bc=SA_SIZE=14 |
| 2 | Tile setting / MLP token tile | 16 tokens | **14 tokens** | 與 SA 尺寸一致 |
| 3 | Tile setting / Q tile | [16,64] | **[14,64]** | Br=14 |
| 4 | Tile setting / K/V tile | [64,64] | **[14,64]** | Bc=14=Br，對稱切割，不需 Score Buffer |
| 5 | K Ping-Pong Bank 大小 | 2×[64,64]=8,192 B | **2×[14,64]=1,792 B** | Bc 縮小 |
| 6 | K Ping-Pong Bank BRAM | 2 RAMB36 | **1 RAMB36** | 1,792 B < 4 KB，1 個即可 |
| 7 | V Ping-Pong Bank 大小 | 2×[64,64]=8,192 B | **2×[14,64]=1,792 B** | 同 K |
| 8 | V Ping-Pong Bank BRAM | 2 RAMB36 | **1 RAMB36** | 同 K |
| 9 | Weight Buffer 大小 | 2×[384,16]=12,288 B | **2×[384,14]=10,752 B** | token tile 14 |
| 10 | On-Chip Global Buffer 大小 | [16,1536]=24,576 B | **[14,1536]=21,504 B** | token tile 14 |
| 11 | Q Ping-Pong Buffer 大小 | 2×[16,64]=2,048 B | **2×[14,64]=1,792 B** | Br=14 |
| 12 | Score Buffer | [16,64] INT16, 1 RAMB18 | **不需要（省 1 RAMB18）** | S_tile=14×14 = SA out_mat，直接用 SA 輸出暫存器 |
| 13 | O Accumulator 大小 | [16,64] INT32=4,096 B | **[14,64] INT32=3,584 B** | Br=14 |
| 14 | Psum Buffer 大小 | 16×16×4=1,024 B | **14×14×4=784 B** | SA 尺寸縮小 |
| 15 | Activation FIFO 大小 | 16×16×1=256 B | **14×14×1=196 B** | SA 尺寸縮小 |
| 16 | PE Registers PE 數量 | 256 PE | **196 PE** | 14×14=196 個 PE |
| 17 | Requant Output FIFO 大小 | 16×16×1=256 B | **14×14×1=196 B** | SA 尺寸縮小 |
| 18 | Residual Read FIFO 大小 | 16×16×1=256 B | **14×14×1=196 B** | SA 尺寸縮小 |
| 19 | RMS Stat Accumulator | 16 tokens | **14 tokens** | token tile 14 |
| 20 | RMSNorm Output Register | 16×16×1=256 B | **14×14×1=196 B** | SA 尺寸縮小 |
| 21 | BRAM 小計 K,V Cache Buffer | 4 RAMB36 | **2 RAMB36** | 各省 1（修改 #6, #8） |
| 22 | BRAM 小計 Score Buffer | 1 RAMB18 | **0（移除）** | 修改 #12 |
| 23 | BRAM 總計 | 61 RAMB36 / 4~5 RAMB18 | **59 RAMB36 / 3~4 RAMB18** | 累計節省 2 RAMB36、1 RAMB18 |

---

## 規格

Target model: ViT-Small/16
N = 197 tokens
D = 384
head = 6
head_dim = 64
MLP hidden = 1536

Data format:
Activation = INT8
Weight = INT8
Psum / opsum = INT32
RMSNorm gamma / inv_rms = fixed-point, assumed 16-bit
Requant scale = Power-of-Two shift

Tile setting:
Systolic array = ~~16 × 16~~ → **14 × 14**　　<!-- ✏️ 修改#1：FA team 確認最終尺寸為 14×14 -->
MLP token tile = ~~16~~ → **14 tokens**　　　　<!-- ✏️ 修改#2：與 SA 尺寸一致 -->
FlashAttention:
  Q tile = ~~[16,64]~~ → **[14,64]**　　　　　<!-- ✏️ 修改#3：Br=14 -->
  K/V tile = ~~[64,64]~~ → **[14,64]**　　　　<!-- ✏️ 修改#4：Bc=14=Br，對稱切割，S_tile=14×14=SA out_mat，無需 Score Buffer -->

BRAM estimate:
1 RAMB36 ≈ 4 KB usable data
1 RAMB18 ≈ 2 KB usable data



## 整體估計

| 對應名稱 | Level | 內部分區 / 實作名稱 | 存什麼 | 大小估算 | 實作 | BRAM 分配 | 原因 |
| -------- | ----- | ------------------ | ------ | -------- | ---- | --------- | ---- |
| **Off-Chip DRAM** | L3 | — | 完整 INT8 weights、input image、必要時 spill 的 feature map | ViT-Small 約 22M params，INT8 約 22 MB | DDR3 | 不用 BRAM | 完整模型不可能放進 PYNQ-Z2 的 630 KB BRAM；DRAM 存 static weights、input image、大型 feature maps。 |
| **Global Controller / DMA Buffer** | L2 / DMA | AXI Burst FIFO | DDR burst read/write 暫存 | 約 16 KB | BRAM | **4 RAMB36** | 避免 DRAM 小碎讀寫，讓 weight / activation 可以 burst 搬到 BRAM。 |
| **Weight Buffer** | L2 | Ping-Pong Weight Buffer | 目前要餵 PE Array 的 weight tile | ~~`2 × [384,16] INT8 = 12,288 B`~~ → **`2 × [384,14] INT8 = 10,752 B`** ✏️#9 | BRAM ping-pong | **6 RAMB36** | Weight Buffer 採 Ping-Pong Buffer，一邊餵 PE Array，一邊從 DRAM 載入下一批權重。token tile 由 16→14，總量 10,752 B，ceil(5,376/4,096)=2 RAMB36/buf，共 6 RAMB36 不變。 |
| **K, V Cache Buffer** | L2 | K Ping-Pong Bank | FlashAttention 的 current / next K tile | ~~`2 × [64,64] INT8 = 8,192 B`~~ → **`2 × [14,64] INT8 = 1,792 B`** ✏️#5 | BRAM ping-pong | ~~**2 RAMB36**~~ → **1 RAMB36** ✏️#6 | K tile 採 ping-pong；Bc 由 64→14，每份 896 B，兩份共 1,792 B < 4 KB，改為 1 RAMB36 即可。 |
| **K, V Cache Buffer** | L2 | V Ping-Pong Bank | FlashAttention 的 current / next V tile | ~~`2 × [64,64] INT8 = 8,192 B`~~ → **`2 × [14,64] INT8 = 1,792 B`** ✏️#7 | BRAM ping-pong | ~~**2 RAMB36**~~ → **1 RAMB36** ✏️#8 | 同 K Ping-Pong Bank。 |
| **Activation-Residual Buffer** | L2 | Bank A | `X` / block input，最後可覆寫成 `X_out` | `[197,384] INT8 = 75,648 B` | BRAM | **19 RAMB36** | Attention phase 需要保留 `X`，等待 PPU 做 `X + O = X_mid`；最後可覆寫成 `X_out`。 |
| **Activation-Residual Buffer** | L2 | Bank B | `X_mid` / MLP residual shortcut | `[197,384] INT8 = 75,648 B` | BRAM | **19 RAMB36** | MLP phase 需要保留 `X_mid`，等待 PPU 做 `X_mid + MLP_out = X_out`。 |
| **On-Chip Global Buffer** | L2 | Shared Intermediate / Output Region | `H_tile = GELU(FC1)`、attention output tile、patch embedding staging tile、Requant 後需要暫存的 INT8 output tile | ~~最大需求為 `[16,1536] INT8 = 24,576 B`~~ → **`[14,1536] INT8 = 21,504 B`** ✏️#10 | BRAM | **6 RAMB36** | 原本的 MLP Intermediate Region、Attention Output Region、Patch Embedding Staging Region 不會同時使用，因此合併成同一塊 shared scratch space。token tile 由 16→14，21,504 B，ceil(21,504/4,096)=6 RAMB36 不變。 |
| **FlashAttention Unit Local Buffer** | FA local | Q Ping-Pong Buffer | FlashAttention 的 current / next Q tile | ~~`2 × [16,64] INT8 = 2,048 B`~~ → **`2 × [14,64] INT8 = 1,792 B`** ✏️#11 | RAMB18 / LUTRAM | **1 RAMB18** | Q tile 也做 ping-pong；Br 由 16→14，1,792 B < 2 KB，仍用 1 RAMB18。 |
| ~~**FlashAttention Unit Local Buffer**~~ | ~~FA local~~ | ~~Score Buffer~~ | ~~`S_tile = Q_tile × K_tile^T`~~ | ~~`[16,64] INT16 ≈ 2,048 B`~~ | ~~RAMB18~~ | ~~**1 RAMB18**~~ → **不需要 ✏️#12** | **【已移除】** Bc=Br=14 → S_tile = 14×14，恰好等於 SA out_mat 的輸出大小。S 結果直接存在 SA 內部暫存器（sa_out 陣列），不需要獨立的 Score Buffer BRAM。省去 1 RAMB18。 |
| **FlashAttention Unit Local Buffer** | FA local | O Accumulator | attention output accumulator | ~~`[16,64] INT32 = 4,096 B`~~ → **`[14,64] INT32 = 3,584 B`** ✏️#13 | BRAM | **1 RAMB36** | Online Softmax 逐 K/V tile 更新 output，需要保存 partial O；Br 由 16→14，3,584 B < 4 KB，仍用 1 RAMB36。 |
| **Psum Buffer & Accumulator** | L1 / PE Array 周邊 | GEMM Psum / Opsum Buffer | Requant 前的 PE Array INT32 partial sum tile | ~~`16×16×4 B = 1,024 B`；double buffer 約 2 KB~~ → **`14×14×4 B = 784 B`；double buffer 約 1.5 KB** ✏️#14 | registers | 不用 BRAM | 這裡的 psum / opsum 指 Requant 前的 INT32 partial sum；SA=14×14 共 196 個 PE，容量縮小，仍以 registers 實作。 |
| **Activation FIFO** | L1 / PE Array 周邊 | Activation FIFO | 餵 PE Array 的 activation tile | ~~`16×16×1 B = 256 B`~~ → **`14×14×1 B = 196 B`** ✏️#15 | registers / LUTRAM | 不用 BRAM | L1 / PE Array 周邊的 activation FIFO，把 L2 讀出的 tile 對齊後送入 PE Array；SA=14×14，大小縮小但仍不需 BRAM。 |
| **PE Registers** | L1 / PE internal | Weight / Activation / Psum Registers | 每個 PE 內部暫存 weight、activation、psum | ~~256 PE~~ → **196 PE 內部 registers** ✏️#16 | FF / registers | 不用 BRAM | 14×14=196 個 PE，每 PE 有 8-bit weight register、8-bit activation register、32-bit psum register。 |
| **PPU Local Register / FIFO** | PPU local | Requant Output FIFO | Requant 後 INT8 tile | ~~`16×16×1 B = 256 B`~~ → **`14×14×1 B = 196 B`** ✏️#17 | registers / LUTRAM | 不用 BRAM | 只有 196 B，可直接 streaming 到 GELU / Residual / L2 writeback。 |
| **PPU Local Register / FIFO** | PPU local | Residual Read FIFO | shortcut tile：`X` 或 `X_mid` | ~~`16×16×1 B = 256 B`~~ → **`14×14×1 B = 196 B`** ✏️#18 | registers / LUTRAM | 不用 BRAM | shortcut 本體在 Activation-Residual Buffer，PPU 只需要讀 tile 做 add。 |
| **PPU Requant Scale / Shift Buffer** | PPU local | Power-of-Two Shift Buffer | per-channel requant shift `s` | 最大 `1536×1 B = 1,536 B` | LUTRAM / registers | 不用 BRAM | 使用 Power-of-Two quantization，Requant 只需存 shift，不需 multiplier；容量小，用 LUTRAM / registers 即可。 |
| **PPU Residual Scale Register** | PPU local | Common Shift / Common Scale Register | residual add 的共同 scale / shift | 每層幾個 bytes | registers | 不用 BRAM | Residual Add 採 scale tying，只要保存目前 layer 的共同 shift / scale 設定。 |
| **Token Stat SRAM** | L2 / RMSNorm | — | `inv_rms[t]` | `197×16-bit ≈ 394 B` | RAMB18 | **1 RAMB18** | Token Stat SRAM 存 197 個 token 的 `inv_rms[t]`。 |
| **Gamma Buffer** | L2 / RMSNorm | norm1 / norm2 gamma | RMSNorm 的 `γ[c]` | 一組 `384×16-bit = 768 B`；兩組約 1,536 B | RAMB18 / LUTRAM | **1 RAMB18** | RMSNorm 每個 channel 需要一個 gamma；同一塊 RAMB18 可放 norm1 / norm2 兩組 gamma。 |
| **RMS Stat Accumulator** | PPU / RMSNorm local | sum_sq accumulator | 暫時累加 `Σx²` | ~~16 tokens 約 `16×32-bit = 64 B`~~ → **14 tokens 約 `14×32-bit = 56 B`** ✏️#19 | registers | 不用 BRAM | `sum_sq` 算完就轉成 Inv-Sqrt LUT address，只存查表後的 `inv_rms` 到 Token Stat SRAM。 |
| **PPU GELU LUT** | PPU local | GELU 近似表 | 約 0.5～2 KB | LUTRAM / ROM | 先不算 BRAM | LUT 小，先不列入主要 BRAM budget。 |
| **FlashAttention Unit Exp LUT** | FA local | Online Softmax exp 近似 | 約 0.5～2 KB | LUTRAM / ROM | 先不算 BRAM | FlashAttention Unit 使用 LUT 近似指數運算。 |
| **Streaming RMSNorm Inv-Sqrt LUT** | RMSNorm local | `1/sqrt(x)` 近似表 | 依 index bits 決定 | LUTRAM / ROM / 小 RAM | 先不算 BRAM | 軟體模擬先統計 `sum_sq` range，再決定 LUT input range / index mapping。 |
| **Streaming RMSNorm Output Register / FIFO** | RMSNorm local | normalized INT8 tile | ~~`16×16×1 B = 256 B`~~ → **`14×14×1 B = 196 B`** ✏️#20 | registers / LUTRAM | 不用 BRAM | RMSNorm output 直接送到 Activation FIFO。 |
| **Global Controller / Layer Scheduler FIFO** | Control | — | tile index、layer state、address queue | 約 1～2 KB | registers / RAMB18 | **0～1 RAMB18** | 控制 DMA、buffer address、PE mode、PPU phase；容量小，可用 registers，若想簡化可用 1 RAMB18。 |
| **Bias Buffer Optional** | Optional | — | Linear bias，若保留 bias | 最大 `1536×4 B = 6,144 B` | BRAM / 可省略 | **0～2 RAMB36** | 若軟體把 bias fold 掉或目前只估 scale，就不列入；若保留 INT32 bias，需額外 buffer。 |

:::warning
Bank A/B 合計約 38 RAMB36，實作時可保守配置為 40 RAMB36，用於 activation read/write banking，減少 intermediate activation 寫回 DRAM。
:::

## BRAM 小計

| 類別 | RAMB36 | RAMB18 | 備註 |
| ---- | -----: | -----: | ---- |
| Global Controller / DMA Buffer | 4 | 0 | |
| Weight Buffer | 6 | 0 | ✏️#9 大小由 12,288→10,752 B，BRAM 數不變 |
| K, V Cache Buffer | ~~4~~ → **2** ✏️#6#8 | 0 | Bc=14 使每份 tile 縮小，各省 1 RAMB36 |
| Activation-Residual Buffer Bank A/B | 40 | 0 | |
| On-Chip Global Buffer Shared Intermediate / Output Region | 6 | 0 | ✏️#10 大小由 24,576→21,504 B，BRAM 數不變 |
| FlashAttention Unit Local O Accumulator | 1 | 0 | ✏️#13 大小由 4,096→3,584 B，BRAM 數不變 |
| FlashAttention Unit Local Q Ping-Pong Buffer | 0 | 1 | ✏️#11 大小由 2,048→1,792 B，BRAM 數不變 |
| ~~FlashAttention Unit Local Score Buffer~~ | ~~0~~ | ~~1~~ → **0** ✏️#12 | **【已移除】** S_tile=14×14=SA out_mat，不需額外 BRAM |
| Token Stat SRAM | 0 | 1 | |
| Gamma Buffer | 0 | 1 | |
| Global Controller / Layer Scheduler FIFO | 0 | 0～1 | |
| **總計** | ~~**61**~~ → **59 RAMB36** ✏️#21 | ~~**4～5**~~ → **3～4 RAMB18** ✏️#22#23 | 節省 2 RAMB36、1 RAMB18 |
