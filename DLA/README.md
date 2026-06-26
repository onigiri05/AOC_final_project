DLA README
===
>[!Note] Brief
>DLA底下共包含 5 個資料夾及一個Makefile, 一個README
>```
>DLA/.
>├── DLA_model/
>├── PPU/
>├── Softmax/
>├── Streaming_RMSNorm_Unit/
>├── Systolic/
>├── Makefile
>└── README.md
>```
>
> - Makefile
>   > 執行PPU, Softmax, Streaming_RMSNorm_Unit, Systolic 四個 Sub-module 的 Unittest (Simulation)
>   > 執行DLA_model eval, 評估硬體推論的準確度
> - PPU/
>   > abcd
> - Softmax/
>   > Attention head中執行Softmax的單元
>   > 包含RTL code和對應的測資
> - Streaming_RMSNorm_Unit
>   > Transformer block中執行RMSNorm的單元，包含RTL code和對應的測資。
> - Systolic/
>   > 16 x 16 Systolic array
>   > 包含RTL code和對應的測資
> - DLA_model/
>   > **不是RTL**
>   >
>   > 用python實作的`DLA演算法模型`, 對應的是我們預期DLA硬體要執行的演算法。

> [!warning] 環境
> - vcs
> - verdi
> - python3 with numpy, Pillow, torch, timm

[TOC]
# Makefile
> 執行PPU, Softmax, Streaming_RMSNorm_Unit, Systolic 四個 Sub-module 的 Unittest (Simulation)
> 執行DLA_model eval, 評估硬體推論的準確度
```
$ make
Usage: make [target] [OPTION FLAGs]

Run targets
    vcs                 - Run vcs simulation
    wave                - 開nWave
    model_eval          - 比較DLA_model 與 .pt的輸出結果
    clean               - 刪除所有simulation, wave, python script...產生的衍生檔案

OPTION FLAGs:
    vcs target:
        module option:
            SYSTOLIC=1          - 跑Systolic array simulation
            SOFTMAX=1           - 跑Softmax simulation
        
        case option:
            CASE=N              - 跑不同testcase
                                - Systolic: N = 0 ~ 4
                                - Softmax: N = 1 ~ 3

        dump waveform option:
            DUMP=1              - Dump 波形檔
            DUMP=2              - Dump 波形檔, "+mda"
                                - Default 不 Dump 波形檔
                                - waveform路徑: ./module/waveform/top.fsdb

        example:
            make vcs SYSTOLIC=1 CASE=4              - 跑Systolic array的第4個case, 不 Dump 波形檔
            make vcs SOFTMAX=1 CASE=2 DUMP=1        - 跑Sofamax的第2個case, Dump 波形檔
                                                        ("./Softmax/waveform/top.fsdb")

    wave target:
        module option:
            SYSTOLIC=1          - 在Systolic array波形路徑下開nWave
            SOFTMAX=1           - 在Softmax dump波形路徑下開nWave
    
    model_eval target:
        End to end inference:
            FULL=1              - 跑12個block並比較每個block的輸出結果
                                - Default只跑block 11, 並比較每層輸出結果
        Heatmap:
            HEATMAP=1           - Dump每個block 六個分類頭的平均 heatmap
            HEATMAP=2           - Dump每個block 六個分類頭的平均 heatmap + 各分類頭的Heatmap
                                - Default不產生heatmap
        example:
            make model_eval HEATMAP=2                   - 跑block 11, 產生分類頭的平均 heatmap & 各分類頭的Heatmap
            make model_eval FULL=1 HEATMAP=1            - 跑完整12個block, 產生分類頭的平均 heatmap
    
    clean target:
        沒有Flag, 刪除所有simulation, wave..產生的衍生檔案
```
----------------------------------------------------------------------------------------------
# PPU/


----------------------------------------------------------------------------------------------
# Softmax/
```
.
├── Makefile
├── README.md                               # 團隊內交接使用的README, 包含更詳細的Spec
├── hex                                     # level 1~3的測資 & exponential LUT
│   ├── README.md                           # 團隊內交接使用的README, 包含更詳細的Spec
│   ├── exp_lut_10bit_Q1_15_range12.hex     # Softmax_Unit.sv 使用的 exponential LUT
│   ├── level1                              
│   ├── level2
│   └── level3
├── src
│   └── Softmax_Unit.sv
└── tb                                      # level 1~3的tb
    ├── tb_Softmax_Unit_Level1.sv
    ├── tb_Softmax_Unit_Level2.sv
    └── tb_Softmax_Unit_Level3.sv
```
## Module Behavior
1. 將輸入的 INT32 score 做算術右移 3 bit，實作除以 `sqrt(64) = 8`。
2. 在有效 token 中尋找 row maximum。
3. 計算 `int_diff = shifted_score - row_max`。
4. 將差值映射至 1024-entry exponential LUT。
5. 累加所有 exponential value，得到 softmax denominator。
6. 正規化並輸出 signed INT8 Q0.7 attention probability。

## Testbench Expalnation
### Test cases
- Level 1 單列測試。 (CASE=1)
    - 測試範圍：
      - 單一 attention head
      - 單一 query row
      - 208 個 key positions
    - 此 testbench 會比較：
      - Shifted score
      - Row maximum
      - Exponential LUT output
      - Exponential sum
      - 最終 Q0.7 attention output
- Level 2 單一 head 完整測試。 (CASE=2)
    - 測試範圍：
      - 1 個 attention head
      - 197 個 query rows
      - 每個 row padding 至 208 個 key positions
    - 此 testbench 逐列載入 golden data，並比較：
      - Shifted score
      - Row maximum
      - Exponential LUT output
      - Exponential sum
      - 最終 attention matrix
    測試完成後會輸出各階段 mismatch 統計。
- Level 3 六個 head 完整測試。 (CASE=3)
    - 測試範圍：
      - 6 個 attention heads
      - 每個 head 197 個 query rows
      - 總列數：`6 × 197 = 1182`
      - 每列 208 個 key positions
    - 此 testbench 主要比較：
      - Row maximum
      - Exponential sum
      - 最終 Q0.7 attention output
### Pass / Fail Terminal Output
- 成功時會看到類似：
    ```text
    LEVEL 1 PASS: all 208 attention values match.
    ```

    ```text
    LEVEL 2 PASS: all 197 rows and all internal stages match.
    ```

    ```text
    LEVEL 3 PASS: all 1182 rows matched.
    ```
- 若 golden data 與 RTL 不一致，testbench 會顯示 row、key index、expected value 與 actual value。
----------------------------------------------------------------------------------------------

# Streaming_RMSNorm_Unit/
```text
.
├── Makefile                               # standalone RMSNorm VCS simulation Makefile
├── Readme.md                              # 團隊內交接使用的 README
├── src
│   ├── Streaming_RMSNorm_Unit.sv          # RMSNorm core + stream controller
│   └── Streaming_RMSNorm_RowPacker.sv     # 將 RMSNorm output pack 成 32-bit BRAM word
├── tb
│   └── tb_Streaming_RMSNorm_Unit.sv       # full-shape standalone testbench
└── hardware_export
    ├── hardware_export_manifest.json
    ├── rmsnorm_vectors
    │   └── synthetic_full_shape_197x384
    │       ├── vector_config.json
    │       ├── x_input.mem                # signed INT8 input activation
    │       ├── inv_rms.mem                # per-token inv_rms
    │       ├── gamma.mem                  # per-channel gamma
    │       └── golden.mem                 # signed INT8 software golden output
    ├── gamma_buffer                       # ViT block/layer gamma export
    └── rmsnorm_inv_sqrt_lut               # inv-sqrt LUT export
```

### Module Behavior

`Streaming_RMSNorm_Unit` 將輸入的 activation stream 做 RMSNorm，並輸出 signed INT8 normalized activation。

目前 standalone unit test 使用 signed INT8 input / signed INT8 output。

```text
TOKEN_NUM   = 197
CHANNEL_NUM = 384
TOTAL_ELEMS = 197 × 384 = 75648

x_in        = signed INT8
inv_rms     = unsigned 16-bit fixed-point, FRAC=14
gamma       = signed 16-bit fixed-point, FRAC=14
y_out       = signed INT8
```

RMSNorm core 的運算為：

```text
y[t,c] = clamp_int8((x[t,c] × inv_rms[t] × gamma[c]) >>> (2 × FRAC + OUT_SHIFT))
```

內部 pipeline：

```text
Stage 1: x × inv_rms
Stage 2: stage1 × gamma
Stage 3: shift + clamp to signed INT8
```

只有 `x_valid && x_ready` 成立時會接收一筆 input；只有 `y_valid && y_ready` 成立時會輸出一筆有效 output。

`Streaming_RMSNorm_RowPacker` 會將 RMSNorm output pack 成 32-bit BRAM word：

```text
4 筆 signed INT8 y_out -> 1 筆 act_wr_data_o[31:0]
```

RowPacker 有 `SIGNED_TO_ZP128` 參數：

```text
SIGNED_TO_ZP128 = 1: signed INT8 轉成 uint8 zero-point 128 後 pack
SIGNED_TO_ZP128 = 0: 保留 signed INT8 bit pattern 直接 pack
```

### Testbench Explanation

`tb_Streaming_RMSNorm_Unit.sv` 是 full-shape standalone testbench，會讀入完整 197 tokens × 384 channels 的測資。

Testbench 預設讀取：

```text
../hardware_export/rmsnorm_vectors/synthetic_full_shape_197x384
```

需要的檔案：

```text
x_input.mem
inv_rms.mem
gamma.mem
golden.mem
```

測試流程：

```text
讀入 x_input / inv_rms / gamma / golden
啟動 Streaming_RMSNorm_Unit
以 token-major 順序送入 75648 筆 signed INT8 activation
逐筆比對 y_out 與 golden.mem
檢查 y_last 是否只在最後一筆 output assert
```

執行方式：

```bash
make vcs RMS=1
```

若需要 waveform：

```bash
make vcs RMS=1 DUMP=1
```

### Test cases

Full-shape RMSNorm test.

測試範圍：

```text
197 tokens
384 channels per token
75648 output elements
```

此 testbench 會比較：

```text
每一筆 signed INT8 RMSNorm output
y_last 位置
valid / ready handshake 下 output 數量
```

### Pass / Fail Terminal Output

成功時會看到類似：

```text
TOKEN_NUM   = 197
CHANNEL_NUM = 384
TOTAL_ELEMS = 75648
========================================
[INFO] All .mem files loaded successfully.
[INFO] x_mem[0]      = 1
[INFO] inv_rms[0]    = 0x34e5
[INFO] gamma[0]      = 13235 / 0x33b3
[INFO] golden_mem[0] = 0
[INFO] Checked 0 / 75648 outputs
[INFO] Checked 4096 / 75648 outputs
...
[INFO] Checked 73728 / 75648 outputs
[INFO] Input driver finished. Sent 75648 elements.
[INFO] Output checker finished. Checked 75648 elements.
========================================
Streaming RMSNorm Unit TEST PASSED
Checked elements = 75648
========================================
```
若 golden data 與 RTL 不一致，testbench 會顯示 token、channel、index、expected value、actual value、x input、inv_rms 與 gamma。

----------------------------------------------------------------------------------------------
# Systolic/
```
.
├── Makefile
├── README.md                   # 團隊內交接使用的README, 包含更詳細的Spec
├── hex                         # case0 ~ 4的側資
│   ├── case0
│   ├── case1
│   ├── case2
│   ├── case3
│   ├── case4
│   └── hex_gen.ipynb           # 用來產生case0 ~ 4 hex的python檔
├── src
│   ├── Act_fifo.v
│   ├── Opsum_acc.v
│   ├── PE_pack.v
│   └── Systolic.v
└── tb
    └── Systolic_tb.sv
```
## Module Behavior
1. 單次大循環(`enable` 1次)可執行`act[16, k]` by `weight[k, 16]`
2. 其中k為任意16的倍數
3. 對應到的`[16,16]`by `[16, 16]`GEMM次數為 k/16 次
4. ex: `act[16, 48]` by `weight[48, 16]`
    總共會執行3次 `[16,16]` by `[16, 16]` GEMM
    1. 第1次算 `act[1~16, 1~16]` by `weight[1~16, 1~16]`
    2. 第2次算 `act[1~16, 17~32]` by `weight[17~32, 1~16]`
    3. 第2次算 `act[1~16, 33~48]` by `weight[33~48, 1~16]`
    4. 三次累加之後就得到[16 by 16] opsum, 並輸出到PPU
5. ex2: 以Vit的QKV Projection 為例 `act[197, 384]` by `weight[384, 1152]`
    1. 需要`enable` ceil(197/16) * ceil(1152/16)次
    2. 每次`enable` 都執行`act [16, 384]` by `weight [384, 16]`
        - 384/16 = 24次`[16,16]` by `[16, 16]` GEMM, 產出 `opsum[16, 16]`, 送至PPU
## Testbench Expalnation
### Test cases
- case0 (CASE=0)
  - [16, 16] by [16, 16]
  - [act] [Identity] = [act]
- case1 (CASE=1)
  - [16, 16] by [16, 16]
  - [act] [weight] = [psum]
  - all random value
- case2 (CASE=2)
  - [16, 32] by [32, 16] (2 k-tile accumulate)
  - [act] [weight] = [psum]
  - all random value
- case3 (CASE=3)
  - [16, 64] by [64, 16] (4 k-tile accumulate)
  - [act] [weight] = [psum]
  - all random value
- case4 (CASE=4)
  - [16, 1536] by [1536, 16]
  - [act] [weight] = [psum]
  - 96 K-tile accumulate, max case in `ViT-small-16`
  - all random value
### Pass / Fail Terminal Output
- Pass時輸出一隻青蛙
    ```
    ⠄⠄⠄⠄⢀⣠⣶⣶⣶⣤⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⢀⣠⣤⣄⡀⠄⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⠄⢠⣾⡟⠁⠄⠈⢻⣿⡀⠄⠄⠄⠄⠄⠄⠄⣼⣿⡿⠋⠉⠻⣷⠄⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⠄⢸⣿⣷⣄⣀⣠⣿⣿⡇⠄⠄⠄⠄⠄⠄⢰⣿⣿⣇⠄⠄⢠⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⠄⢸⣿⣿⣿⣿⣿⣿⣿⣦⣤⣤⣤⣤⣤⣤⣼⣿⣿⣿⣿⣿⣿⣿⡇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⠄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⣿⣿⣿⣿⣿⡏⣍⡻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⢛⣩⡍⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿         Simulation Pass !!!     ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⣿⣿⣿⣿⣿⣇⢿⠻⠮⠭⠭⠭⢭⣭⣭⣭⣛⣭⣭⠶⠿⠛⣽⢱⣿⣿⣿⣿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⣿⣿⣿⣿⣿⣿⣦⢱⡀⠄⢰⣿⡇⠄⠄⠄⠄⠄⠄⠄⢀⣾⢇⣿⣿⣿⣿⡿⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠻⢿⣿⣿⣿⢛⣭⣥⣭⣤⣼⣿⡇⠤⠤⠤⣤⣤⣤⡤⢞⣥⣿⣿⣿⣿⣿⠃⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⠄⣛⣛⠃⣿⣿⣿⣿⣿⣿⣿⢇⡙⠻⢿⣶⣶⣶⣾⣿⣿⣿⠿⢟⣛⠃⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⠄⣼⣿⣿⡘⣿⣿⣿⣿⣿⣿⡏⣼⣿⣿⣶⣬⣭⣭⣭⣭⣭⣴⣾⣿⣿⡄⠄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⠄⣼⣿⣿⣿⣷⣜⣛⣛⣛⣛⣛⣀⡛⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣶⣦⣭⣙⣛⣛⣩⣭⣭⣿⣿⣿⣿⣷⡀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ```
- Fail時output出錯的index, opsum, golden
    ```
    At cnt          12, opsum got 0x0000045a, golden got 0x0000000e
    .....
    At cnt         252, opsum got 0x00000393, golden got 0x000000eb
    ⠄⣾⠟⢋⣉⣙⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡀⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣶⠏⣰⣿⣿⣿⣿⣶⣌⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⠄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⣴⠟⣋⣩⣭⣭⣿⣿⣶⣾⣿⣿⣿⣿⣿⣿⡿⠟⢛⣉⣉⣉⡙⠻⣿⣿⣿⣿⡄⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⢁⣾⣿⣿⣿⣯⣛⠛⣿⣿⣿⣿⣿⣿⣿⣯⣤⡾⣿⣿⣿⣿⣿⣷⣤⠉⠻⣿⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⠘⠋⠉⠉⣉⣉⣉⡙⠻⢿⣿⣿⣿⣿⣿⣿⠏⣴⣾⣥⣶⣶⣤⣍⣻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿
    ⣿⠘⠁⠄⠄⢸⣿⣿⣿⡷⠂⣹⣿⣿⣿⣿⣿⠘⠛⠛⠛⠻⣿⣿⣿⣿⣿⣿⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⡄⠄⣀⣀⣚⣛⣉⣥⡴⠾⠿⢻⣿⣿⣿⣿⡇⢀⡋⠄⠄⣀⠉⠛⠿⢿⠟⣸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⣿⣄⠛⠿⠟⢻⣿⣡⣴⡶⠄⣾⣿⣿⣿⣿⣿⣌⡙⠿⣶⣶⣶⣶⣶⣶⣶⣿⣿⣿⣿⣿⣿⣿            Simulation Fail !!!     ⣿⣿⣿⣿⣿⣿
    ⣿⣿⣿⣿⡿⠟⢋⣽⣿⠟⣠⣿⣿⣿⣿⣿⣿⡿⣿⣿⣶⣦⣶⣿⣿⣮⣭⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⡿⠏⣁⣀⢔⠿⠋⡏⢸⣿⣿⣿⡿⠛⠉⠉⣰⣟⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⣶⠞⣉⣁⣉⣁⣈⡀⠘⠛⠛⠉⣤⣚⣛⣙⣋⣻⣇⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⠁⣼⡟⠻⠿⠿⠿⣿⣿⣦⣤⣄⣀⣉⣉⣉⣉⡛⢻⣀⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⣦⡈⠻⢷⣤⣙⠒⠶⢤⣭⣭⣭⣭⠍⢉⣩⣾⡿⠈⣿⣿⣿⣦⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⣿⣿⣿⣦⣤⣈⡛⠻⠿⠶⠶⠶⠶⠶⠚⣛⣉⣠⣴⣾⣿⣿⠿⢛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ⢿⠟⣽⢿⣿⣿⣿⠻⠶⡶⢶⠲⣶⣿⣿⣿⣿⣿⣿⡟⢿⣷⣶⣿⣿⣿⣿⣿⠟⠋⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
    ```

----------------------------------------------------------------------------------------------
# DLA_model/
```
├── DLA_model.py                    # DLA的演算法模型, 與DLA RTL各層的推論結果完全一致
├── Image                           # Inference時使用的圖片(.jpg)
│   ├── n01440764_10040.jpg
│   ├── n01440764_10194.jpg
│   └── n01582220_10061.jpg
├── Makefile
├── compare_12blocks.py             # end to end inference and eval
└── compare_block11.py              # Block11 inference and eval
```
## File expalnation
1. DLA_model.py
    - **不是RTL**, 是用python實作的`DLA演算法模型`, 對應的是我們預期DLA硬體要執行的演算法。
    - 這邊主要用途是比對`DLA演算法`與`Pytorch model (.pt)`執行推論的結果差異,藉此評估硬體推論的準確度。
    - 在FPGA驗證時, 相同的演算法模型也用來產生DLA的golden, i.e. DLA硬體實際推論的結果會與這邊的DLA演算法模型完全一致
2. compare_12blocks.py
    - 跑end to end的inference, 並比較.pt模型與DLA每個block的output activation
    - 比較結果會輸出到`./DLA_model/results`
        包含
        - block11.md: 每層的比較數據, cosine, mse, rmse等評估標準
        - block11.json: 更詳細的數據, 包含scale等
        - heatmaps: optional, 用熱圖表現attention head主要關注的區塊
1. compare_block11.py
    - 跑Block11的inference, 並比較.pt模型與DLA每個layer的output activation
    - 比較結果會輸出到`./DLA_model/results`
        包含
        - 12blocks.md: 每層的比較數據, cosine, mse, rmse等評估標準
        - 12blocks.json: 更詳細的數據, 包含scale等
        - heatmaps: optional, 用熱圖表現attention head主要關注的區塊
## Dependency
1. 會用到pt權重, 請將`rms_qat_best.pt`放在`AOC_final_project/PT_DIR`
   ```
    AOC_final_project/
    ├── DLA
    ├── FPGA
    ├── PT_DIR
    │   ├── README.md
    │   └── rms_qat_best.pt
    ├── README.md
    ├── model_verification
    └── profiling
   ```
2. 需要的Python套件
    ```
    numpy
    Pillow
    torch
    timm
    ```
