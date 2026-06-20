# Attention Head Score V (S x V) 驗證測試環境說明

此資料夾包含 Vision Transformer (ViT) 模型中 **Attention Head Score V** 階段的硬體驗證環境。該模組負責將經由 Softmax 歸一化後的注意力權重矩陣 ($S$) 與數值矩陣 ($V$) 進行矩陣乘法，並通過 PPU 進行量化、殘差連接與 RMS 統計量累加。

## 1. 驗證架構說明
本驗證採用 **Systolic Array + PPU (Tail Stage)** 級聯架構，對齊 Attention 運算後的輸出資料流：
- **PPU Mode**: `2'b00` (Attention Output Phase)
- **硬體管線行為**:
    1. **串流收集 (SIPO)**: 將脈動陣列輸出的 256 個串流部分和封裝為一塊 8192-bit 的 Tile。
    2. **Requantization**: 執行算術右移（Scale=8）與飽和截斷，轉換為 uint8 格式 (Zero-point = 128)。
    3. **Residual Add**: 將 Requant 後的 Attention Output ($O$) 與殘差輸入 ($X$) 進行 `O + X - 128` 運算，並進行飽和截斷以產生 $X_{mid}$。
    4. **RMS 統計累加**: 對最終輸出的 $X_{mid}$ 進行 `(q - 128)^2` 累加，計算平方和，以作為下一級 Layer Normalization 的歸一化基準。

## 2. 檔案需求
請確保您的模擬目錄包含以下由 `golden_gen_attn_v.py` 生成的測試資料：

| 檔案名稱 | 說明 |
| :--- | :--- |
| `tb_attn_v_act.hex` | Softmax 輸出權重矩陣 $S$ (16x16, uint8) |
| `tb_attn_v_weight.hex` | Value 權重矩陣 $V$ (16x16, signed int8) |
| `tb_attn_v_bias.hex` | 偏置數據 (16 個 32-bit word) |
| `tb_attn_v_residual.hex` | 殘差輸入 Tile $X$ (64 個 32-bit word) |
| `tb_attn_v_golden.hex` | 最終處理後的 Feature Map $X_{mid}$ (256 筆 uint8) |
| `tb_attn_v_sum_sq.hex` | RMSNorm 預期平方和統計量 (16 個 32-bit word) |

## 3. 環境設置
若需重新指定模擬頂層，請在 Vivado 的 Tcl Console 執行：
```tcl
set_property top Top_Attn_V_TB [get_filesets sim_1]
update_compile_order -fileset sim_1
```
若模擬尚未跑完可在 Tcl Console 執行：
```tcl
run 5000ns
```
直到出現finish模擬結束訊息。
當模擬執行完畢後，Tcl Console 將印出比對結果