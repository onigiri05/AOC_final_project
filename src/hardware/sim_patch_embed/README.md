# Patch Embedding (Patch_Embed) 驗證測試環境說明

此資料夾包含 Vision Transformer (ViT) 模型中 **Patch Embedding** 階段的硬體驗證環境。該模組為模型首層輸入，負責接收原始圖像 Patch 與權重卷積後的輸出，並執行殘差連接與 RMSNorm 的統計量累加。

## 1. 驗證架構說明
本驗證採用 **Systolic Array + PPU (Tail Stage)** 級聯架構，對齊首層模型資料流：
- **PPU Mode**: `2'b00` (Attention Output / Patch Embedding 模式)
- **硬體管線行為**:
    1. **串流收集 (SIPO)**: 將脈動陣列輸出的 256 個串流部分和封裝為一塊 8192-bit 的 Tile。
    2. **Requantization**: 執行算術右移與飽和截斷，轉換為 uint8 格式 (Zero-point = 128)。
    3. **Residual Add**: 將 Requant 後的資料與輸入的初始特徵向量進行 `main + residual - 128` 運算，並進行飽和截斷。
    4. **RMS 統計累加**: 對最終輸出的 uint8 資料進行 `(q - 128)^2` 累加，計算平方和，支援後續模型層的 RMSNorm 歸一化需求。

## 2. 檔案需求
請確保您的模擬目錄 (`sim_1/behav/xsim/`) 包含以下由 `golden_gen_patch_embed.py` 生成的測試資料：

| 檔案名稱 | 說明 |
| :--- | :--- |
| `tb_pe_act.hex` | 16x16 圖像 Patch 激活輸入矩陣 (32-bit 連續存放) |
| `tb_pe_weight.hex` | 16x16 權重卷積矩陣 (32-bit 連續存放) |
| `tb_pe_bias.hex` | 偏置數據 (16 個 32-bit word) |
| `tb_pe_residual.hex` | 殘差/初始特徵向量輸入 Tile (64 個 32-bit word) |
| `tb_pe_golden.hex` | Patch Embedding 最終預期輸出 (256 筆 uint8) |
| `tb_pe_sum_sq.hex` | RMSNorm 預期平方和統計量 (16 個 32-bit word) |

## 3. 環境設置
若需重新指定模擬頂層，請在 Vivado 的 Tcl Console 執行：
```tcl
set_property top Top_Patch_Embed_TB [get_filesets sim_1]
update_compile_order -fileset sim_1
```
若模擬尚未跑完可在 Tcl Console 執行：
```tcl
run 5000ns
```
直到出現finish模擬結束訊息。
當模擬執行完畢後，Tcl Console 將印出比對結果