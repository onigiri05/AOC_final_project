# Attention Projection (Attn.Proj) 驗證測試環境說明

此資料夾包含 Vision Transformer (ViT) 模型中 **Attention Projection** 階段的硬體驗證環境。該模組負責接收脈動陣列 (Systolic Array) 的輸出，並執行殘差連接 (Residual Add) 與 RMSNorm 的統計量累加。

## 1. 驗證架構說明
本驗證採用 **Systolic Array + PPU (Tail Stage)** 級聯架構，對齊真實硬體資料流：
- **PPU Mode**: `2'b10` (Attn.Proj / FC2 Mode)
- **硬體管線行為**:
    1. **串流收集 (SIPO)**: 將脈動陣列輸出的 256 個串流部分和封裝為一塊 8192-bit 的 Tile [cite: 8, 14, 15]。
    2. **Requantization**: 執行算術右移與飽和截斷，轉換為 uint8 格式 (Zero-point = 128) [cite: 381, 386]。
    3. **Residual Add**: 將 Requant 後的資料與殘差輸入進行 `main + residual - 128` 運算，並進行飽和截斷 [cite: 388, 393, 395]。
    4. **RMS 統計累加**: 對最終輸出的 uint8 資料進行 `(q - 128)^2` 累加，並跨 Token 通道計算平方和，以支援後續 RMSNorm 運算 [cite: 401, 412, 414]。

## 2. 檔案需求
請確保您的模擬目錄包含以下由 `golden_gen_proj.py` 生成的測試資料：

| 檔案名稱 | 說明 |
| :--- | :--- |
| `tb_proj_act.hex` | 16x16 激活輸入矩陣 (32-bit 連續存放)  |
| `tb_proj_weight.hex` | 16x16 權重矩陣 (32-bit 連續存放) |
| `tb_proj_bias.hex` | 偏置數據 (16 個 32-bit word) |
| `tb_proj_residual.hex` | 殘差輸入 Tile (64 個 32-bit word)|
| `tb_proj_golden.hex` | 最終處理後的 Feature Map (256 筆 uint8) |
| `tb_proj_sum_sq.hex` | RMSNorm 預期平方和統計量 (16 個 64-bit word)  |

## 3. 環境設置
若需重新指定模擬頂層，請在 Vivado 的 Tcl Console 執行：
```tcl
set_property top Top_Proj_TB [get_filesets sim_1]
update_compile_order -fileset sim_1
```
若模擬尚未跑完可在 Tcl Console 執行：
```tcl
run 5000ns
```
直到出現finish模擬結束訊息。
當模擬執行完畢後，Tcl Console 將印出比對結果