# FFN FC2 驗證測試環境說明

此資料夾包含 Vision Transformer (ViT) 模型中 FFN 第二階段 (FC2) 的硬體驗證環境。該模組負責接收 MLP 前級輸出並執行殘差加法 (Residual Add) 與量化輸出。

## 1. 驗證架構說明
本驗證採用與 FC1 相同的 **Systolic Array + PPU 級聯** 架構，但啟動不同的控制模式：
- **PPU Mode**: `2'b10` (FC2 Mode)
- **硬體管線行為**:
    1. **Bypass GELU**: 跳過激活函數計算。
    2. **Requantization**: 進行算術右移 (`scaling_factor`) 與飽和截斷。
    3. **Residual Add**: 將 `main_tile` 與 `residual_tile` 進行加法，並確保輸出正確對齊零點 (Zero-point = 128)。

## 2. 檔案需求
請確保您的模擬目錄 (`sim_1/behav/xsim/`) 包含以下由 `golden_gen_fc2.py` 生成的測試資料：
(sim_1根據模擬名稱不同變更)

| 檔案名稱 | 說明 |
| :--- | :--- |
| `tb_fc2_act.hex` | 16x16 激活輸入矩陣 (32-bit 連續存放) |
| `tb_fc2_weight.hex` | 16x16 權重矩陣 (32-bit 連續存放) |
| `tb_fc2_bias.hex` | Bias 偏置 (設為全零) |
| `tb_fc2_residual.hex` | 殘差輸入 (8-bit uint8 格式) |
| `tb_fc2_golden.hex` | 硬體兩階段飽和後的golden result |

## 3. 模擬操作步驟
### 環境設置
確保 `Top_FC2_TB.sv` 已加入 Vivado 的 Simulation Sources。若需重新指定頂層，請在 Tcl Console 執行：
```tcl
set_property top Top_FC2_TB [get_filesets sim_1]
update_compile_order -fileset sim_1
```
若模擬尚未跑完可在 Tcl Console 執行：
```tcl
run 5000ns
```
直到出現finish模擬結束訊息。
當模擬執行完畢後，Tcl Console 將印出比對結果