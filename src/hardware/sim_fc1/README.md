# FFN FC1 驗證測試環境說明

此資料夾包含 Vision Transformer (ViT) 模型中 FFN 第一階段 (FC1) 的硬體驗證環境。該模組負責接收脈動陣列 (Systolic Array) 的輸出，執行 GELU 激活函數查表，並進行 Requantization 量化。

## 1. 驗證架構說明
本驗證採用 **Systolic Array + PPU 級聯** 架構：
- **PPU Mode**: `2'b01` (FC1 Mode)
- **硬體管線行為**:
    1. **串流收集 (SIPO)**: 將脈動陣列輸出的 256 個串流 (Stream) 部分和封裝為一塊 8192-bit 的 Tile。
    2. **GELU Activation**: 啟用內建 ROM 查表進行非線性映射。
    3. **Requantization**: 執行算術右移與飽和截斷，轉換為 uint8 格式 (Zero-point = 128)。
    4. **Bypass**: 在 FC1 階段，自動旁路殘差相加與 RMS 統計單元。

## 2. 檔案需求
請確保您的模擬目錄 (`sim_1/behav/xsim/`) 包含以下由 `golden_gen_fc1.py` 生成的測試資料：
(sim_1根據模擬名稱不同變更)
| 檔案名稱 | 說明 |
| :--- | :--- |
| `tb_act.hex` | 16x16 激活輸入矩陣 (32-bit 連續存放) |
| `tb_weight.hex` | 16x16 權重矩陣 (32-bit 連續存放) |
| `tb_bias.hex` | Bias 偏置 (全零) |
| `tb_golden.hex` | 經過 GELU 與 Requant 兩階段處理後的golden result |

### 環境設置
若需重新指定模擬頂層，請在 Vivado 的 Tcl Console 執行：
```tcl
set_property top Top_FC1_TB [get_filesets sim_1]
update_compile_order -fileset sim_1
```
若模擬尚未跑完可在 Tcl Console 執行：
```tcl
run 5000ns
```
直到出現finish模擬結束訊息。
當模擬執行完畢後，Tcl Console 將印出比對結果