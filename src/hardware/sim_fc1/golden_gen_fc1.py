import numpy as np

def build_gelu_lut():
    """與硬體 GELU_Unit.sv 的 ROM 完全對齊的對照表"""
    # 建立 256 深度查表表單
    lut = np.zeros(256, dtype=np.uint8)
    
    # 根據提供的硬體常數進行初始化
    lut[0] = 0x00; lut[1] = 0x0F; lut[2] = 0x24; lut[3] = 0x3C
    lut[4] = 0x57; lut[5] = 0x73; lut[6] = 0x90; lut[7] = 0xAD
    lut[8] = 0xC9; lut[9] = 0xE4
    # 索引 10~127 (正數飽和最大值)
    for i in range(10, 128):
        lut[i] = 0xFF
    # 索引 128~255 (負數區間全歸零)
    for i in range(128, 256):
        lut[i] = 0x00
    return lut

def hardware_gelu_and_requant(psum_matrix, scaling_factor=2):
    """模擬硬體 GELU 查表與 Requant 截斷邏輯"""
    lut = build_gelu_lut()
    golden_out = np.zeros(256, dtype=np.uint8)
    
    # 平整化以便逐一處理
    flat_psum = psum_matrix.flatten()
    
    for i, val in enumerate(flat_psum):
        # 1. 模擬 GELU 查表映射：取 data_in[15:8]
        # 轉成 32-bit 有符號數進行位元操作
        val_32 = np.int32(val)
        gelu_in_mapped = (val_32 >> 8) & 0xFF
        gelu_lut_out = lut[gelu_in_mapped]
        
        # 擴展回 32-bit 高精度
        lane_post_gelu = np.int32(gelu_lut_out)
        
        # 2. 執行 Requant 算術右移
        data_shifted = lane_post_gelu >> scaling_factor
        
        # 3. 正負溢位偵測與飽和截斷 (以有符號 8-bit 有效範圍 [-128, 127] 為硬體切分基準)
        if data_shifted > 127:
            data_out = 0xFF # 上溢飽和最大值 255
        elif data_shifted < -128:
            data_out = 0x00 # 下溢飽和最小值 0
        else:
            # 轉換為以 128 為零點的 uint8 格式：翻轉最高位元 (Bit 7)
            # 等價於將有符號 8-bit 加上 128
            signed_8b = np.int8(data_shifted)
            data_out = np.uint8(int(signed_8b) + 128)
            
        golden_out[i] = data_out
        
    return golden_out

# 設定隨機種子以確保測資可重複驗證
np.random.seed(42)

# 1. 產生脈動陣列所需的 INT8 輸入矩陣 (16x16)
# Activation 為 uint8，經硬體對齊補零至 18' 有符號格式
act_matrix = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
# Weight 為 signed int8 格式
weight_matrix = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)

# 2. 模擬 Systolic Array 內部理想的 GEMM 矩陣相乘部分和 (INT32)
# 注意：Systolic.v 程式碼中包含了 bias_load 與累加部分，但在 FC1 模式下核心為計算矩陣
# 此處簡化為標準乘加，可依需求調整 bias
psum_matrix = np.matmul(act_matrix.astype(np.int32), weight_matrix.astype(np.int32))

# 3. 計算 PPU 後處理後的最終預期黃金數據 (scaling_factor=2)
scaling_factor = 2
golden_output = hardware_gelu_and_requant(psum_matrix, scaling_factor)

# 4. 將矩陣轉為符合真實硬體 BRAM 排布的 32-bit 串流 Hex 格式
# 真實矩陣依據 Systolic.v 要求需存放為連續 32-bit word 的形式，總共 64 筆地址
def save_matrix_to_32b_hex(matrix, filename):
    with open(filename, 'w') as f:
        # 將矩陣資料平整化為 byte 串流
        flat_bytes = matrix.tobytes()
        # 每 4 個 bytes (32-bit) 組合成一組 Hex 字串寫出
        for i in range(0, len(flat_bytes), 4):
            word_bytes = flat_bytes[i:i+4]
            # 依照 Little-Endian 或 大端排布，此處採用標準硬體拼接方向
            word_val = int.from_bytes(word_bytes, byteorder='little')
            f.write(f"{word_val:08X}\n")

# 儲存 Act 與 Weight 至 BRAM 測資檔
save_matrix_to_32b_hex(act_matrix, "tb_act.hex")
save_matrix_to_32b_hex(weight_matrix, "tb_weight.hex")

# 產生 Dummy Bias 全零測資 (共 16 筆 word)
with open("tb_bias.hex", "w") as f:
    for _ in range(16):
        f.write("00000000\n")

# 儲存最終黃金比對測資 (逐行 uint8 格式)
with open("tb_golden.hex", "w") as f:
    for val in golden_output:
        f.write(f"{val:02X}\n")

print(">> [Python] Testbench hex files successfully generated!")
print("   - tb_act.hex (Systolic Act BRAM Input)")
print("   - tb_weight.hex (Systolic Weight BRAM Input)")
print("   - tb_bias.hex (Dummy Bias Input)")
print("   - tb_golden.hex (PPU Expected Output Verification)")