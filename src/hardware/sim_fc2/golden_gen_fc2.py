import numpy as np
from google.colab import files
import os

def save_matrix_to_32b_hex(matrix, filename):
    with open(filename, 'w') as f:
        flat_bytes = matrix.tobytes()
        for i in range(0, len(flat_bytes), 4):
            word_bytes = flat_bytes[i:i+4]
            word_val = int.from_bytes(word_bytes, byteorder='little')
            f.write(f"{word_val:08X}\n")

def save_array_to_8b_hex(array, filename):
    with open(filename, 'w') as f:
        for val in array.flatten():
            f.write(f"{val:02X}\n")

def hardware_fc2_logic(psum_matrix, residual_matrix, scaling_factor=2):
    """
    完全對齊 ASIC.svh (Requant_Unit.sv -> Residual_Add_Unit.sv) 的兩階段飽和邏輯
    """
    golden_out = np.zeros(256, dtype=np.uint8)
    flat_psum = psum_matrix.flatten()
    flat_res = residual_matrix.flatten()
    
    for i in range(256):
        val_32 = np.int32(flat_psum[i])
        
        # ====================================================
        # Stage 1: Requant_Unit.sv 模擬
        # ====================================================
        data_shifted = val_32 >> scaling_factor
        
        # 1. 正負溢位偵測與飽和
        if data_shifted > 127:
            requant_out = 255
        elif data_shifted < -128:
            requant_out = 0
        else:
            # 翻轉最高位元 (等價於 +128)
            requant_out = int(np.int8(data_shifted)) + 128
            
        # ====================================================
        # Stage 2: Residual_Add_Unit.sv 模擬
        # ====================================================
        main_q = requant_out
        residual_q = int(flat_res[i])
        
        # 2. 殘差相加公式: q_out = main_q + residual_q - 128
        sum_q = main_q + residual_q - 128
        
        # 3. uint8 範圍飽和 (Clamp to 0~255)
        if sum_q < 0:
            data_out = 0
        elif sum_q > 255:
            data_out = 255
        else:
            data_out = sum_q
            
        golden_out[i] = np.uint8(data_out)
        
    return golden_out

# ==========================================
# 參數設定與生成流程 (與上次相同)
# ==========================================
np.random.seed(42)

act_matrix = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
weight_matrix = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)
residual_matrix = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)

psum_matrix = np.matmul(act_matrix.astype(np.int32), weight_matrix.astype(np.int32))
golden_output = hardware_fc2_logic(psum_matrix, residual_matrix, scaling_factor=2)

file_list = [
    ("tb_fc2_act.hex", act_matrix, save_matrix_to_32b_hex),
    ("tb_fc2_weight.hex", weight_matrix, save_matrix_to_32b_hex),
    ("tb_fc2_residual.hex", residual_matrix, save_array_to_8b_hex),
    ("tb_fc2_golden.hex", golden_output, save_array_to_8b_hex)
]

for filename, data, save_func in file_list:
    save_func(data, filename)

with open("tb_fc2_bias.hex", "w") as f:
    for _ in range(16): f.write("00000000\n")

print("\n檔案已更新，準備下載...")
all_files = [f[0] for f in file_list] + ["tb_fc2_bias.hex"]
for f in all_files:
    if os.path.exists(f): files.download(f)