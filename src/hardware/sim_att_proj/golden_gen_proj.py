import numpy as np

def rtl_style_requant(data_in, scaling_factor):
    """
    完全對齊 Requant_Unit.sv 的有符號 INT32 右移與飽和截斷邏輯
    """
    # 1. 模擬算術右移 (>>>)
    data_shifted = np.right_shift(data_in.astype(np.int32), scaling_factor)

    # 2. 依據硬體有符號 8-bit 範圍 [-128, 127] 偵測溢位並進行飽和截斷
    clipped = np.clip(data_shifted, -128, 127)

    # 3. 翻轉最高位元 (Bit 7)，等價於加上 128 偏移量轉為 uint8 格式
    data_out = (clipped + 128).astype(np.uint8)
    return data_out

def rtl_style_residual_add(main_q, residual_q):
    """
    完全對齊 Residual_Add_Unit.sv 的 uint8 zero-point=128 殘差加法
    公式: q_out = clamp(main_q + residual_q - 128, 0, 255)
    """
    sum_q = main_q.astype(np.int32) + residual_q.astype(np.int32) - 128
    q_out = np.clip(sum_q, 0, 255).astype(np.uint8)
    return q_out

def rtl_style_rms_accumulation(data_tile_uint8):
    """
    完全對齊 RMS_Stat_Accumulator.sv 的平方和映射
    公式: centered_x = q - 128 -> square_x = centered_x^2
    """
    # 每個元素減去 128 (轉為 signed)
    centered_x = data_tile_uint8.astype(np.int32) - 128
    # 計算平方
    square_x = centered_x ** 2
    # 依 Row (Token) 方向加總，產生 16 個 Token 的當前 Tile 平方和貢獻
    row_sums = np.sum(square_x, axis=1).astype(np.uint64)
    return row_sums

# =====================================================================
# 主程式：生成與 Testbench 對齊的 Hex 測資檔
# =====================================================================
def main():
    # 設定隨機種子以確保測資可重複驗證
    np.random.seed(42)

    scaling_factor = 2

    # 1. 產生脈動陣列所需的輸入矩陣 (16x16)
    # Activation 為 uint8
    act_matrix = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    # Weight 為 signed int8 格式
    weight_matrix = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)

    # 2. 模擬 Systolic Array 輸出的理想矩陣相乘部分和 (INT32)
    # 這裡加入一些隨機 Bias 數值以豐富測試場景
    gemm_out = np.matmul(act_matrix.astype(np.int32), weight_matrix.astype(np.int32))
    bias_vector = np.random.randint(-50, 50, size=(16,), dtype=np.int32)
    for r in range(16):
        gemm_out[r, :] += bias_vector

    # 3. 產生輸入 PPU 的殘差資料 (uint8, zero_point=128)
    residual_uint8_tile = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)

    # 4. 硬體流水線行為計算
    # Step A: 經由 Requant 單元轉為 uint8
    main_uint8_tile = rtl_style_requant(gemm_out, scaling_factor)

    # Step B: 經由 Residual Add 單元與殘差融合
    final_data_tile = rtl_style_residual_add(main_uint8_tile, residual_uint8_tile)

    # Step C: 經由 RMS 統計累加器計算 16 個 Token 的當前 Tile 平方和
    golden_sum_sq = rtl_style_rms_accumulation(final_data_tile)

    # =================================================================
    # 5. 輸出符合硬體 BRAM 排佈的 Hex 檔案
    # =================================================================

    # 輔助函式：將 16x16 矩陣轉為連續 32-bit word 的 Hex 格式寫出 (共 64 筆地址)
    def save_matrix_to_32b_hex(matrix, filename):
        with open(filename, 'w') as f:
            flat_bytes = matrix.tobytes()
            for i in range(0, len(flat_bytes), 4):
                word_bytes = flat_bytes[i:i+4]
                word_val = int.from_bytes(word_bytes, byteorder='little')
                f.write(f"{word_val:08X}\n")

    # 儲存輸入至 BRAM 的測資檔
    save_matrix_to_32b_hex(act_matrix, "tb_proj_act.hex")
    save_matrix_to_32b_hex(weight_matrix, "tb_proj_weight.hex")
    save_matrix_to_32b_hex(residual_uint8_tile, "tb_proj_residual.hex")

    
    # 儲存通道 Bias 至 BRAM 測資檔 (16 筆 32-bit word)
    with open("tb_proj_bias.hex", "w") as f:
        for val in bias_vector:
            # 使用 format 配合遮罩轉為純粹的 32-bit 無符號 16 進位字串
            # 確保不會觸發 Python 內部的 int32 邊界檢查
            hex_str = format(int(val) & 0xFFFFFFFF, '08X')
            f.write(f"{hex_str}\n")

    # 儲存最終輸出特徵圖的黃金比對測資 (平整化後逐行 8-bit uint8 格式，共 256 行)
    with open("tb_proj_golden.hex", "w") as f:
        for val in final_data_tile.flatten():
            f.write(f"{val:02X}\n")

    
    # 儲存 RMSNorm 預期平方和比對測資 (16 個 Token 的 64-bit 數值，共 16 行)
    with open("tb_proj_sum_sq.hex", "w") as f:
        for val in golden_sum_sq:
            hex_str = format(int(val) & 0xFFFFFFFFFFFFFFFF, '016X')
            f.write(f"{hex_str}\n")

    print(">> [Python] Attention Projection verification hex files successfully generated!")
    print("   - tb_proj_act.hex       -> Systolic Act BRAM Input")
    print("   - tb_proj_weight.hex    -> Systolic Weight BRAM Input")
    print("   - tb_proj_bias.hex      -> Systolic Bias BRAM Input")
    print("   - tb_proj_residual.hex  -> PPU Residual Input (uint8)")
    print("   - tb_proj_golden.hex    -> PPU Expected Output Map (uint8)")
    print("   - tb_proj_sum_sq.hex    -> RMS_Stat Expected Sum of Squares")
    
if __name__ == "__main__":
    main()