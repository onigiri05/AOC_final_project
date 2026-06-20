import numpy as np

def rtl_style_requant(data_in, scaling_factor):
    """
    完全對齊 Requant_Unit.sv 的有符號 INT32 右移與飽和截斷邏輯
    """
    # 1. 模擬算術右移 (>>>)
    data_shifted = np.right_shift(data_in.astype(np.int32), scaling_factor)
    
    # 2. 依據硬體有符號 8-bit 範圍 [-128, 127] 進行飽和截斷
    clipped = np.clip(data_shifted, -128, 127)
    
    # 3. 翻轉最高位元 (Bit 7)，等價於加上 128 偏移量轉為 uint8
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

def main():
    # 設定隨機種子以確保測資可重複驗證
    np.random.seed(2026)
    
    scaling_factor = 2

    # 1. 產生 Patch 影像特徵輸入與權重卷積核 (16x16)
    act_matrix = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)
    weight_matrix = np.random.randint(-128, 127, size=(16, 16), dtype=np.int8)
    
    # 2. 模擬 Systolic Array 矩陣相乘與加上偏置 (INT32)
    gemm_out = np.matmul(act_matrix.astype(np.int32), weight_matrix.astype(np.int32))
    bias_vector = np.random.randint(-40, 40, size=(16,), dtype=np.int32)
    for r in range(16):
        gemm_out[r, :] += bias_vector

    # 3. 產生輸入 PPU 的基礎殘差/位置編碼 Tile (uint8, zero_point=128)
    # 註：此處若在實際模型首層無殘差時可改為全 128 (即實際值為0)，這裡使用隨機 uint8 以確保測試覆蓋率
    residual_uint8_tile = np.random.randint(0, 255, size=(16, 16), dtype=np.uint8)

    # 4. 硬體管線行為計算
    # Step A: 經由 Requant 單元轉為 uint8
    main_uint8_tile = rtl_style_requant(gemm_out, scaling_factor)
    
    # Step B: 經由 2'b00 模式下的 Residual Add 單元融合
    final_data_tile = rtl_style_residual_add(main_uint8_tile, residual_uint8_tile)
    
    # Step C: 經由 RMS 統計累加器計算 16 個 Token 的最終平方和
    golden_sum_sq = rtl_style_rms_accumulation(final_data_tile)

    # =================================================================
    # 5. 輸出符合硬體 BRAM 排佈的 Hex 檔案 (安全格式化避免 Overflow)
    # =================================================================
    
    def save_matrix_to_32b_hex(matrix, filename):
        with open(filename, 'w') as f:
            flat_bytes = matrix.tobytes()
            for i in range(0, len(flat_bytes), 4):
                word_bytes = flat_bytes[i:i+4]
                word_val = int.from_bytes(word_bytes, byteorder='little')
                f.write(f"{word_val:08X}\n")

    # 儲存輸入至 BRAM 的測資檔 (檔名完全對齊 TB 的 $readmemh 宣告)
    save_matrix_to_32b_hex(act_matrix, "tb_pe_act.hex")
    save_matrix_to_32b_hex(weight_matrix, "tb_pe_weight.hex")
    save_matrix_to_32b_hex(residual_uint8_tile, "tb_pe_residual.hex")
    
    # 儲存通道 Bias (安全無符號 32-bit Hex 轉換)
    with open("tb_pe_bias.hex", "w") as f:
        for val in bias_vector:
            hex_str = format(int(val) & 0xFFFFFFFF, '08X')
            f.write(f"{hex_str}\n")

    # 儲存最終預期輸出特徵圖 (256 行 uint8 格式)
    with open("tb_pe_golden.hex", "w") as f:
        for val in final_data_tile.flatten():
            f.write(f"{val:02X}\n")

    # 儲存 RMSNorm 預期平方和統計量 (16 行 32-bit 無符號 Hex 格式)
    with open("tb_pe_sum_sq.hex", "w") as f:
        for val in golden_sum_sq:
            hex_str = format(int(val) & 0xFFFFFFFF, '08X')
            f.write(f"{hex_str}\n")

    print(">> [Python] Patch_Embed Testbench hex files successfully generated!")
    print("   - tb_pe_act.hex        -> Systolic Act BRAM Input")
    print("   - tb_pe_weight.hex     -> Systolic Weight BRAM Input")
    print("   - tb_pe_bias.hex       -> Systolic Bias BRAM Input")
    print("   - tb_pe_residual.hex   -> PPU Residual Input (uint8)")
    print("   - tb_pe_golden.hex     -> PPU Expected Output Map (uint8)")
    print("   - tb_pe_sum_sq.hex     -> RMS_Stat Expected Sum of Squares")

if __name__ == "__main__":
    main()