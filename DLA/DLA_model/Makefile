# ==========================================
# Python 評估腳本設定 (Variables)
# ==========================================
PYTHON := python3
SCRIPT_BLK11  := compare_block11.py
SCRIPT_12BLKS := compare_12blocks.py

# 預設輸出路徑
OUT_DIR := results
FLAG	:=

ifeq ($(HEATMAP), 2)
	FLAG += --save-head-heatmaps
else ifeq ($(HEATMAP), 1)
	FLAG +=
else
	FLAG += --no-heatmaps
endif

# 定義假目標 (結合了原本的硬體模擬與新的軟體評估)
.PHONY: eval_b11 eval_12b clean

# ==========================================
# 指令目標 (Targets)
# ==========================================
eval_b11:
	$(PYTHON) $(SCRIPT_BLK11) --out $(OUT_DIR) $(FLAG)

eval_12b:
	$(PYTHON) $(SCRIPT_12BLKS) --out $(OUT_DIR) $(FLAG)

# ------------------------------------------
# 4. 清除輸出
# ------------------------------------------
clean:
	rm -rf $(OUT_DIR)
	rm -rf __pycache__