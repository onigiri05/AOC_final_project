# test/testbench/vit/Makefile
# Build ViT-Small/16 end-to-end testbench (links against FA hardware simulation)
#
# Usage (inside Docker container):
#   make             — build tb_vit
#   make run         — Mode A: synthetic single-block test (~4.5 M FA cycles)
#   make run_real WEIGHTS=weights/  — Mode B: full 12-block inference (~54 M cycles)
#   make run_full WEIGHTS=weights/ IMAGE=cat.bin CLASS=281  — Mode C: real image
#   make clean

SHELL = bash

RESET := \033[0m
GREY  := \033[0;37m
GREEN := \033[0;32m
RED   := \033[0;31m
CYAN  := \033[0;36m

define msg_grey
	@printf "$(GREY)$(1)$(RESET)\n"
endef
define msg_green
	@printf "$(GREEN)$(1)$(RESET)\n\n"
endef
define msg_cyan
	@printf "$(CYAN)$(1)$(RESET)\n"
endef

# ─── Path layout (mirrors flash_attn testbench) ──────────────────────────────
FA_ROOT   := $(abspath ../../..)
TB_DIR    := $(abspath .)
HW_DIR    := $(FA_ROOT)/src/hardware/flash_attn/obj_dir
SRC_DIR   := $(FA_ROOT)/src
INC_DIR   := $(FA_ROOT)/include
BUILD_DIR := $(TB_DIR)/build
LOG_DIR   := $(FA_ROOT)/build/logs

CXX ?= g++
VERILATOR_ROOT := /usr/local/share/verilator

LDFLAGS := -pthread $(HW_DIR)/libVflash_attn_wrapper.a -lm -lz

C_INCLUDES := \
    -I $(VERILATOR_ROOT)/include \
    -I $(VERILATOR_ROOT)/include/vltstd \
    -I $(INC_DIR)/hal \
    -I $(INC_DIR)/runtime \
    -I $(HW_DIR) \
    -I $(SRC_DIR)/runtime/flash_attn \
    -I $(SRC_DIR)/runtime/vit

# Optional build flags (pass on command line):
#   STATS_DEBUG=1  → per-call cycle diagnostics printed to stderr
#   SCALE_TYING=1  → tie Q and K to the same quantization scale
#   RMSNORM=1      → use RMSNorm instead of LayerNorm (needs matching weights)
_stats_flag := $(if $(filter 1,$(STATS_DEBUG)),-DFA_STATS_DEBUG,)
_scale_flag := $(if $(filter 1,$(SCALE_TYING)),-DFA_SCALE_TYING,)
_rms_flag   := $(if $(filter 1,$(RMSNORM)),-DVIT_USE_RMSNORM,)

CXXFLAGS := -O2 -g -std=c++17 $(_stats_flag) $(_scale_flag) $(_rms_flag)

# ─── Source files ─────────────────────────────────────────────────────────────
FA_SRCS := \
    $(SRC_DIR)/hal/flash_attn_hal.cpp \
    $(SRC_DIR)/runtime/flash_attn/driver_flash_attn.cpp \
    $(SRC_DIR)/runtime/flash_attn/runtime_flash_attn.cpp

VIT_SRCS := \
    $(SRC_DIR)/runtime/vit/vit_weights.cpp \
    $(SRC_DIR)/runtime/vit/vit_ops.cpp \
    $(SRC_DIR)/runtime/vit/runtime_vit.cpp

ALL_SRCS := $(TB_DIR)/tb_vit.cpp $(FA_SRCS) $(VIT_SRCS)

TARGET := $(BUILD_DIR)/tb_vit

# ─── Build rules ──────────────────────────────────────────────────────────────
.PHONY: all hw_build run run_real run_full run_batch clean

# 'all' must be the first real target so bare 'make' builds tb_vit
all: $(TARGET)

hw_build:
	@if [ ! -f "$(HW_DIR)/libVflash_attn_wrapper.a" ]; then \
	    printf "$(GREY)[VIT] Building libVflash_attn_wrapper.a...$(RESET)\n"; \
	    $(MAKE) -C $(FA_ROOT)/src/hardware/flash_attn; \
	fi

$(TARGET): $(ALL_SRCS) | hw_build
	@mkdir -p $(BUILD_DIR) $(LOG_DIR)
	$(call msg_grey,[VIT] Compiling tb_vit...)
	$(CXX) $(CXXFLAGS) $(C_INCLUDES) \
	    $(ALL_SRCS) \
	    $(LDFLAGS) \
	    -o $@
	$(call msg_green,[VIT] Build done → $(TARGET))

# Mode A: synthetic test (no arguments)
run: $(TARGET)
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[VIT] Mode A — synthetic single-block test...)
	@set -o pipefail; $(TARGET) 2>&1 | tee $(LOG_DIR)/vit_synthetic.txt

# Resolve WEIGHTS / IMAGE to absolute paths.
# If the user passes a relative path (no leading /), prepend FA_ROOT so
# "make run_real WEIGHTS=weights/" works from any directory.
_abs = $(if $(filter /%,$(1)),$(1),$(FA_ROOT)/$(1))
WEIGHTS_ABS := $(call _abs,$(WEIGHTS))
IMAGE_ABS   := $(if $(IMAGE),$(call _abs,$(IMAGE)),)

# Mode B: full inference, real weights, synthetic image
run_real: $(TARGET)
	@[ -n "$(WEIGHTS)" ] || (echo "Usage: make run_real WEIGHTS=<dir>"; exit 1)
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[VIT] Mode B — full inference with weights: $(WEIGHTS_ABS))
	@set -o pipefail; $(TARGET) $(WEIGHTS_ABS) 2>&1 | tee $(LOG_DIR)/vit_real.txt

# Mode C: full inference, real weights + real image
run_full: $(TARGET)
	@[ -n "$(WEIGHTS)" ] || (echo "Usage: make run_full WEIGHTS=<dir> IMAGE=<bin> [CLASS=<n>]"; exit 1)
	@[ -n "$(IMAGE)"   ] || (echo "Usage: make run_full WEIGHTS=<dir> IMAGE=<bin> [CLASS=<n>]"; exit 1)
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[VIT] Mode C — image inference: $(IMAGE_ABS))
	@set -o pipefail; $(TARGET) $(WEIGHTS_ABS) $(IMAGE_ABS) $(CLASS) 2>&1 | tee $(LOG_DIR)/vit_image.txt

# Mode D: batch accuracy (SW by default; use BATCH_HW=1 for hardware)
# Example: make run_batch WEIGHTS=weights/ BATCH=batch_50/
#          make run_batch WEIGHTS=weights/ BATCH=batch_10/ BATCH_HW=1
BATCH_ABS := $(if $(BATCH),$(call _abs,$(BATCH)),)
_hw_flag  := $(if $(filter 1,$(BATCH_HW)),--hw,)

run_batch: $(TARGET)
	@[ -n "$(WEIGHTS)" ] || (echo "Usage: make run_batch WEIGHTS=<dir> BATCH=<dir> [BATCH_HW=1]"; exit 1)
	@[ -n "$(BATCH)"   ] || (echo "Usage: make run_batch WEIGHTS=<dir> BATCH=<dir> [BATCH_HW=1]"; exit 1)
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[VIT] Mode D — Batch accuracy: $(BATCH_ABS)  HW=$(BATCH_HW))
	@set -o pipefail; $(TARGET) $(WEIGHTS_ABS) --batch $(BATCH_ABS) $(_hw_flag) \
	    2>&1 | tee $(LOG_DIR)/vit_batch.txt

clean:
	$(call msg_grey,[VIT] Cleaning...)
	@rm -rf $(BUILD_DIR)
