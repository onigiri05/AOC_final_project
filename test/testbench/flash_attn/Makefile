# test/testbench/flash_attn/Makefile
# Build and run FlashAttention hardware simulation testbench (case 0, 1, 2)

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

FA_ROOT   := $(abspath ../../..)
TB_DIR    := $(abspath .)
HW_DIR    := $(FA_ROOT)/src/hardware/flash_attn/obj_dir
SRC_DIR   := $(FA_ROOT)/src
INC_DIR   := $(FA_ROOT)/include
CASES_DIR := $(FA_ROOT)/test/cases
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
    -I $(SRC_DIR)/runtime/flash_attn

CXXFLAGS  := -O2 -g -std=c++17
C_DEFS    := $(EXTRA_DEFS)

# Sources for every testbench binary
FA_SRCS_CPP := \
    $(SRC_DIR)/hal/flash_attn_hal.cpp \
    $(SRC_DIR)/runtime/flash_attn/driver_flash_attn.cpp \
    $(SRC_DIR)/runtime/flash_attn/runtime_flash_attn.cpp

CASES := 0 1 2

ROOT_BUILD_DIR := $(FA_ROOT)/build

# ─── Per-case binary rule ────────────────────────────────────────────────────
define BUILD_CASE
$(BUILD_DIR)/tb_fa_case$(1): \
    $(TB_DIR)/tb.cpp \
    $(FA_SRCS_CPP)
	@mkdir -p $(BUILD_DIR) $(ROOT_BUILD_DIR)
	$(call msg_grey,[FA/case$(1)] Building tb_fa_case$(1)...)
	$(CXX) $(CXXFLAGS) $(C_DEFS) -DCASE_NUM=$(1) \
	    -I $(CASES_DIR)/case$(1) \
	    $(C_INCLUDES) \
	    $(TB_DIR)/tb.cpp \
	    $(FA_SRCS_CPP) \
	    $(LDFLAGS) \
	    -o $(BUILD_DIR)/tb_fa_case$(1)
	$(call msg_green,[FA/case$(1)] Build done.)
endef

$(foreach c,$(CASES),$(eval $(call BUILD_CASE,$(c))))

TB_EXES := $(foreach c,$(CASES),$(BUILD_DIR)/tb_fa_case$(c))

# ─── Phony targets ────────────────────────────────────────────────────────────
.PHONY: all hw_build \
        $(foreach c,$(CASES),case$(c) run$(c)) \
        clean

hw_build:
	@if [ ! -f "$(HW_DIR)/libVflash_attn_wrapper.a" ]; then \
	    printf "$(GREY)[FA] Building libVflash_attn_wrapper.a...$(RESET)\n"; \
	    $(MAKE) -C $(FA_ROOT)/src/hardware/flash_attn; \
	fi

all: hw_build $(TB_EXES)
	$(call msg_green,[FA] All testbench cases built.)

case0: hw_build $(BUILD_DIR)/tb_fa_case0
case1: hw_build $(BUILD_DIR)/tb_fa_case1
case2: hw_build $(BUILD_DIR)/tb_fa_case2

run0: case0
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[FA/case0] Running... (log -> build/logs/fa_case0.txt))
	@set -o pipefail; $(BUILD_DIR)/tb_fa_case0 2>&1 | tee $(LOG_DIR)/fa_case0.txt

run1: case1
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[FA/case1] Running... (log -> build/logs/fa_case1.txt))
	@set -o pipefail; $(BUILD_DIR)/tb_fa_case1 2>&1 | tee $(LOG_DIR)/fa_case1.txt

run2: case2
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[FA/case2] Running... (log -> build/logs/fa_case2.txt))
	@set -o pipefail; $(BUILD_DIR)/tb_fa_case2 2>&1 | tee $(LOG_DIR)/fa_case2.txt

# ─── Scale Tying comparison (case2 only) ─────────────────────────────────────
# Builds two variants of case2 (independent vs tied QK scale) and compares.
# Usage: make run_scale_tying_cmp

$(BUILD_DIR)/tb_fa_case2_untied: $(TB_DIR)/tb.cpp $(FA_SRCS_CPP)
	@mkdir -p $(BUILD_DIR) $(ROOT_BUILD_DIR)
	$(call msg_grey,[FA/ScaleTying] Building case2 WITHOUT scale tying...)
	$(CXX) $(CXXFLAGS) $(C_DEFS) -DCASE_NUM=2 \
	    -I $(CASES_DIR)/case2 \
	    $(C_INCLUDES) \
	    $(TB_DIR)/tb.cpp $(FA_SRCS_CPP) $(LDFLAGS) \
	    -o $@
	$(call msg_green,[FA/ScaleTying] untied build done.)

$(BUILD_DIR)/tb_fa_case2_tied: $(TB_DIR)/tb.cpp $(FA_SRCS_CPP)
	@mkdir -p $(BUILD_DIR) $(ROOT_BUILD_DIR)
	$(call msg_grey,[FA/ScaleTying] Building case2 WITH scale tying (-DFA_SCALE_TYING)...)
	$(CXX) $(CXXFLAGS) $(C_DEFS) -DCASE_NUM=2 -DFA_SCALE_TYING \
	    -I $(CASES_DIR)/case2 \
	    $(C_INCLUDES) \
	    $(TB_DIR)/tb.cpp $(FA_SRCS_CPP) $(LDFLAGS) \
	    -o $@
	$(call msg_green,[FA/ScaleTying] tied build done.)

.PHONY: run_scale_tying_cmp
run_scale_tying_cmp: hw_build \
    $(BUILD_DIR)/tb_fa_case2_untied \
    $(BUILD_DIR)/tb_fa_case2_tied
	@mkdir -p $(LOG_DIR)
	$(call msg_cyan,[FA/ScaleTying] --- Independent Q/K scales (baseline) ---)
	@set -o pipefail; $(BUILD_DIR)/tb_fa_case2_untied 2>&1 | tee $(LOG_DIR)/fa_case2_untied.txt
	$(call msg_cyan,[FA/ScaleTying] --- Tied QK scale (max of both) ---)
	@set -o pipefail; $(BUILD_DIR)/tb_fa_case2_tied   2>&1 | tee $(LOG_DIR)/fa_case2_tied.txt
	@echo ""
	@echo "Scale Tying comparison complete."
	@echo "  untied log : build/logs/fa_case2_untied.txt"
	@echo "  tied   log : build/logs/fa_case2_tied.txt"
	@grep "Max abs diff" $(LOG_DIR)/fa_case2_untied.txt | sed 's/^/  [untied] /'
	@grep "Max abs diff" $(LOG_DIR)/fa_case2_tied.txt   | sed 's/^/  [tied  ] /'

clean:
	$(call msg_grey,[FA] Cleaning build artifacts...)
	@rm -rf $(BUILD_DIR)
