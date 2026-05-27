FA_ROOT := $(abspath .)

RESET  := \033[0m
GREY   := \033[0;37m
GREEN  := \033[0;32m
RED    := \033[0;31m
CYAN   := \033[0;36m
WHITE  := \033[1;37m
YELLOW := \033[0;33m

define msg_grey
	@printf "$(GREY)$(1)$(RESET)\n"
endef
define msg_green
	@printf "$(GREEN)$(1)$(RESET)\n\n"
endef
define msg_red
	@printf "$(RED)$(1)$(RESET)\n"
endef
define msg_cyan
	@printf "$(CYAN)$(1)$(RESET)\n"
endef
define msg_yellow
	@printf "$(YELLOW)$(1)$(RESET)\n"
endef

_FA_EXTRA_DEFS := $(if $(filter 1,$(DEBUG)),-DDEBUG) $(if $(filter 1,$(TRACE)),-DUSE_FST)
BUILD_DIR := $(FA_ROOT)/build

.PHONY: all hardware fa clean clean_hw clean_tb help \
        run_fa_case0 run_fa_case1 run_fa_case2 run_fa

help:
	@echo "Usage: make [target] [OPTIONS]"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1    Print DMA HAL verbose log (MMIO R/W, DMA addr+data, IRQ)"
	@echo "  TRACE=1    Enable FST waveform tracing (output: build/fa_N.fst)"
	@echo ""
	@echo "Targets:"
	@echo "  hardware           Build Verilated library (libVflash_attn_wrapper.a)"
	@echo "  fa                 Build all testbench binaries"
	@echo "  run_fa_case0       Build + run case 0  (N=4,  d=4,  Br=4)"
	@echo "  run_fa_case1       Build + run case 1  (N=8,  d=8,  Br=4)"
	@echo "  run_fa_case2       Build + run case 2  (N=196, d=64, Br=14) [ViT-Small]"
	@echo "  run_fa             Run all three cases"
	@echo "  clean              Remove all build artifacts"
	@echo ""

# ─── Build Verilated library ──────────────────────────────────────────────────
hardware:
	$(call msg_grey,[ROOT] Building FlashAttention Verilated library...)
	$(MAKE) -C src/hardware/flash_attn

# ─── Build testbench ──────────────────────────────────────────────────────────
fa: hardware
	$(call msg_grey,[ROOT] Building FlashAttention testbench...)
	$(MAKE) -C test/testbench/flash_attn all EXTRA_DEFS='$(_FA_EXTRA_DEFS)'

all: fa

# ─── Individual case run targets ──────────────────────────────────────────────
run_fa_case0:
	$(call msg_grey,[ROOT] Building FA case0 testbench...)
	$(MAKE) -B -C test/testbench/flash_attn case0 EXTRA_DEFS='$(_FA_EXTRA_DEFS)'
	$(call msg_cyan,[ROOT] Running FA case0 (N=4, d=4, Br=4)...)
	$(MAKE) -C test/testbench/flash_attn run0

run_fa_case1:
	$(call msg_grey,[ROOT] Building FA case1 testbench...)
	$(MAKE) -B -C test/testbench/flash_attn case1 EXTRA_DEFS='$(_FA_EXTRA_DEFS)'
	$(call msg_cyan,[ROOT] Running FA case1 (N=8, d=8, Br=4)...)
	$(MAKE) -C test/testbench/flash_attn run1

run_fa_case2:
	$(call msg_grey,[ROOT] Building FA case2 testbench...)
	$(MAKE) -B -C test/testbench/flash_attn case2 EXTRA_DEFS='$(_FA_EXTRA_DEFS)'
	$(call msg_cyan,[ROOT] Running FA case2 (N=196, d=64, Br=14 — ViT-Small)...)
	$(MAKE) -C test/testbench/flash_attn run2

run_fa:
	$(call msg_grey,[ROOT] Running all FA cases...)
	@pass_count=0; total_count=3; \
	for c in 0 1 2; do \
	    $(MAKE) -B -C test/testbench/flash_attn case$$c EXTRA_DEFS='$(_FA_EXTRA_DEFS)' && \
	    $(MAKE) -C test/testbench/flash_attn run$$c && pass_count=$$((pass_count+1)) || true; \
	done; \
	printf "\n"; \
	if [ $$pass_count -eq $$total_count ]; then \
	    printf "\033[0;32m[TB/FA] ALL TESTS PASSED ($$pass_count/$$total_count)\033[0m\n\n"; \
	else \
	    printf "\033[0;31m[TB/FA] FAILED ($$pass_count/$$total_count passed)\033[0m\n\n"; \
	    exit 1; \
	fi

# ─── Clean ────────────────────────────────────────────────────────────────────
clean_hw:
	$(call msg_grey,[ROOT] Cleaning hardware...)
	$(MAKE) -C src/hardware/flash_attn clean

clean_tb:
	$(call msg_grey,[ROOT] Cleaning testbench...)
	$(MAKE) -C test/testbench/flash_attn clean

clean: clean_hw clean_tb
	@rm -rf $(BUILD_DIR)
	$(call msg_green,[ROOT] Done.)
