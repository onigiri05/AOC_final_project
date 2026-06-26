ROOT_DIR     := $(shell pwd)
SYSTOLIC_DIR := $(ROOT_DIR)/Systolic
SOFTMAX_DIR  := $(ROOT_DIR)/Softmax
PPU_DIR  := $(ROOT_DIR)/PPU
Streaming_RMSNorm_DIR  := $(ROOT_DIR)/Streaming_RMSNorm
MODEL_DIR	:= $(ROOT_DIR)/DLA_model

.PHONY: vcs wave clean model_eval

vcs:
ifeq ($(SYSTOLIC), 1)
	$(MAKE) -C $(SYSTOLIC_DIR) vcs
else ifeq ($(SOFTMAX), 1)
	$(MAKE) -C $(SOFTMAX_DIR) vcs
else ifeq ($(PPU), 1)
	$(MAKE) -C $(PPU_DIR) vcs
else ifeq ($(RMS), 1)
	$(MAKE) -C $(Streaming_RMSNorm_DIR) vcs
endif

wave:
ifeq ($(SYSTOLIC), 1)
	$(MAKE) -C $(SYSTOLIC_DIR) wave
else ifeq ($(SOFTMAX), 1)
	$(MAKE) -C $(SOFTMAX_DIR) wave
else ifeq ($(PPU), 1)
	$(MAKE) -C $(PPU_DIR) wave
else ifeq ($(RMS), 1)
	$(MAKE) -C $(Streaming_RMSNorm_DIR) wave
endif

model_eval:
ifeq ($(FULL), 1)
	$(MAKE) -C $(MODEL_DIR) eval_12b
else
	$(MAKE) -C $(MODEL_DIR) eval_b11
endif

clean:
	$(MAKE) -C $(SYSTOLIC_DIR) clean
	$(MAKE) -C $(SOFTMAX_DIR) clean
	$(MAKE) -C $(PPU_DIR) clean
	$(MAKE) -C $(Streaming_RMSNorm_DIR) clean
	$(MAKE) -C $(MODEL_DIR) clean
