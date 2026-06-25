ROOT_DIR     := $(shell pwd)
SYSTOLIC_DIR := $(ROOT_DIR)/Systolic
SOFTMAX_DIR  := $(ROOT_DIR)/Softmax

.PHONY: vcs wave clean

vcs:
ifeq ($(SYSTOLIC), 1)
	$(MAKE) -C $(SYSTOLIC_DIR) vcs
else ifeq ($(SOFTMAX), 1)
	$(MAKE) -C $(SOFTMAX_DIR) vcs
endif

wave:
ifeq ($(SYSTOLIC), 1)
	$(MAKE) -C $(SYSTOLIC_DIR) wave
else ifeq ($(SOFTMAX), 1)
	$(MAKE) -C $(SOFTMAX_DIR) wave
endif

clean:
	$(MAKE) -C $(SYSTOLIC_DIR) clean
	$(MAKE) -C $(SOFTMAX_DIR) clean