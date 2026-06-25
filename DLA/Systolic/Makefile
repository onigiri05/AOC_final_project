#################################################################
# Project Settings
#################################################################
ROOT_DIR     := $(shell pwd)
VC           := vcs
DEBUG_VIEW   := nWave
TB_DIR       := tb
TB_FILE := $(ROOT_DIR)/$(TB_DIR)/Systolic_tb.sv
SRC_DIR      := src
RSLT_DIR     := waveform
BLD_DIR      := build
INC_DIRS     := $(SRC_DIR) $(TB_DIR)
MACROS       :=

#################################################################
# Environment / Logic Mapping
#################################################################

# case
ifeq ($(CASE), 1)
	MACROS += CASE1
else ifeq ($(CASE), 2)
	MACROS += CASE2
else ifeq ($(CASE), 3)
	MACROS += CASE3
else ifeq ($(CASE), 4)
	MACROS += CASE4
else
	MACROS += CASE0
endif

# Waveform (DUMP)
ifeq ($(DUMP),1)
    MACROS += FSDB
else ifeq ($(DUMP),2)
    MACROS += FSDB_ALL
else
	NO_MEM_CBK := -memcbk
endif

# Show output value on terminal
ifeq ($(SHOW), 1)
	MACROS += SHOW
endif

# Flags Processing
MACROS_FLAG  := $(addprefix +define+, $(MACROS))
INC_DIR_FLAG := $(addprefix +incdir+, $(addprefix $(ROOT_DIR)/, $(INC_DIRS)))

# VCS Flags
VCSFLAGS := -sverilog -debug_access+all$(NO_MEM_CBK) -debug_region+cell -full64 \
            $(INC_DIR_FLAG) $(MACROS_FLAG) $(SYN_FLAGS)

#################################################################
# Targets
#################################################################
$(RSLT_DIR):
	mkdir $(RSLT_DIR)

$(BLD_DIR):
	mkdir $(BLD_DIR)

.PHONY: vcs wave clean
# vcs target
vcs: $(RSLT_DIR)
	@cd $(RSLT_DIR); \
	$(VC) -R $(VCSFLAGS) $(TB_FILE); \

wave:
	@cd $(RSLT_DIR); $(DEBUG_VIEW) &

clean:
	rm -rf $(RSLT_DIR) $(BLD_DIR)
