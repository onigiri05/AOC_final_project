#################################################################
# Project Settings
#################################################################
ROOT_DIR     := $(shell pwd)
VC           := vcs
DEBUG_VIEW   := nWave
TB_DIR       := tb
SRC_DIR      := src
RSLT_DIR     := waveform
BLD_DIR      := build
INC_DIRS     := $(SRC_DIR) $(TB_DIR)
MACROS       :=
TB_FILE := $(ROOT_DIR)/$(TB_DIR)/PPU_TB.sv
#################################################################
# Environment / Logic Mapping
#################################################################

# Waveform (DUMP)
ifeq ($(DUMP),1)
    MACROS += FSDB
else ifeq ($(DUMP),2)
    MACROS += FSDB_ALL
else
	NO_MEM_CBK := -memcbk
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
