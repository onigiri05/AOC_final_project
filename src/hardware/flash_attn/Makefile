FA_ROOT := $(abspath ../../..)

BIN0 = obj_dir/Vflash_attn_wrapper.h
LIB0 = obj_dir/libVflash_attn_wrapper.a

SV_SRC  := $(FA_ROOT)/src/hardware/flash_attn/rtl/systolic_array.sv \
            $(FA_ROOT)/src/hardware/flash_attn/rtl/flash_attn_wrapper.sv
DPI_SRC := $(FA_ROOT)/src/dpi/dpi_math.c

VFLAGS := -Wall -Wno-lint -Wno-style --cc --trace-fst \
          --top-module flash_attn_wrapper -sv \
          --clk ACLK --Mdir obj_dir

VC := verilator

VERILATOR_ROOT := /usr/local/share/verilator
VERILATOR_SOURCES := \
    $(VERILATOR_ROOT)/include/verilated.cpp \
    $(VERILATOR_ROOT)/include/verilated_fst_c.cpp \
    $(VERILATOR_ROOT)/include/verilated_threads.cpp

INC := -I. -I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd

.PHONY: all clean

all: $(LIB0)

$(BIN0): $(SV_SRC)
	mkdir -p obj_dir
	$(VC) $(VFLAGS) $^

$(LIB0): $(BIN0) $(DPI_SRC)
	cp $(DPI_SRC) obj_dir/
	cd obj_dir && \
	  g++ -c -fPIC $(INC) $(VERILATOR_SOURCES) && \
	  g++ -c -fPIC $(INC) *.cpp && \
	  gcc -c -fPIC $(INC) dpi_math.c && \
	  ar rcs $(notdir $(LIB0)) *.o && \
	  rm -f *.o

clean:
	$(RM) -rv obj_dir
