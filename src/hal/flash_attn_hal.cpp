// flash_attn_hal.cpp — FlashAttnHAL implementation
// Adapted from lab-4 dla_hal.cpp; replaces Eyeriss DLA with FlashAttn accelerator.

#include "flash_attn_hal.hpp"

#include <cstdio>
#include <cstring>

#ifdef USE_FST
void FlashAttnHAL::fst_init() {
    Verilated::traceEverOn(true);
    FST_FP = new VerilatedFstC();
    device_->trace(FST_FP, FA_TRACE_DEPTH);
    fprintf(stdout, "[FA-HAL] FST trace enabled\n");
}

void FlashAttnHAL::fst_final() {
    if (FST_FP) {
        delete FST_FP;
        FST_FP = nullptr;
    }
}
#endif

FlashAttnHAL::FlashAttnHAL(uint32_t baseaddr, uint32_t mmio_size)
    : info_{},
      baseaddr_(baseaddr),
      mmio_size_(mmio_size),
      device_(nullptr),
      vm_addr_h_(0) {
    vm_addr_h_ = (reinterpret_cast<uint64_t>(this) & 0xffffffff00000000ULL);
#ifdef DEBUG
    fprintf(stderr, "[FA-HAL] vm_addr_h = 0x%lx\n", (unsigned long)vm_addr_h_);
#endif
    device_ = new Vflash_attn_wrapper("TOP");
}

FlashAttnHAL::~FlashAttnHAL() {
    if (device_) {
        delete device_;
        device_ = nullptr;
    }
}

void FlashAttnHAL::init() {
#ifdef USE_FST
    fst_init();
#endif
    reset_runtime_info();
    reset();
}

void FlashAttnHAL::reset() {
    device_->ARESETn = 0;
    for (uint32_t i = 0; i < FA_RESET_CYCLE; i++) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->ARESETn = 1;
    device_->eval();
}

void FlashAttnHAL::final() {
#ifdef USE_FST
    fst_final();
#endif
}

struct runtime_info FlashAttnHAL::get_runtime_info() const { return info_; }

void FlashAttnHAL::reset_runtime_info() {
    info_.elapsed_cycle = 0;
    info_.elapsed_time  = 0;
    info_.memory_read   = 0;
    info_.memory_write  = 0;
}

/* MMIO write (AXI4 Slave Write) */
bool FlashAttnHAL::memory_set(uint32_t addr, uint32_t data) {
    if (!device_) return false;
#ifdef DEBUG
    fprintf(stderr, "[FA-HAL] memory_set(0x%08x) = 0x%08x\n", addr, data);
#endif
    if (addr < baseaddr_ || addr >= baseaddr_ + mmio_size_) return false;

    /* AW channel */
    device_->AWID_S    = 0;
    device_->AWADDR_S  = addr;
    device_->AWLEN_S   = 0;
    device_->AWSIZE_S  = AXI_SIZE_WORD;
    device_->AWBURST_S = AXI_BURST_INC;
    device_->AWVALID_S = 1;
    device_->eval();
    while (!device_->AWREADY_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->AWVALID_S = 0;

    /* W channel */
    device_->WDATA_S  = data;
    device_->WSTRB_S  = AXI_STRB_WORD;
    device_->WLAST_S  = 1;
    device_->WVALID_S = 1;
    device_->eval();
    while (!device_->WREADY_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->WVALID_S = 0;

    /* B channel */
    device_->BREADY_S = 1;
    device_->eval();
    while (!device_->BVALID_S) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->BREADY_S = 0;

    int resp = device_->BRESP_S;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    return resp == AXI_RESP_OKAY;
}

/* MMIO read (AXI4 Slave Read) */
bool FlashAttnHAL::memory_get(uint32_t addr, uint32_t& data) {
    if (!device_) return false;
    if (addr < baseaddr_ || addr >= baseaddr_ + mmio_size_) return false;

    /* AR channel */
    device_->ARID_S    = 0;
    device_->ARADDR_S  = addr;
    device_->ARLEN_S   = 0;
    device_->ARSIZE_S  = AXI_SIZE_WORD;
    device_->ARBURST_S = AXI_BURST_INC;
    device_->ARVALID_S = 1;
    do {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    } while (!device_->ARREADY_S);
    device_->ARVALID_S = 0;

    /* R channel */
    device_->RREADY_S = 1;
    do {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    } while (!device_->RVALID_S);
    device_->RREADY_S = 0;

    data = device_->RDATA_S;
    int resp = device_->RRESP_S;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    return resp == AXI_RESP_OKAY;
}

/* Block until FlashAttention accelerator asserts interrupt */
void FlashAttnHAL::wait_for_irq() {
    if (!device_) return;
#ifdef DEBUG
    fprintf(stderr, "[FA-HAL] wait_for_irq\n");
#endif
#ifdef USE_FST
#ifndef FA_FST_DIR
#define FA_FST_DIR ""
#endif
    char filename[256];
    snprintf(filename, sizeof(filename), "%sfa_%d.fst", FA_FST_DIR, fst_task_id_);
    FST_FP->open(filename);
#endif

    while (!device_->FA_interrupt) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        if (device_->ARVALID_M) handle_dma_read();
        if (device_->AWVALID_M) handle_dma_write();
    }

#ifdef USE_FST
    FST_FP->close();
    fst_task_id_++;
#endif
}

/* DMA read — accelerator requests data from host memory */
void FlashAttnHAL::handle_dma_read() {
    uint32_t* addr =
        reinterpret_cast<uint32_t*>(vm_addr_h_ | device_->ARADDR_M);
    uint32_t len = device_->ARLEN_M;

    device_->ARREADY_M = 1;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->ARREADY_M = 0;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);

#ifdef DEBUG
    fprintf(stderr, "[FA-HAL] DMA read addr=%p len=%u\n", (void*)addr, len + 1);
#endif

    device_->RID_M   = 0;
    device_->RRESP_M = AXI_RESP_OKAY;

    for (int i = 0; i <= (int)len; i++) {
        device_->RDATA_M = *(addr + i);
        info_.elapsed_cycle += MEM_ACCESS_CYCLE;
        info_.elapsed_time  += MEM_ACCESS_CYCLE * CYCLE_TIME;

        device_->RLAST_M  = (i == (int)len);
        device_->RVALID_M = 1;
        device_->eval();

        while (!device_->RREADY_M) {
            clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        }
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        device_->RVALID_M = 0;
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->eval();

    info_.memory_read += sizeof(uint32_t) * (len + 1);
}

/* DMA write — accelerator writes data to host memory */
void FlashAttnHAL::handle_dma_write() {
    uint32_t* addr =
        reinterpret_cast<uint32_t*>(vm_addr_h_ | device_->AWADDR_M);
    uint32_t len = device_->AWLEN_M;

    device_->AWREADY_M = 1;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->AWREADY_M = 0;
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);

#ifdef DEBUG
    fprintf(stderr, "[FA-HAL] DMA write addr=%p len=%u\n", (void*)addr, len + 1);
#endif

    /* W channel */
    for (uint32_t i = 0; i <= len; i++) {
        *(addr + i) = static_cast<uint32_t>(device_->WDATA_M);
        info_.elapsed_cycle += MEM_ACCESS_CYCLE;
        info_.elapsed_time  += MEM_ACCESS_CYCLE * CYCLE_TIME;

        device_->WREADY_M = 1;
        device_->eval();

        while (!device_->WVALID_M) {
            clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        }
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
        device_->WREADY_M = 0;
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    device_->eval();

    /* B channel */
    device_->BID_M    = 0;
    device_->BRESP_M  = AXI_RESP_OKAY;
    device_->BVALID_M = 1;
    device_->eval();
    while (!device_->BREADY_M) {
        clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    }
    clock_step(device_, ACLK, info_.elapsed_cycle, info_.elapsed_time);
    device_->BVALID_M = 0;
    device_->eval();

    info_.memory_write += sizeof(uint32_t) * (len + 1);
}
