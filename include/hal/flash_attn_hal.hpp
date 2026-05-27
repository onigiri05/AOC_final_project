// flash_attn_hal.hpp — FlashAttention Verilator simulation HAL.
//
// Wraps Vflash_attn_wrapper, provides MMIO read/write, DMA handling, and
// interrupt-driven completion. Mirrors the DlaHAL pattern from lab-4.

#ifndef FLASH_ATTN_HAL_HPP
#define FLASH_ATTN_HAL_HPP

#include "Vflash_attn_wrapper.h"
#include "hal.hpp"

// Advance the DUT by one clock cycle and update counters
#ifdef USE_FST
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
    do {                                                     \
        FST_FP->dump(elapsed_time);                          \
        (dut)->signal = 0;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME / 2;                    \
        FST_FP->dump(elapsed_time);                          \
        (dut)->signal = 1;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME / 2;                    \
        (elapsed_cycle)++;                                   \
    } while (0)
#else
#define clock_step(dut, signal, elapsed_cycle, elapsed_time) \
    do {                                                     \
        (dut)->signal = 0;                                   \
        (dut)->eval();                                       \
        (dut)->signal = 1;                                   \
        (dut)->eval();                                       \
        (elapsed_time) += CYCLE_TIME;                        \
        (elapsed_cycle)++;                                   \
    } while (0)
#endif

// AXI protocol constants
#define AXI_SIZE_BYTE  0b000
#define AXI_SIZE_HWORD 0b001
#define AXI_SIZE_WORD  0b010
#define AXI_BURST_INC  0x1
#define AXI_STRB_WORD  0b1111
#define AXI_RESP_OKAY  0x0
#define AXI_RESP_SLVERR 0x2

#ifdef USE_FST
#include <verilated_fst_c.h>
#endif

constexpr uint32_t FA_MAX_CYCLE   = 10000000;  // max simulation cycles
constexpr uint32_t FA_RESET_CYCLE = 10;        // reset hold cycles
#define FA_FST_FILE_NAME "flash_attn.fst"
constexpr int FA_TRACE_DEPTH = 3;

// FlashAttention Hardware Abstraction Layer: lifecycle, MMIO, IRQ wait
class FlashAttnHAL : public HALBase {
   private:
    struct runtime_info info_;
    uint32_t baseaddr_;
    uint32_t mmio_size_;
    Vflash_attn_wrapper* device_;
    uint64_t vm_addr_h_;  // upper 32 bits of this object's address (for DMA)

    void handle_dma_read();
    void handle_dma_write();

#ifdef USE_FST
    VerilatedFstC* FST_FP = nullptr;
    int fst_task_id_ = 0;

    void fst_init();
    void fst_final();
#endif

   public:
    FlashAttnHAL(uint32_t baseaddr, uint32_t mmio_size);
    ~FlashAttnHAL() override;

    /* HALBase lifecycle */
    void init() override;
    void reset() override;
    void final() override;

    /* HALBase performance tracking */
    struct runtime_info get_runtime_info() const override;
    void reset_runtime_info() override;

    /* MMIO / IRQ */
    bool memory_set(uint32_t addr, uint32_t data);
    bool memory_get(uint32_t addr, uint32_t& data);
    void wait_for_irq();
};

#endif  // FLASH_ATTN_HAL_HPP
