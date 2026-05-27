// driver_flash_attn.h — FlashAttention accelerator register map and driver API.
//
// Register map base: FA_MMIO_BASE_ADDR (0x10050000)
//
// Offset  Register        Description
// 0x00    FA_CONTROL       [0]=start, [1]=irq_clear
// 0x04    FA_SHAPE         [31:16]=N (seq len), [15:0]=d (head dim)
// 0x08    FA_TILE          [15:0]=Br (= Bc, tile size)
// 0x0C    FA_Q_ADDR        Low 32b of INT8-packed Q base address (row stride = d bytes)
// 0x10    FA_K_ADDR        Low 32b of INT8-packed K base address
// 0x14    FA_V_ADDR        Low 32b of INT8-packed V base address
// 0x18    FA_O_ADDR        Low 32b of fp32 O base address (row stride = d×4 bytes)
// 0x1C    FA_STATUS        [0]=busy, [1]=done (read-only)
// 0x20    FA_Q_SCALE       Global Q quantization scale (IEEE fp32 bits)
// 0x24    FA_K_SCALE       Global K quantization scale (IEEE fp32 bits)
// 0x28    FA_V_SCALE       Global V quantization scale (IEEE fp32 bits)

#ifndef DRIVER_FLASH_ATTN_H
#define DRIVER_FLASH_ATTN_H

#include <stdint.h>

/* ─── Base address and MMIO region size ─── */
#define FA_MMIO_BASE_ADDR 0x10050000u
#define FA_MMIO_SIZE      0x1000u

/* ─── Register offsets ─── */
#define FA_CONTROL_OFFSET  0x00u
#define FA_SHAPE_OFFSET    0x04u
#define FA_TILE_OFFSET     0x08u
#define FA_Q_ADDR_OFFSET   0x0Cu
#define FA_K_ADDR_OFFSET   0x10u
#define FA_V_ADDR_OFFSET   0x14u
#define FA_O_ADDR_OFFSET   0x18u
#define FA_STATUS_OFFSET   0x1Cu
#define FA_Q_SCALE_OFFSET  0x20u
#define FA_K_SCALE_OFFSET  0x24u
#define FA_V_SCALE_OFFSET  0x28u

/* ─── Control bit positions ─── */
#define FA_CTRL_START     (1u << 0)
#define FA_CTRL_IRQ_CLR   (1u << 1)

/* ─── Status bit positions ─── */
#define FA_STATUS_BUSY    (1u << 0)
#define FA_STATUS_DONE    (1u << 1)

#ifdef __cplusplus
#include "flash_attn_hal.hpp"
void set_fa_hal(FlashAttnHAL* hal);
FlashAttnHAL* get_fa_hal();
#endif

/* ─── Driver function prototypes ─── */

/** Write a 32-bit value to a register at the given offset. */
void fa_reg_write(uint32_t offset, uint32_t value);

/** Read a 32-bit value from a register at the given offset. */
uint32_t fa_reg_read(uint32_t offset);

/** Set N (sequence length) and d (head dimension). */
void fa_set_shape(uint32_t N, uint32_t d);

/** Set tile size Br = Bc. */
void fa_set_tile(uint32_t Br);

/** Set base address of INT8-packed Q (low 32 bits). */
void fa_set_q_addr(const void* Q);

/** Set base address of INT8-packed K (low 32 bits). */
void fa_set_k_addr(const void* K);

/** Set base address of INT8-packed V (low 32 bits). */
void fa_set_v_addr(const void* V);

/** Set base address of fp32 O (low 32 bits). */
void fa_set_o_addr(void* O);

/** Set global Q quantization scale (written as IEEE fp32 bits). */
void fa_set_q_scale(float scale);

/** Set global K quantization scale. */
void fa_set_k_scale(float scale);

/** Set global V quantization scale. */
void fa_set_v_scale(float scale);

/** Start the accelerator (write FA_CTRL_START to CONTROL register). */
void fa_start();

#endif  // DRIVER_FLASH_ATTN_H
