// driver_flash_attn.cpp — register-level driver implementation

#include <string.h>
#include "driver_flash_attn.h"
#include "flash_attn_hal.hpp"

static FlashAttnHAL* g_hal = nullptr;

void set_fa_hal(FlashAttnHAL* hal) { g_hal = hal; }
FlashAttnHAL* get_fa_hal() { return g_hal; }

void fa_reg_write(uint32_t offset, uint32_t value) {
    g_hal->memory_set(FA_MMIO_BASE_ADDR + offset, value);
}

uint32_t fa_reg_read(uint32_t offset) {
    uint32_t val = 0;
    g_hal->memory_get(FA_MMIO_BASE_ADDR + offset, val);
    return val;
}

void fa_set_shape(uint32_t N, uint32_t d) {
    fa_reg_write(FA_SHAPE_OFFSET, (N << 16) | (d & 0xFFFFu));
}

void fa_set_tile(uint32_t Br) {
    fa_reg_write(FA_TILE_OFFSET, Br & 0xFFFFu);
}

void fa_set_q_addr(const void* Q) {
    fa_reg_write(FA_Q_ADDR_OFFSET, static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(Q) & 0xFFFFFFFFu));
}

void fa_set_k_addr(const void* K) {
    fa_reg_write(FA_K_ADDR_OFFSET, static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(K) & 0xFFFFFFFFu));
}

void fa_set_v_addr(const void* V) {
    fa_reg_write(FA_V_ADDR_OFFSET, static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(V) & 0xFFFFFFFFu));
}

void fa_set_o_addr(void* O) {
    fa_reg_write(FA_O_ADDR_OFFSET, static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(O) & 0xFFFFFFFFu));
}

void fa_set_q_scale(float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, 4);
    fa_reg_write(FA_Q_SCALE_OFFSET, bits);
}

void fa_set_k_scale(float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, 4);
    fa_reg_write(FA_K_SCALE_OFFSET, bits);
}

void fa_set_v_scale(float scale) {
    uint32_t bits;
    memcpy(&bits, &scale, 4);
    fa_reg_write(FA_V_SCALE_OFFSET, bits);
}

void fa_start() {
    fa_reg_write(FA_CONTROL_OFFSET, FA_CTRL_START);
}
