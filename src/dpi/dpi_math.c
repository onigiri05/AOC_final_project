/*
 * dpi_math.c — DPI-C helper functions for FlashAttention RTL behavioral model.
 *
 * Provides fp32 bit-pattern ↔ double conversion and math primitives
 * used by flash_attn_wrapper.sv during Verilator simulation.
 */

#include <math.h>
#include <string.h>
#include "svdpi.h"

/* Reinterpret 32-bit bit pattern as IEEE 754 float, return as double */
double dpi_fp32_bits_to_real(unsigned int bits) {
    float f;
    memcpy(&f, &bits, sizeof(float));
    return (double)f;
}

/* Convert double to nearest IEEE 754 float; return bit pattern */
unsigned int dpi_real_to_fp32_bits(double d) {
    float f = (float)d;
    unsigned int bits;
    memcpy(&bits, &f, sizeof(unsigned int));
    return bits;
}

/* exp() wrapper */
double dpi_expf(double x) { return exp(x); }

/* sqrt() wrapper */
double dpi_sqrtf(double x) { return sqrt(x); }
