// ============================================================
// 檔名  : ASIC.svh
// 功能  : 全域硬體參數與巨集定義 (Global Definitions)
// 專案  : Vision Transformer (ViT-Small/16) Accelerator
// ============================================================

`ifndef ASIC_SVH
`define ASIC_SVH

// ─────────────────────────────────────────────────────────────────
// 資料位元寬度定義 (Data Width Definitions)
// ─────────────────────────────────────────────────────────────────

// DATA_BITS: 脈動陣列 (Systolic Array) 累積輸出之部分和 (Partial Sum / opsum) 精度。
// 在 INT8 的 MAC 運算中，為了防止溢位，累加器通常設定為 32-bit。
`define DATA_BITS 32

// ACT_BITS: 特徵圖 (Activation / Feature Map) 之 INT8 位元寬度。
// PPU 的輸出與 Global Buffer 的儲存皆使用此精度。
`define ACT_BITS 8

// WEIGHT_BITS: 模型權重 (Weights) 之 INT8 位元寬度。
`define WEIGHT_BITS 8

// ─────────────────────────────────────────────────────────────────
// 硬體架構規模定義 (Architecture Scale Definitions)
// ─────────────────────────────────────────────────────────────────

// SA_SIZE: Systolic Array 的實體規模 (16x16)
`define SA_SIZE 16

// ─────────────────────────────────────────────────────────────────
// 量化零點定義 (Quantization Zero Point)
// ─────────────────────────────────────────────────────────────────

// ZERO_POINT: uint8 格式下的硬體偏置零點 (128)
`define ZERO_POINT 8'd128

`endif // ASIC_SVH