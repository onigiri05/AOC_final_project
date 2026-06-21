from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from typing import Dict, Any


@dataclass(frozen=True)
class ViTModelSpec:
    """Core ViT dimensions used by the analytical profiler."""
    name: str = "vit_small_patch16_224"
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 1000
    tokens: int = 197          # 196 patches + CLS
    embed_dim: int = 384
    num_heads: int = 6
    head_dim: int = 64
    mlp_dim: int = 1536
    num_blocks: int = 12
    qkv_out_dim: int = 1152
    has_cls_token: bool = True
    source: str = "manual_default"

    @property
    def patch_tokens(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def padded_tokens_16(self) -> int:
        return ceil(self.tokens / 16) * 16

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareConfig:
    """PYNQ-Z2-oriented hardware assumptions for the theoretical model."""
    name: str = "PYNQ-Z2_16x16_systolic"
    pe_rows: int = 16
    pe_cols: int = 16
    clock_hz: float = 100e6 # Target profiling clock: 100 MHz
    dram_peak_bandwidth_Bps: float = 2.1e9 ## 先採用 DDR 理論 peak bandwidth
    dram_efficiency: float = 0.50 # 假設實際只能達到 50% peak
    bram_capacity_bytes: int = 140 * 4096  # effective 4 KB per RAMB36E1 = 560 KB
    tile_m: int = 16
    tile_n: int = 16
    tile_k: int = 16
    bram_service_bytes_per_cycle: float = 8.0 # 假設 32B/cycle，每 cycle PE array 吃 16 個 INT8 activation + 16 個 INT8 weight
    dsp_packing_factor: float = 2.0 # 假設 INT8 DSP packing 可以讓有效 MAC throughput 變成 2 倍 -> roofline 水平線變兩倍
    pingpong_enabled: bool = True # without ping-pong: compute_cycles + memory_cycles，with ping-pong: max(compute_cycles, memory_cycles)

    @property
    def base_peak_macs_per_cycle(self) -> float:
        return float(self.pe_rows * self.pe_cols) # 256 MACs/cycle

    @property
    def optimized_peak_macs_per_cycle(self) -> float:
        # DSP packing improves effective INT8 MAC throughput.
        return self.base_peak_macs_per_cycle * self.dsp_packing_factor # 256*2 MACs/cycle

    @property
    def effective_dram_bandwidth_Bps(self) -> float:
        return self.dram_peak_bandwidth_Bps * self.dram_efficiency # 2.1*0.5 = 1.05 GB/s

    @property
    def effective_dram_bytes_per_cycle(self) -> float:
        return self.effective_dram_bandwidth_Bps / self.clock_hz # bandtwidth: bytes/s -> btyes/cycle

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnergyConfig:
    """Coarse analytical energy model. Tune after Vivado/PYNQ measurements."""
    energy_per_fp32_mac_pj: float = 4.0 # 設 1 FP32 MAC = 4 pJ
    energy_per_int8_mac_pj: float = 1.0 # 設 1 INT8 MAC = 1 pJ
    energy_per_bram_byte_pj: float = 5.0 # 每 access 1 byte BRAM，估 5 pJ
    energy_per_dram_byte_pj: float = 200.0 # 每 access 1 byte DRAM，估 200 pJ
    energy_per_lut_access_pj: float = 1.0 # 每 access 1 byte BRAM，估 1 pJ
    leakage_power_w: float = 0.145 # 設 0.145 W

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
