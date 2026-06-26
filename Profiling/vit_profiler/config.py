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
        # Kept for backward compatibility with older scripts.
        return ceil(self.tokens / 16) * 16

    def padded_tokens_for_tile(self, tile: int) -> int:
        """Return token count padded to the selected systolic tile size."""
        return ceil(self.tokens / tile) * tile

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HardwareConfig:
    """PYNQ-Z2 RTL-oriented hardware assumptions for the analytical model.

    This RTL-oriented version uses the current shared 8x8 systolic array.
    The optimized INT8 peak still applies dsp_packing_factor, so with the
    default factor 2.0 the optimized peak is 8*8*2 = 128 MAC/cycle.
    """
    name: str = "PYNQ-Z2_8x8_shared_systolic_RTL"
    pe_rows: int = 8
    pe_cols: int = 8
    clock_hz: float = 25e6
    dram_peak_bandwidth_Bps: float = 2.1e9
    dram_efficiency: float = 0.50
    bram_capacity_bytes: int = 140 * 4096  # effective 4 KB per RAMB36E1 = 560 KB
    tile_m: int = 8
    tile_n: int = 8
    tile_k: int = 8
    bram_service_bytes_per_cycle: float = 8.0
    dsp_packing_factor: float = 2.0
    pingpong_enabled: bool = True

    # RTL-oriented non-GEMM / page-cache parameters.
    softmax_passes_per_row: int = 4       # shift, rowmax, LUT/sum, normalize
    softmax_row_overhead_cycles: int = 1  # done / control overhead per row
    gelu_page_words: int = 1024
    gelu_page_bytes_per_word: int = 1     # INT8 GELU output
    # Conservative FC2 model: hidden pages are reloaded once for each FC2 output tile.
    # Set to 1.0 in a more optimistic page scheduling model.
    gelu_ddr_reload_per_fc2_output_tile: bool = True

    @property
    def base_peak_macs_per_cycle(self) -> float:
        return float(self.pe_rows * self.pe_cols)

    @property
    def optimized_peak_macs_per_cycle(self) -> float:
        return self.base_peak_macs_per_cycle * self.dsp_packing_factor

    @property
    def effective_dram_bandwidth_Bps(self) -> float:
        return self.dram_peak_bandwidth_Bps * self.dram_efficiency

    @property
    def effective_dram_bytes_per_cycle(self) -> float:
        return self.effective_dram_bandwidth_Bps / self.clock_hz

    @property
    def gelu_page_bytes(self) -> int:
        return int(self.gelu_page_words * self.gelu_page_bytes_per_word)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EnergyConfig:
    """Coarse analytical energy model. Tune after Vivado/PYNQ measurements."""
    energy_per_fp32_mac_pj: float = 4.0
    energy_per_int8_mac_pj: float = 1.0
    energy_per_bram_byte_pj: float = 5.0
    energy_per_dram_byte_pj: float = 200.0
    energy_per_lut_access_pj: float = 1.0
    leakage_power_w: float = 0.2

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
