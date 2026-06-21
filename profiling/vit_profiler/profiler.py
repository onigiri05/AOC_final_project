from __future__ import annotations

from dataclasses import dataclass, asdict
from math import ceil
from typing import Dict, List

import pandas as pd

from .config import ViTModelSpec, HardwareConfig, EnergyConfig
from .sections import SectionShape, build_group_definitions


@dataclass(frozen=True)
class ProfilingResult:
    model: str
    model_source: str
    section: str
    kind: str
    dtype: str
    macs: int
    math_macs: int
    lut_accesses: int
    dram_read_bytes: int
    dram_write_bytes: int
    dram_total_bytes: int
    bram_read_bytes: int
    bram_write_bytes: int
    bram_total_bytes: int
    dram_usage_bytes: int
    bram_usage_bytes: int
    bram_usage_ramb36: int
    compute_cycles: float
    dram_cycles: float
    bram_cycles: float
    memory_cycles: float
    cycles_total: float
    latency_ms: float
    performance_macs_per_cycle: float
    operational_intensity: float
    total_memory_intensity: float
    peak_macs_per_cycle: float
    peak_dram_bytes_per_cycle: float
    bound: str
    energy_compute_uj: float
    energy_bram_uj: float
    energy_dram_uj: float
    energy_lut_uj: float
    energy_leakage_uj: float
    energy_total_uj: float
    notes: str

    def to_dict(self) -> Dict:
        return asdict(self)


def _bytes(elems: int | float, bytes_per_elem: int | float) -> int:
    return int(elems * bytes_per_elem)


def _ceil_div(a: int, b: int) -> int:
    return int(ceil(a / b))


def _tile_bytes(tm: int, tn: int, bytes_per_elem: int) -> int:
    return tm * tn * bytes_per_elem


def _gemm_macs(M: int, K: int, N: int) -> int:
    return int(M * K * N)


def _gemm_dram_access_all_dram(M: int, K: int, N: int, bp: int, tile: int = 16, include_bias: bool = True) -> tuple[int, int]:
    """Loop-accurate all-DRAM tiled GEMM access.

    Loop order is N-tile outer, M-tile inner, K-phase inner, but because this
    baseline has no BRAM/local reuse, each operand tile is read from DRAM whenever
    it is fed to the PE array.
    """
    mt = _ceil_div(M, tile)
    nt = _ceil_div(N, tile)
    kt = _ceil_div(K, tile)
    act_tile = tile * tile * bp
    w_tile = tile * tile * bp
    out_tile = tile * tile * bp
    # Feed A and W for every M/N/K tile operation.
    read = mt * nt * kt * (act_tile + w_tile)
    # Bias tile read once per output tile.
    if include_bias:
        read += mt * nt * tile * bp
    write = mt * nt * out_tile
    return int(read), int(write)


def _gemm_bram_access_weight_reuse(M: int, K: int, N: int, act_bp: int, w_bp: int, out_bp: int, tile: int = 16) -> tuple[int, int]:
    """Loop-accurate BRAM/weight-buffer access for max weight reuse ordering.

    DRAM preloads each W[:, n_tile:n_tile+16] once. During compute, activation
    and weight sub-tiles are still read from local buffers every K-phase.
    """
    mt = _ceil_div(M, tile)
    nt = _ceil_div(N, tile)
    kt = _ceil_div(K, tile)
    act_tile = tile * tile * act_bp
    w_tile = tile * tile * w_bp
    out_tile = tile * tile * out_bp
    bram_read = mt * nt * kt * (act_tile + w_tile)
    bram_write = mt * nt * out_tile
    return int(bram_read), int(bram_write)


def _finalize(
    *,
    model: str,
    spec: ViTModelSpec,
    section: SectionShape,
    dtype: str,
    math_macs: int,
    lut_accesses: int,
    dram_read: int,
    dram_write: int,
    bram_read: int,
    bram_write: int,
    dram_usage: int,
    bram_usage: int,
    peak: float,
    compute_energy_pj: float,
    hw: HardwareConfig,
    energy: EnergyConfig,
    notes: List[str],
    use_pingpong: bool = False,
) -> ProfilingResult:
    dram_total = dram_read + dram_write
    bram_total = bram_read + bram_write
    compute_cycles = math_macs / peak if peak > 0 else 0.0
    dram_bpc = hw.effective_dram_bytes_per_cycle
    dram_cycles = dram_total / dram_bpc if dram_bpc > 0 else 0.0
    bram_cycles = bram_total / hw.bram_service_bytes_per_cycle if hw.bram_service_bytes_per_cycle > 0 else 0.0

    if use_pingpong and hw.pingpong_enabled:
        # Ping-pong hides exposed DRAM preload behind compute when possible, while
        # local BRAM feeding is still a lower-bound service requirement.
        cycles_total = max(compute_cycles, dram_cycles, bram_cycles)
        notes.append("Ping-pong uses max(compute, DRAM preload, BRAM service) as the latency lower bound.")
    else:
        cycles_total = compute_cycles + dram_cycles + bram_cycles

    latency_s = cycles_total / hw.clock_hz
    perf = math_macs / cycles_total if cycles_total > 0 else 0.0

    if dram_total > 0:
        oi = math_macs / dram_total
        balance = peak / dram_bpc if dram_bpc > 0 else float("inf")
        # Report the most likely bottleneck among the modeled lower bounds.
        if bram_cycles > max(compute_cycles, dram_cycles):
            bound = "on-chip-memory"
        else:
            bound = "compute" if oi > balance else "memory"
    else:
        oi = float("nan")
        bound = "on-chip"

    total_mem = dram_total + bram_total
    total_mem_oi = math_macs / total_mem if total_mem > 0 else float("nan")

    e_compute = math_macs * compute_energy_pj / 1e6
    e_bram = bram_total * energy.energy_per_bram_byte_pj / 1e6
    e_dram = dram_total * energy.energy_per_dram_byte_pj / 1e6
    e_lut = lut_accesses * energy.energy_per_lut_access_pj / 1e6
    e_leak = energy.leakage_power_w * latency_s * 1e6

    return ProfilingResult(
        model=model,
        model_source=spec.source,
        section=section.name,
        kind=section.kind,
        dtype=dtype,
        macs=math_macs,
        math_macs=math_macs,
        lut_accesses=lut_accesses,
        dram_read_bytes=dram_read,
        dram_write_bytes=dram_write,
        dram_total_bytes=dram_total,
        bram_read_bytes=bram_read,
        bram_write_bytes=bram_write,
        bram_total_bytes=bram_total,
        dram_usage_bytes=dram_usage,
        bram_usage_bytes=bram_usage,
        bram_usage_ramb36=_ceil_div(bram_usage, 4096),
        compute_cycles=compute_cycles,
        dram_cycles=dram_cycles,
        bram_cycles=bram_cycles,
        memory_cycles=dram_cycles + bram_cycles,
        cycles_total=cycles_total,
        latency_ms=latency_s * 1e3,
        performance_macs_per_cycle=perf,
        operational_intensity=oi,
        total_memory_intensity=total_mem_oi,
        peak_macs_per_cycle=peak,
        peak_dram_bytes_per_cycle=dram_bpc,
        bound=bound,
        energy_compute_uj=e_compute,
        energy_bram_uj=e_bram,
        energy_dram_uj=e_dram,
        energy_lut_uj=e_lut,
        energy_leakage_uj=e_leak,
        energy_total_uj=e_compute + e_bram + e_dram + e_lut + e_leak,
        notes=" ".join(notes),
    )


def baseline_b_fp32_hardware_profile_section(section: SectionShape, spec: ViTModelSpec, hw: HardwareConfig, energy: EnergyConfig) -> ProfilingResult:
    """FP32 hardware-aware tiled baseline.

    It uses the same 16x16 tiling loop structure as the optimized design, but it
    does not model BRAM/local-buffer reuse. Every tile feed comes from DRAM and
    every tile result is written back to DRAM. It also uses LayerNorm, standard
    softmax, no operator fusion, no ping-pong overlap, and no DSP packing.
    """
    T, Tp, D, H, Dh, M = spec.tokens, spec.padded_tokens_16, spec.embed_dim, spec.num_heads, spec.head_dim, spec.mlp_dim
    fp32 = 4
    tile = hw.tile_m
    dram_read = dram_write = 0
    dram_usage = 0
    math_macs = 0
    lut_accesses = 0
    notes = ["Baseline-B: FP32 tiled model, all tile operands/results access DRAM; no BRAM reuse."]

    if section.name == "Norm":
        # Two LayerNorms. Approx: mean pass + variance pass + normalize pass per norm.
        x = Tp * D
        math_macs = 2 * 5 * x
        dram_read = 2 * (3 * _bytes(x, fp32) + _bytes(2 * D, fp32))
        dram_write = 2 * _bytes(x, fp32)
        dram_usage = 2 * _bytes(x, fp32) + _bytes(4 * D, fp32)
        notes.append("Uses LayerNorm-style multi-pass FP32 normalization.")

    elif section.name == "QKV Projection":
        math_macs = _gemm_macs(Tp, D, 3 * D)
        r, w = _gemm_dram_access_all_dram(Tp, D, 3 * D, fp32, tile)
        dram_read += r
        dram_write += w
        dram_usage = _bytes(Tp * D + D * 3 * D + Tp * 3 * D + 3 * D, fp32)

    elif section.name == "Attention Score":
        # Per head: [Tp,Dh] x [Dh,Tp] -> [Tp,Tp]
        math_macs = H * _gemm_macs(Tp, Dh, Tp)
        for _ in range(H):
            r, w = _gemm_dram_access_all_dram(Tp, Dh, Tp, fp32, tile, include_bias=False)
            dram_read += r
            dram_write += w
        dram_usage = _bytes(H * (2 * Tp * Dh + Tp * Tp), fp32)

    elif section.name == "Softmax":
        elems = H * Tp * Tp
        math_macs = 5 * elems
        # Standard stable softmax: read score for rowmax, read for exp/sum, read/write probabilities.
        dram_read = 3 * _bytes(elems, fp32)
        dram_write = _bytes(elems, fp32)
        dram_usage = 2 * _bytes(elems, fp32)
        notes.append("Uses standard FP32 softmax; exp/division modeled as elementwise operations, not LUT.")

    elif section.name == "Attention Value":
        math_macs = H * _gemm_macs(Tp, Tp, Dh)
        for _ in range(H):
            r, w = _gemm_dram_access_all_dram(Tp, Tp, Dh, fp32, tile, include_bias=False)
            dram_read += r
            dram_write += w
        dram_usage = _bytes(H * (Tp * Tp + Tp * Dh + Tp * Dh), fp32)

    elif section.name == "Output Projection":
        math_macs = _gemm_macs(Tp, D, D)
        r, w = _gemm_dram_access_all_dram(Tp, D, D, fp32, tile)
        # residual read and final write
        r += _bytes(Tp * D, fp32)
        w += _bytes(Tp * D, fp32)
        dram_read, dram_write = r, w
        dram_usage = _bytes(Tp * D + D * D + Tp * D + D, fp32)

    elif section.name == "MLP":
        # FC1 write hidden, GELU read/write hidden, FC2 read hidden and write output, residual read/write.
        math_macs = _gemm_macs(Tp, D, M) + _gemm_macs(Tp, M, D) + 4 * Tp * M
        r1, w1 = _gemm_dram_access_all_dram(Tp, D, M, fp32, tile)
        # GELU materialization.
        gelu_read = _bytes(Tp * M, fp32)
        gelu_write = _bytes(Tp * M, fp32)
        r2, w2 = _gemm_dram_access_all_dram(Tp, M, D, fp32, tile)
        # residual read/write.
        res_r = _bytes(Tp * D, fp32)
        res_w = _bytes(Tp * D, fp32)
        dram_read = r1 + gelu_read + r2 + res_r
        dram_write = w1 + gelu_write + w2 + res_w
        dram_usage = _bytes(Tp * D + D * M + Tp * M + M * D + Tp * D + M + D, fp32)
        notes.append("No FC1+GELU fusion: hidden activation is materialized in DRAM.")

    return _finalize(
        model="Baseline-B FP32 hardware-aware tiled",
        spec=spec,
        section=section,
        dtype="FP32",
        math_macs=math_macs,
        lut_accesses=lut_accesses,
        dram_read=dram_read,
        dram_write=dram_write,
        bram_read=0,
        bram_write=0,
        dram_usage=dram_usage,
        bram_usage=0,
        peak=hw.base_peak_macs_per_cycle,
        compute_energy_pj=energy.energy_per_fp32_mac_pj,
        hw=hw,
        energy=energy,
        notes=notes,
        use_pingpong=False,
    )


def optimized_int8_profile_section(section: SectionShape, spec: ViTModelSpec, hw: HardwareConfig, energy: EnergyConfig) -> ProfilingResult:
    """INT8 hardware-aware model aligned with maximum-weight-reuse md loop ordering."""
    T, Tp, D, H, Dh, M = spec.tokens, spec.padded_tokens_16, spec.embed_dim, spec.num_heads, spec.head_dim, spec.mlp_dim
    int8, int16, int32 = 1, 2, 4
    tile = hw.tile_m
    dram_read = dram_write = bram_read = bram_write = 0
    bram_usage = dram_usage = 0
    math_macs = 0
    lut_accesses = 0
    notes: List[str] = ["Optimized INT8 hardware-aware model with maximum weight reuse loop ordering."]

    x_bytes = _bytes(Tp * D, int8)
    q_bytes = _bytes(Tp * D, int8)
    k_bytes = _bytes(Tp * D, int8)
    v_bytes = _bytes(Tp * D, int8)
    qkv_bytes = _bytes(Tp * 3 * D, int8)
    weight_pingpong_d384 = _bytes(D * tile * 2, int8)
    weight_pingpong_m1536 = _bytes(M * tile * 2, int8)

    if section.name == "Norm":
        # RMSNorm1 + RMSNorm2, streaming from/to activation-residual banks.
        elems = Tp * D
        math_macs = 2 * 2 * elems
        bram_read = 2 * (_bytes(elems, int8) + _bytes(Tp, int16) + _bytes(D, int16))
        bram_write = 2 * _bytes(elems, int8)
        bram_usage = 2 * x_bytes + _bytes(Tp, int16) + 2 * _bytes(D, int16)
        dram_read = 2 * _bytes(D, int16)  # gamma preload approximation
        dram_usage = 2 * _bytes(D, int16)
        lut_accesses = 2 * Tp
        notes.append("LayerNorm is replaced by streaming RMSNorm with inv-rms/gamma buffers and LUT-like reciprocal sqrt support.")

    elif section.name == "QKV Projection":
        math_macs = _gemm_macs(Tp, D, 3 * D)
        # Maximum weight reuse: each [384,16] W tile is loaded once per N-tile, not once per M-tile.
        dram_read = _bytes(D * 3 * D, int8) + _bytes(3 * D, int32)
        dram_usage = _bytes(D * 3 * D, int8) + _bytes(3 * D, int32)
        bram_read, bram_write = _gemm_bram_access_weight_reuse(Tp, D, 3 * D, int8, int8, int8, tile)
        # Live buffers: residual X, temporary QKV / Q+K+V, plus ping-pong weight tile.
        bram_usage = _bytes(Tp * D, int8) + qkv_bytes + weight_pingpong_d384
        notes.append("W_qkv tile [384,16] is loaded once per N-tile and reused across all token M-tiles.")

    elif section.name == "Attention Score":
        math_macs = H * _gemm_macs(Tp, Dh, Tp)
        # Q/K/V are already on-chip. Access counts accumulate over all heads; live score buffer is per current head.
        for _ in range(H):
            r, w = _gemm_bram_access_weight_reuse(Tp, Dh, Tp, int8, int8, int32, tile)
            bram_read += r
            bram_write += w
        score_head = _bytes(Tp * Tp, int32)
        bram_usage = x_bytes + q_bytes + k_bytes + v_bytes + score_head
        notes.append("Q/K/V remain on-chip; current-head INT32 score buffer is materialized in shared intermediate BRAM.")

    elif section.name == "Softmax":
        elems = H * Tp * Tp
        score_head = _bytes(Tp * Tp, int32)
        attn_head = _bytes(Tp * Tp, int8)
        math_macs = 2 * elems
        # rowmax + exp/sum + normalize: two score reads, one INT8 attention write.
        bram_read = H * score_head
        bram_write = H * attn_head
        bram_usage = x_bytes + q_bytes + k_bytes + v_bytes + max(score_head, attn_head)
        lut_accesses = elems
        notes.append("Score is read from BRAM, exp is replaced by LUT, and INT8 attention overwrites score space per head.")

    elif section.name == "Attention Value":
        math_macs = H * _gemm_macs(Tp, Tp, Dh)
        for _ in range(H):
            r, w = _gemm_bram_access_weight_reuse(Tp, Tp, Dh, int8, int8, int8, tile)
            bram_read += r
            bram_write += w
        attn_head = _bytes(Tp * Tp, int8)
        out_attn = _bytes(Tp * D, int8)
        bram_usage = x_bytes + v_bytes + attn_head + out_attn
        notes.append("A_h and V_h are read from BRAM; O_h is written directly into O_attn layout.")

    elif section.name == "Output Projection":
        math_macs = _gemm_macs(Tp, D, D)
        dram_read = _bytes(D * D, int8) + _bytes(D, int32)
        dram_usage = _bytes(D * D, int8) + _bytes(D, int32)
        br, bw = _gemm_bram_access_weight_reuse(Tp, D, D, int8, int8, int8, tile)
        bram_read = br + _bytes(Tp * D, int8)  # residual read for add
        bram_write = bw  # final X_mid write after residual add
        bram_usage = x_bytes + _bytes(Tp * D, int8) + weight_pingpong_d384
        notes.append("W_o tile [384,16] is loaded once per output-channel tile and reused across all token M-tiles.")

    elif section.name == "MLP":
        # FC1 + GELU + FC2 with maximum weight reuse and fusion.
        math_macs = _gemm_macs(Tp, D, M) + _gemm_macs(Tp, M, D)
        dram_read = _bytes(D * M + M * D, int8) + _bytes(M + D, int32)
        dram_usage = _bytes(D * M + M * D, int8) + _bytes(M + D, int32)
        br1, bw1 = _gemm_bram_access_weight_reuse(Tp, D, M, int8, int8, int8, tile)
        # FC1 output is immediately sent through GELU and written as hidden_gelu.
        hidden_bytes = _bytes(Tp * M, int8)
        br2, bw2 = _gemm_bram_access_weight_reuse(Tp, M, D, int8, int8, int8, tile)
        # residual read for final add
        bram_read = br1 + br2 + _bytes(Tp * D, int8)
        bram_write = bw1 + bw2
        # Peak live: X_norm2 + X_mid residual + hidden_gelu + largest weight ping-pong tile.
        bram_usage = max(2 * x_bytes + hidden_bytes + weight_pingpong_m1536, 2 * x_bytes + weight_pingpong_d384)
        lut_accesses = Tp * M
        notes.append("FC1+GELU+requant fusion avoids FP32 hidden materialization; W_fc1/W_fc2 tiles are reused across all M-tiles.")

    return _finalize(
        model="Optimized INT8 hardware-aware",
        spec=spec,
        section=section,
        dtype="INT8/INT32",
        math_macs=math_macs,
        lut_accesses=lut_accesses,
        dram_read=dram_read,
        dram_write=dram_write,
        bram_read=bram_read,
        bram_write=bram_write,
        dram_usage=dram_usage,
        bram_usage=bram_usage,
        peak=hw.optimized_peak_macs_per_cycle,
        compute_energy_pj=energy.energy_per_int8_mac_pj,
        hw=hw,
        energy=energy,
        notes=notes,
        use_pingpong=section.kind in {"gemm", "gemm_elementwise"},
    )


def profile_block(baseline_spec: ViTModelSpec, optimized_spec: ViTModelSpec, hw: HardwareConfig, energy: EnergyConfig) -> pd.DataFrame:
    from .sections import build_vit_block_sections

    rows: List[Dict] = []
    for section in build_vit_block_sections(baseline_spec):
        rows.append(baseline_b_fp32_hardware_profile_section(section, baseline_spec, hw, energy).to_dict())
    for section in build_vit_block_sections(optimized_spec):
        rows.append(optimized_int8_profile_section(section, optimized_spec, hw, energy).to_dict())
    return pd.DataFrame(rows)


def make_group_summary(df: pd.DataFrame) -> pd.DataFrame:
    groups = build_group_definitions()
    rows = []
    for model, sub in df.groupby("model"):
        for group_name, section_names in groups.items():
            g = sub[sub["section"].isin(section_names)]
            if g.empty:
                continue
            macs = float(g["macs"].sum())
            cycles = float(g["cycles_total"].sum())
            dram = float(g["dram_total_bytes"].sum())
            bram = float(g["bram_total_bytes"].sum())
            dram_usage = float(g["dram_usage_bytes"].max())
            bram_usage = float(g["bram_usage_bytes"].max())
            energy_total = float(g["energy_total_uj"].sum())
            perf = macs / cycles if cycles else 0.0
            oi = macs / dram if dram else float("nan")
            total_mem = dram + bram
            tmi = macs / total_mem if total_mem else float("nan")
            peak = float(g["peak_macs_per_cycle"].max())
            bpc = float(g["peak_dram_bytes_per_cycle"].max())
            if dram == 0:
                bound = "on-chip"
            else:
                bound = "compute" if oi > (peak / bpc if bpc else float("inf")) else "memory"
            rows.append({
                "model": model,
                "section": group_name,
                "kind": "group",
                "dtype": ",".join(sorted(set(map(str, g["dtype"])))) ,
                "macs": macs,
                "math_macs": macs,
                "lut_accesses": float(g.get("lut_accesses", pd.Series([0])).sum()),
                "dram_read_bytes": float(g["dram_read_bytes"].sum()),
                "dram_write_bytes": float(g["dram_write_bytes"].sum()),
                "dram_total_bytes": dram,
                "bram_read_bytes": float(g["bram_read_bytes"].sum()),
                "bram_write_bytes": float(g["bram_write_bytes"].sum()),
                "bram_total_bytes": bram,
                "dram_usage_bytes": dram_usage,
                "bram_usage_bytes": bram_usage,
                "bram_usage_ramb36": _ceil_div(bram_usage, 4096),
                "compute_cycles": float(g["compute_cycles"].sum()),
                "dram_cycles": float(g["dram_cycles"].sum()),
                "bram_cycles": float(g["bram_cycles"].sum()),
                "memory_cycles": float(g["memory_cycles"].sum()),
                "cycles_total": cycles,
                "latency_ms": 0.0,  # overwritten by main with the selected clock
                "performance_macs_per_cycle": perf,
                "operational_intensity": oi,
                "total_memory_intensity": tmi,
                "peak_macs_per_cycle": peak,
                "peak_dram_bytes_per_cycle": bpc,
                "bound": bound,
                "energy_total_uj": energy_total,
                "notes": f"Group summary for {group_name}",
            })
    return pd.DataFrame(rows)
