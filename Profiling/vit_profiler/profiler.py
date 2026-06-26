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
    operations: int
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


def _gemm_macs(M: int, K: int, N: int) -> int:
    return int(M * K * N)


def _padded_tokens(spec: ViTModelSpec, hw: HardwareConfig) -> int:
    """Pad tokens to the actual RTL tile size rather than always to 16."""
    if hasattr(spec, "padded_tokens_for_tile"):
        return spec.padded_tokens_for_tile(hw.tile_m)
    return _ceil_div(spec.tokens, hw.tile_m) * hw.tile_m


def _gemm_dram_access_all_dram(M: int, K: int, N: int, bp: int, tile: int = 8, include_bias: bool = True) -> tuple[int, int]:
    """Loop-accurate all-DRAM tiled GEMM access.

    Baseline-B uses the same RTL tile size but no BRAM/local reuse: every
    activation/weight tile feed and output tile write is modeled as DRAM traffic.
    """
    mt = _ceil_div(M, tile)
    nt = _ceil_div(N, tile)
    kt = _ceil_div(K, tile)
    act_tile = tile * tile * bp
    w_tile = tile * tile * bp
    out_tile = tile * tile * bp
    read = mt * nt * kt * (act_tile + w_tile)
    if include_bias:
        read += mt * nt * tile * bp
    write = mt * nt * out_tile
    return int(read), int(write)


def _gemm_bram_access_weight_reuse(M: int, K: int, N: int, act_bp: int, w_bp: int, out_bp: int, tile: int = 8) -> tuple[int, int]:
    """Loop-accurate BRAM/weight-buffer access with N-tile weight reuse."""
    mt = _ceil_div(M, tile)
    nt = _ceil_div(N, tile)
    kt = _ceil_div(K, tile)
    act_tile = tile * tile * act_bp
    w_tile = tile * tile * w_bp
    out_tile = tile * tile * out_bp
    bram_read = mt * nt * kt * (act_tile + w_tile)
    bram_write = mt * nt * out_tile
    return int(bram_read), int(bram_write)


def _hw_param(hw: HardwareConfig, name: str, default: int | float) -> int | float:
    """Read an optional HardwareConfig attribute while staying compatible with older config.py files."""
    return getattr(hw, name, default)


def _baseline_softmax_cycles_and_ops(rows: int, Tp: int, hw: HardwareConfig) -> tuple[int, int, int, int]:
    """Row-wise FP32 softmax latency with explicit exp/div initiation intervals.

    The baseline keeps the all-DRAM memory model, but its cycle floor now models
    FP32 special functions as more expensive than a LUT lookup:

        load + rowmax + exp_ii*exp/sum + div_ii*normalize + overhead

    Returns (cycles, modeled_operations, exp_ii, div_ii).
    """
    exp_ii = int(_hw_param(hw, "baseline_softmax_exp_ii", 4))
    div_ii = int(_hw_param(hw, "baseline_softmax_div_ii", 4))
    overhead = int(_hw_param(hw, "softmax_row_overhead_cycles", 1))
    cycles = rows * ((1 + 1 + exp_ii + div_ii) * Tp + overhead)
    ops = rows * ((1 + 1 + exp_ii + div_ii) * Tp)
    return int(cycles), int(ops), exp_ii, div_ii


def _optimized_softmax_cycles_and_ops(rows: int, Tp: int, hw: HardwareConfig) -> tuple[int, int, int, int]:
    """Row-wise LUT softmax latency with parameterized LUT/normalize IIs.

    The default corresponds to a 4-pass one-element-per-cycle FSM:
        load + rowmax + LUT/sum + fixed-point normalize + overhead

    Returns (cycles, modeled_operations, lut_ii, norm_ii).
    """
    lut_ii = int(_hw_param(hw, "optimized_softmax_lut_ii", 1))
    norm_ii = int(_hw_param(hw, "optimized_softmax_norm_ii", 1))
    overhead = int(_hw_param(hw, "softmax_row_overhead_cycles", 1))
    cycles = rows * ((1 + 1 + lut_ii + norm_ii) * Tp + overhead)
    ops = rows * ((1 + 1 + lut_ii + norm_ii) * Tp)
    return int(cycles), int(ops), lut_ii, norm_ii


def _finalize(
    *,
    model: str,
    spec: ViTModelSpec,
    section: SectionShape,
    dtype: str,
    math_macs: int,
    operations: int,
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
    cycle_floor: float | None = None,
    cycle_floor_note: str | None = None,
) -> ProfilingResult:
    dram_total = dram_read + dram_write
    bram_total = bram_read + bram_write
    compute_work = math_macs + operations
    compute_cycles = compute_work / peak if peak > 0 else 0.0
    dram_bpc = hw.effective_dram_bytes_per_cycle
    dram_cycles = dram_total / dram_bpc if dram_bpc > 0 else 0.0
    bram_cycles = bram_total / hw.bram_service_bytes_per_cycle if hw.bram_service_bytes_per_cycle > 0 else 0.0

    if use_pingpong and hw.pingpong_enabled:
        cycles_total = max(compute_cycles, dram_cycles, bram_cycles)
        notes.append("Ping-pong uses max(compute, DRAM preload, BRAM service) as the latency lower bound.")
    else:
        cycles_total = compute_cycles + dram_cycles + bram_cycles

    if cycle_floor is not None:
        cycles_total = max(cycles_total, cycle_floor)
        if cycle_floor_note:
            notes.append(cycle_floor_note)

    latency_s = cycles_total / hw.clock_hz
    perf = math_macs / cycles_total if cycles_total > 0 else 0.0

    if dram_total > 0:
        oi = math_macs / dram_total if math_macs > 0 else float("nan")
        balance = peak / dram_bpc if dram_bpc > 0 else float("inf")
        if bram_cycles > max(compute_cycles, dram_cycles):
            bound = "on-chip-memory"
        else:
            bound = "compute" if (math_macs > 0 and oi > balance) else "memory"
    else:
        oi = float("nan")
        bound = "on-chip"

    total_mem = dram_total + bram_total
    total_mem_oi = math_macs / total_mem if (total_mem > 0 and math_macs > 0) else float("nan")

    e_compute = compute_work * compute_energy_pj / 1e6
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
        operations=operations,
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
    """FP32 hardware-aware tiled baseline using the same 8x8 RTL tile size.

    Baseline-B remains an all-DRAM baseline. Therefore Q tile recomputation and
    current-head K/V BRAM lifetime changes do not apply to it: it has no modeled
    BRAM reuse and every tile feed/result is DRAM traffic.
    """
    T, Tp, D, H, Dh, M = spec.tokens, _padded_tokens(spec, hw), spec.embed_dim, spec.num_heads, spec.head_dim, spec.mlp_dim
    fp32 = 4
    tile = hw.tile_m
    dram_read = dram_write = 0
    dram_usage = 0
    math_macs = 0
    operations = 0
    lut_accesses = 0
    cycle_floor = None
    cycle_floor_note = None
    notes = ["Baseline-B: FP32 all-DRAM tiled model using the same RTL 8x8 tile size; no BRAM reuse."]

    if section.name == "Norm":
        x = Tp * D
        math_macs = 0
        operations = 2 * Tp * (8 * D + 2)
        per_ln_read = 3 * _bytes(x, fp32) + _bytes(2 * D, fp32) + 3 * _bytes(Tp, fp32)
        per_ln_write = _bytes(x, fp32) + 2 * _bytes(Tp, fp32)
        dram_read = 2 * per_ln_read
        dram_write = 2 * per_ln_write
        dram_usage = 2 * _bytes(x, fp32) + _bytes(4 * D, fp32) + _bytes(4 * Tp, fp32)
        notes.append("LayerNorm true MACs are 0; detailed modeled operations are reported separately.")

    elif section.name == "QKV Projection":
        math_macs = _gemm_macs(Tp, D, 3 * D)
        r, w = _gemm_dram_access_all_dram(Tp, D, 3 * D, fp32, tile)
        dram_read, dram_write = r, w
        dram_usage = _bytes(Tp * D + D * 3 * D + Tp * 3 * D + 3 * D, fp32)

    elif section.name == "Attention Score":
        math_macs = H * _gemm_macs(Tp, Dh, Tp)
        for _ in range(H):
            r, w = _gemm_dram_access_all_dram(Tp, Dh, Tp, fp32, tile, include_bias=False)
            dram_read += r
            dram_write += w
        dram_usage = _bytes(H * (2 * Tp * Dh + Tp * Tp), fp32)

    elif section.name == "Softmax":
        elems = H * Tp * Tp
        rows = H * Tp
        math_macs = 0
        # Softmax is non-GEMM, so true MACs remain 0.  The operations column is
        # used only as a modeled row-wise softmax operation / special-function
        # cost for the Softmax operations comparison chart.
        #
        # Baseline still uses the all-DRAM memory model, but its row-wise FSM
        # latency now includes parameterized FP32 exp/div initiation intervals.
        # Pass 1: load/shift score       -> 1 element/cycle
        # Pass 2: row max                -> 1 element/cycle
        # Pass 3: FP32 exp + sum         -> baseline_softmax_exp_ii element-cycles
        # Pass 4: FP32 div normalize     -> baseline_softmax_div_ii element-cycles
        dram_read = 3 * _bytes(elems, fp32)
        dram_write = 2 * _bytes(elems, fp32)
        dram_usage = 3 * _bytes(elems, fp32)
        softmax_cycles, operations, exp_ii, div_ii = _baseline_softmax_cycles_and_ops(rows, Tp, hw)
        cycle_floor = float(softmax_cycles)
        cycle_floor_note = (
            f"Baseline FP32 Softmax uses row-wise FSM latency with explicit exp/div IIs: "
            f"rows*((2+exp_ii+div_ii)*Tp+overhead) = {rows}*((2+{exp_ii}+{div_ii})*{Tp}+"
            f"{int(_hw_param(hw, 'softmax_row_overhead_cycles', 1))})."
        )
        notes.append(
            "Standard FP32 softmax all-DRAM memory model. True MACs are 0 because Softmax is "
            "non-GEMM; the operations column reports modeled row-wise softmax cost with FP32 "
            f"exp/div initiation intervals exp_ii={exp_ii}, div_ii={div_ii}."
        )

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
        r += _bytes(Tp * D, fp32)  # residual read
        w += _bytes(Tp * D, fp32)  # residual output write
        dram_read, dram_write = r, w
        dram_usage = _bytes(Tp * D + D * D + Tp * D + D, fp32)

    elif section.name == "MLP":
        math_macs = _gemm_macs(Tp, D, M) + _gemm_macs(Tp, M, D) + 4 * Tp * M
        r1, w1 = _gemm_dram_access_all_dram(Tp, D, M, fp32, tile)
        gelu_read = _bytes(Tp * M, fp32)
        gelu_write = _bytes(Tp * M, fp32)
        r2, w2 = _gemm_dram_access_all_dram(Tp, M, D, fp32, tile)
        res_r = _bytes(Tp * D, fp32)
        res_w = _bytes(Tp * D, fp32)
        dram_read = r1 + gelu_read + r2 + res_r
        dram_write = w1 + gelu_write + w2 + res_w
        dram_usage = _bytes(Tp * D + D * M + Tp * M + M * D + Tp * D + M + D, fp32)
        notes.append("No FC1+GELU fusion: hidden activation is materialized in DRAM.")

    return _finalize(
        model="Baseline FP32 hardware-aware",
        spec=spec,
        section=section,
        dtype="FP32",
        math_macs=math_macs,
        operations=operations,
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
        cycle_floor=cycle_floor,
        cycle_floor_note=cycle_floor_note,
    )


def optimized_int8_profile_section(section: SectionShape, spec: ViTModelSpec, hw: HardwareConfig, energy: EnergyConfig) -> ProfilingResult:
    """INT8 hardware-aware model aligned with current 8x8 RTL dataflow."""
    T, Tp, D, H, Dh, M = spec.tokens, _padded_tokens(spec, hw), spec.embed_dim, spec.num_heads, spec.head_dim, spec.mlp_dim
    int8, int16, int32 = 1, 2, 4
    tile = hw.tile_m
    dram_read = dram_write = bram_read = bram_write = 0
    bram_usage = dram_usage = 0
    math_macs = 0
    operations = 0
    lut_accesses = 0
    cycle_floor = None
    cycle_floor_note = None
    notes: List[str] = ["Optimized INT8 RTL-aware model: 8x8 shared systolic, physical buffer lifetime/sharing, no PPU profiling."]

    x_bytes = _bytes(Tp * D, int8)
    k_head_bytes = _bytes(Tp * Dh, int8)
    v_head_bytes = _bytes(Tp * Dh, int8)
    q_cache_bytes = _bytes(tile * Dh, int8)  # only current query tile Q_h, not full Q
    out_attn_bytes = _bytes(Tp * D, int8)
    weight_pingpong_d384 = _bytes(D * tile * 2, int8)
    weight_pingpong_m1536 = _bytes(M * tile * 2, int8)
    gelu_page_bytes = hw.gelu_page_bytes

    if section.name == "Norm":
        elems = Tp * D
        math_macs = 0
        operations = 2 * 2 * elems  # apply-only RMSNorm: x*inv_rms and *gamma for two norms
        bram_read = 2 * (_bytes(elems, int8) + _bytes(Tp, int16) + _bytes(D, int16))
        bram_write = 2 * _bytes(elems, int8)
        bram_usage = 2 * x_bytes + _bytes(Tp, int16) + 2 * _bytes(D, int16)
        dram_read = 2 * _bytes(D, int16)
        dram_usage = 2 * _bytes(D, int16)
        lut_accesses = 2 * Tp
        notes.append("Streaming RMSNorm apply-only stage; true MACs are 0 and modeled operations are reported separately.")

    elif section.name == "QKV Projection":
        # Current RTL precomputes current-head K/V and does not store full Q.
        # The Q projection work is moved into Attention Score as Q tile recomputation.
        math_macs = _gemm_macs(Tp, D, 2 * D)
        dram_read = _bytes(D * 2 * D, int8) + _bytes(2 * D, int32)
        dram_usage = _bytes(D * 2 * D, int8) + _bytes(2 * D, int32)
        bram_read, bram_write = _gemm_bram_access_weight_reuse(Tp, D, 2 * D, int8, int8, int8, tile)
        # Physical live set: X_norm + current-head K/V + K/V weight ping-pong.
        # Full Q and full K/V logical buffers are not simultaneously stored.
        bram_usage = x_bytes + k_head_bytes + v_head_bytes + weight_pingpong_d384
        notes.append("K/V precompute only: full Q is not stored; current-head K/V buffers are used to reduce BRAM footprint.")

    elif section.name == "Attention Score":
        # Q tile recomputation + QK^T.
        q_recompute_macs = H * _gemm_macs(Tp, D, Dh)      # equals Tp*D*D across all heads
        qkt_macs = H * _gemm_macs(Tp, Dh, Tp)
        math_macs = q_recompute_macs + qkt_macs

        # Q weights/bias are loaded here because Q was not produced/stored in QKV Projection.
        dram_read = _bytes(D * D, int8) + _bytes(D, int32)
        dram_usage = _bytes(D * D, int8) + _bytes(D, int32)

        # BRAM access for Q recompute: [Tp,D] x [D,D] -> Q, accumulated over all heads.
        br_q, bw_q = _gemm_bram_access_weight_reuse(Tp, D, D, int8, int8, int8, tile)
        bram_read += br_q
        bram_write += bw_q

        # BRAM access for QK^T per head: [Tp,Dh] x [Dh,Tp] -> [Tp,Tp] INT32 score.
        for _ in range(H):
            r, w = _gemm_bram_access_weight_reuse(Tp, Dh, Tp, int8, int8, int32, tile)
            bram_read += r
            bram_write += w

        score_head = _bytes(Tp * Tp, int32)
        bram_usage = x_bytes + k_head_bytes + v_head_bytes + q_cache_bytes + score_head + weight_pingpong_d384
        notes.append("Includes PH_Q_TILE recomputation before QK^T. Only a small current-query Q cache is live; full Q is not stored.")

    elif section.name == "Softmax":
        elems = H * Tp * Tp
        rows = H * Tp
        score_head = _bytes(Tp * Tp, int32)
        attn_head = _bytes(Tp * Tp, int8)
        math_macs = 0
        # Softmax is non-GEMM, so true MACs remain 0.  The operations column is
        # a modeled row-wise cost for comparing FP32 exp/div vs LUT/fixed-point
        # softmax.
        bram_read = H * score_head
        bram_write = H * attn_head
        # During Softmax, X_norm is still needed for later heads. V current-head remains live for A*V.
        # K can be released after QK^T, so it is not included in this physical live set.
        bram_usage = x_bytes + v_head_bytes + max(score_head, attn_head)
        lut_accesses = elems
        softmax_cycles, operations, lut_ii, norm_ii = _optimized_softmax_cycles_and_ops(rows, Tp, hw)
        cycle_floor = float(softmax_cycles)
        cycle_floor_note = (
            f"Optimized LUT Softmax uses row-wise FSM latency: "
            f"rows*((2+lut_ii+norm_ii)*Tp+overhead) = {rows}*((2+{lut_ii}+{norm_ii})*{Tp}+"
            f"{int(_hw_param(hw, 'softmax_row_overhead_cycles', 1))})."
        )
        notes.append(
            "Score/A share BRAM; row input/output is streamed. Softmax true MACs are 0. "
            f"The operations column reports modeled LUT/fixed-point row-wise cost with lut_ii={lut_ii}, norm_ii={norm_ii}."
        )

    elif section.name == "Attention Value":
        math_macs = H * _gemm_macs(Tp, Tp, Dh)
        for _ in range(H):
            r, w = _gemm_bram_access_weight_reuse(Tp, Tp, Dh, int8, int8, int8, tile)
            bram_read += r
            bram_write += w
        attn_head = _bytes(Tp * Tp, int8)
        bram_usage = x_bytes + v_head_bytes + attn_head + out_attn_bytes
        notes.append("A from shared Score/A BRAM and current-head V from u_v_xmid_bram; O_attn is written into u_norm_bram.")

    elif section.name == "Output Projection":
        math_macs = _gemm_macs(Tp, D, D)
        dram_read = _bytes(D * D, int8) + _bytes(D, int32)
        dram_usage = _bytes(D * D, int8) + _bytes(D, int32)
        br, bw = _gemm_bram_access_weight_reuse(Tp, D, D, int8, int8, int8, tile)
        bram_read = br + _bytes(Tp * D, int8)  # residual X read
        bram_write = bw
        bram_usage = x_bytes + out_attn_bytes + weight_pingpong_d384
        notes.append("O_attn is read from u_norm_bram; residual X is read from u_x_bram; X_mid is written into u_v_xmid_bram.")

    elif section.name == "MLP":
        math_macs = _gemm_macs(Tp, D, M) + _gemm_macs(Tp, M, D)
        hidden_bytes = _bytes(Tp * M, int8)
        fc2_output_tiles = _ceil_div(D, tile)
        fc2_hidden_reload_factor = fc2_output_tiles if hw.gelu_ddr_reload_per_fc2_output_tile else 1

        # Weights/biases + GELU page streaming through DDR.
        dram_read = _bytes(D * M + M * D, int8) + _bytes(M + D, int32) + hidden_bytes * fc2_hidden_reload_factor
        dram_write = hidden_bytes
        dram_usage = _bytes(D * M + M * D, int8) + _bytes(M + D, int32) + hidden_bytes

        br1, bw1 = _gemm_bram_access_weight_reuse(Tp, D, M, int8, int8, int8, tile)
        br2, bw2 = _gemm_bram_access_weight_reuse(Tp, M, D, int8, int8, int8, tile)
        bram_read = br1 + br2 + _bytes(Tp * D, int8)
        bram_write = bw1 + bw2
        # Physical live set: no full hidden_gelu on chip. Use 1K-word GELU page BRAM.
        bram_usage = 2 * x_bytes + gelu_page_bytes + max(weight_pingpong_m1536, weight_pingpong_d384)
        lut_accesses = Tp * M
        notes.append(
            "FC1+GELU uses a 1K-word page BRAM and DDR page streaming. Full hidden_gelu is not kept on chip; "
            f"FC2 hidden DDR reload factor is {fc2_hidden_reload_factor}."
        )

    return _finalize(
        model="Optimized INT8 hardware-aware",
        spec=spec,
        section=section,
        dtype="INT8/INT32",
        math_macs=math_macs,
        operations=operations,
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
        cycle_floor=cycle_floor,
        cycle_floor_note=cycle_floor_note,
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
            operations = float(g["operations"].sum()) if "operations" in g.columns else 0.0
            cycles = float(g["cycles_total"].sum())
            dram = float(g["dram_total_bytes"].sum())
            bram = float(g["bram_total_bytes"].sum())
            dram_usage = float(g["dram_usage_bytes"].max())
            bram_usage = float(g["bram_usage_bytes"].max())
            energy_total = float(g["energy_total_uj"].sum())
            perf = macs / cycles if cycles else 0.0
            oi = macs / dram if (dram and macs > 0) else float("nan")
            total_mem = dram + bram
            tmi = macs / total_mem if (total_mem and macs > 0) else float("nan")
            peak = float(g["peak_macs_per_cycle"].max())
            bpc = float(g["peak_dram_bytes_per_cycle"].max())
            if dram == 0:
                bound = "on-chip"
            else:
                bound = "compute" if (macs > 0 and oi > (peak / bpc if bpc else float("inf"))) else "memory"
            rows.append({
                "model": model,
                "section": group_name,
                "kind": "group",
                "dtype": ",".join(sorted(set(map(str, g["dtype"])))) ,
                "macs": macs,
                "math_macs": macs,
                "operations": operations,
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
                "latency_ms": 0.0,
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
