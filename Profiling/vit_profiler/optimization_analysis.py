from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASELINE = "Baseline-B FP32 hardware-aware tiled"
OPTIMIZED = "Optimized INT8 hardware-aware"

OPTIMIZATION_MAP = [
    {
        "optimization": "INT8 QAT",
        "affected_sections": "Almost all sections",
        "main_metric": "DRAM/BRAM bytes and energy",
        "explanation": "Activation and weight tensors shrink from FP32 4B to INT8 1B. Psum and bias remain INT32.",
    },
    {
        "optimization": "LayerNorm to Streaming RMSNorm + LUT",
        "affected_sections": "Norm",
        "main_metric": "Modeled operations, cycles, memory traffic, LUT accesses",
        "explanation": "Replaces LayerNorm-style multi-pass normalization with streaming RMSNorm and reciprocal-sqrt/LUT support.",
    },
    {
        "optimization": "Softmax LUT",
        "affected_sections": "Softmax",
        "main_metric": "Cycles, DRAM reduction, LUT accesses",
        "explanation": "Replaces expensive FP32 exp/div softmax with row-wise LUT softmax. True MACs are not counted for Softmax.",
    },
    {
        "optimization": "Operator Fusion / page streaming",
        "affected_sections": "MLP, Output Projection",
        "main_metric": "Intermediate materialization and memory traffic",
        "explanation": "Linear+GELU+requant and output projection+residual reduce FP32 intermediate materialization. Current RTL stores GELU pages through DDR instead of keeping full hidden_gelu on chip.",
    },
    {
        "optimization": "Ping-Pong Buffer",
        "affected_sections": "QKV Projection, Attention Score, Output Projection, MLP",
        "main_metric": "Latency cycles",
        "explanation": "Overlaps tile loading and computation when applicable. It may increase live buffer footprint but reduces exposed load latency.",
    },
    {
        "optimization": "DSP Data Packing",
        "affected_sections": "QKV Projection, Attention Score, Attention Value, Output Projection, MLP",
        "main_metric": "Effective MACs/cycle and compute cycles",
        "explanation": "Increases effective INT8 throughput. Mathematical MAC count is unchanged; compute cycles decrease through a higher peak MAC/cycle.",
    },
]


def _safe_ratio(base: float, opt: float) -> float:
    if opt == 0:
        return float("inf") if base > 0 else 1.0
    return base / opt


def build_optimization_impact(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["model"] == BASELINE].set_index("section")
    opt = df[df["model"] == OPTIMIZED].set_index("section")
    rows = []
    for item in OPTIMIZATION_MAP:
        affected = [s.strip() for s in item["affected_sections"].split(",")]
        if item["affected_sections"] == "Almost all sections":
            affected = sorted(set(base.index).intersection(set(opt.index)))
        affected = [s for s in affected if s in base.index and s in opt.index]
        if not affected:
            continue
        b = base.loc[affected]
        o = opt.loc[affected]
        b_ops = float(b["operations"].sum()) if "operations" in b.columns else 0.0
        o_ops = float(o["operations"].sum()) if "operations" in o.columns else 0.0
        rows.append({
            **item,
            "sections_used": ", ".join(affected),
            "mac_reduction_x": _safe_ratio(float(b["macs"].sum()), float(o["macs"].sum())),
            "operation_reduction_x": _safe_ratio(b_ops, o_ops),
            "dram_reduction_x": _safe_ratio(float(b["dram_total_bytes"].sum()), float(o["dram_total_bytes"].sum())),
            "bram_access_change_x_opt_over_base": _safe_ratio(float(o["bram_total_bytes"].sum()), float(b["bram_total_bytes"].sum())) if float(b["bram_total_bytes"].sum()) else float("inf"),
            "cycle_speedup_x": _safe_ratio(float(b["cycles_total"].sum()), float(o["cycles_total"].sum())),
            "energy_reduction_x": _safe_ratio(float(b["energy_total_uj"].sum()), float(o["energy_total_uj"].sum())),
        })
    return pd.DataFrame(rows)


def export_optimization_analysis(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    impact = build_optimization_impact(df)
    impact.to_csv(output_dir / "optimization_impact.csv", index=False)
    impact.to_markdown(output_dir / "optimization_impact.md", index=False)

    if not impact.empty:
        plot_df = impact[["optimization", "dram_reduction_x", "cycle_speedup_x", "energy_reduction_x"]].set_index("optimization")
        ax = plot_df.plot(kind="bar", figsize=(12, 5.5))
        ax.set_ylabel("Improvement ratio (baseline / optimized)")
        ax.set_title("Optimization impact summary")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "optimization_impact.png", dpi=200)
        plt.close()
    return impact
