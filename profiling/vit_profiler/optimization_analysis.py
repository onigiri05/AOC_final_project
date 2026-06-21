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
        "main_metric": "Cycles, memory traffic, LUT accesses",
        "explanation": "Replaces LayerNorm-style multi-pass normalization with streaming RMSNorm and reciprocal-sqrt/LUT support.",
    },
    {
        "optimization": "Softmax LUT",
        "affected_sections": "Softmax",
        "main_metric": "Elementwise cycles and special-function cost",
        "explanation": "Replaces expensive exp operation with LUT lookup for quantized score differences.",
    },
    {
        "optimization": "Operator Fusion",
        "affected_sections": "MLP, Output Projection",
        "main_metric": "Intermediate read/write bytes",
        "explanation": "FC1 + GELU + requant and output projection + residual reduce intermediate materialization.",
    },
    {
        "optimization": "Ping-Pong Buffer",
        "affected_sections": "QKV Projection, Output Projection, MLP",
        "main_metric": "Latency cycles",
        "explanation": "Overlaps tile loading and computation. It may increase BRAM usage but reduces exposed load latency.",
    },
    {
        "optimization": "DSP Data Packing",
        "affected_sections": "QKV Projection, Attention Score, Attention Value, Output Projection, MLP",
        "main_metric": "Effective MACs/cycle",
        "explanation": "Increases effective INT8 throughput. Mathematical MAC count is unchanged, but compute cycles decrease.",
    },
]


def _safe_ratio(base: float, opt: float) -> float: # 算 imporved ratio，如 DRAM 使用量減少幾倍
    if opt == 0:
        return float("inf") if base > 0 else 1.0
    return base / opt


def build_optimization_impact(df: pd.DataFrame) -> pd.DataFrame: # 只拿特定section算improvement
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
        rows.append({
            **item,
            "sections_used": ", ".join(affected),
            "dram_reduction_x": _safe_ratio(float(b["dram_total_bytes"].sum()), float(o["dram_total_bytes"].sum())), # DRAM traffic 減少幾倍
            "bram_access_change_x_opt_over_base": _safe_ratio(float(o["bram_total_bytes"].sum()), float(b["bram_total_bytes"].sum())) if float(b["bram_total_bytes"].sum()) else float("inf"), # optimized 的 BRAM access 是 baseline 的幾倍
            "cycle_speedup_x": _safe_ratio(float(b["cycles_total"].sum()), float(o["cycles_total"].sum())), # latency cycles 加速幾倍
            "energy_reduction_x": _safe_ratio(float(b["energy_total_uj"].sum()), float(o["energy_total_uj"].sum())), # 代表 energy 減少幾倍
        })
    return pd.DataFrame(rows)


def export_optimization_analysis(df: pd.DataFrame, output_dir: str | Path) -> pd.DataFrame: #　把optimization impact結果存成檔案和畫圖
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
