from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _compact_label(x: float) -> str:
    if abs(x) >= 1e9:
        return f"{x/1e9:.2f}G"
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f}M"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f}K"
    return f"{x:.2f}"


def _default_offsets() -> dict[tuple[str, str], tuple[int, int]]:
    """Manual label offsets tuned for this roofline plot.

    Key = (model_type, section), where model_type is either "Baseline",
    "Optimized", or "Any".
    """
    return {
        ("Baseline", "QKV Projection"): (8, -14),
        ("Baseline", "Attention Score"): (8, 10),
        ("Baseline", "Softmax"): (8, -24),
        ("Baseline", "Attention Value"): (8, 0),
        ("Baseline", "Output Projection"): (8, 20),
        ("Baseline", "MLP"): (8, -6),
        ("Baseline", "Norm"): (8, 12),
        ("Optimized", "QKV Projection"): (8, 16),
        ("Optimized", "Attention Score"): (8, -18),
        ("Optimized", "Softmax"): (8, -18),
        ("Optimized", "Attention Value"): (8, -2),
        ("Optimized", "Output Projection"): (8, -12),
        ("Optimized", "MLP"): (8, 3),
        ("Optimized", "Norm"): (8, -14),
        ("Any", "Full MHSA"): (8, 8),
        ("Any", "Full one block"): (8, -12),
    }


def _model_type(model: str) -> str:
    return "Optimized" if "Optimized" in str(model) else "Baseline"


def _label_offset(model: str, section: str, idx_within_cluster: int = 0) -> tuple[int, int]:
    offsets = _default_offsets()
    mtype = _model_type(model)
    if (mtype, section) in offsets:
        return offsets[(mtype, section)]
    if ("Any", section) in offsets:
        return offsets[("Any", section)]

    # Generic fallback if a new section appears.
    dx = 8
    dy_cycle = [12, -12, 22, -22, 6, -6]
    return dx, dy_cycle[idx_within_cluster % len(dy_cycle)]


def plot_roofline(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Roofline Model",
    exclude_sections: Optional[list[str]] = None,
    show_label_arrows: bool = True,
    legend_fontsize: int = 8,
    legend_markerscale: float = 1.0,
) -> None:
    """Plot a DRAM roofline figure.

    By default, Norm and Softmax are excluded because they are non-GEMM
    elementwise/reduction-style sections and their operational intensity can be
    misleading in a MAC-based DRAM roofline model. Pass exclude_sections=[] if you want to include every
    section.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if exclude_sections is None:
        exclude_sections = ["Norm", "Softmax"]

    plot_df = df.dropna(subset=["operational_intensity", "performance_macs_per_cycle"]).copy()
    if exclude_sections:
        plot_df = plot_df[~plot_df["section"].astype(str).isin(exclude_sections)].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 7))
    xmax = max(float(plot_df["operational_intensity"].max()) * 1.20, 1.0)
    xs = np.linspace(0, xmax, 300)

    for model, sub in plot_df.groupby("model"):
        peak = float(sub["peak_macs_per_cycle"].max())
        bw = float(sub["peak_dram_bytes_per_cycle"].max())
        ys = np.minimum(peak, xs * bw)
        ax.plot(xs, ys, linewidth=2, label=f"{model} roof")

    markers = {
        "Baseline FP32 hardware-aware": "o",
        "Optimized INT8 hardware-aware": "s",
    }

    for model, sub in plot_df.groupby("model"):
        ax.scatter(
            sub["operational_intensity"],
            sub["performance_macs_per_cycle"],
            marker=markers.get(model, "o"),
            s=88,
            edgecolors="white",
            linewidths=0.8,
            label=f"{model} sections",
            zorder=3,
        )

        # Cluster by rounded x-position so nearby labels get different fallback offsets.
        sub = sub.copy()
        sub["cluster"] = sub["operational_intensity"].round(1)
        for _, cluster_sub in sub.groupby("cluster"):
            for idx, (_, row) in enumerate(cluster_sub.iterrows()):
                dx, dy = _label_offset(str(model), str(row["section"]), idx)
                ax.annotate(
                    str(row["section"]),
                    (row["operational_intensity"], row["performance_macs_per_cycle"]),
                    textcoords="offset points",
                    xytext=(dx, dy),
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                    arrowprops=(
                        dict(arrowstyle="-", lw=0.6, alpha=0.5)
                        if show_label_arrows
                        else None
                    ),
                    zorder=4,
                )

    ax.set_xlabel("Operational intensity (MACs / DRAM byte)")
    ax.set_ylabel("Performance (MACs / cycle)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=legend_fontsize, markerscale=legend_markerscale)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bar_compare(
    df: pd.DataFrame,
    metric: str,
    out_path: str | Path,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.copy()
    if metric in {"macs", "math_macs"}:
        # Norm and Softmax are non-GEMM sections. Their true MAC count is 0 and
        # they should not appear in the MAC-by-section figure.
        plot_df = plot_df[~plot_df["section"].isin(["Norm", "Softmax"])].copy()

    pivot = plot_df.pivot(index="section", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5))
    ax.set_title(title or metric)
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    # When main.py calls plot_bar_compare(..., metric="macs"), also produce a
    # separate Norm operations comparison chart without requiring a main.py change.
    if metric in {"macs", "math_macs"} and "operations" in df.columns:
        norm_ops_path = out_path.parent / "norm_operations_by_section.png"
        plot_norm_operations_compare(df, norm_ops_path)
        softmax_ops_path = out_path.parent / "softmax_operations_by_section.png"
        plot_softmax_operations_compare(df, softmax_ops_path)



def plot_norm_operations_compare(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Plot Baseline vs Optimized modeled operations for Norm only.

    Norm is a non-GEMM section, so it is excluded from true MAC plots. The
    separate operations column is used to compare LayerNorm vs Streaming RMSNorm.
    """
    if "operations" not in df.columns:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    focus_df = df[df["section"].astype(str) == "Norm"].copy()
    if focus_df.empty:
        return

    pivot = focus_df.pivot(index="section", columns="model", values="operations")
    ax = pivot.plot(kind="bar", figsize=(8, 5.5), width=0.72)

    ax.set_title(title or "Norm modeled operations")
    ax.set_ylabel(ylabel or "Modeled operations")
    ax.set_xlabel("Section")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=0)

    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(_compact_label(h) if pd.notna(h) and h != 0 else "")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()



def plot_softmax_operations_compare(
    df: pd.DataFrame,
    out_path: str | Path,
    *,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Plot Baseline vs Optimized modeled operations for Softmax only.

    Softmax is a non-GEMM section and is excluded from true MAC plots.  The
    operations column is used here as a modeled row-wise softmax cost.  In the
    updated profiler, Baseline FP32 Softmax uses parameterized exp/div IIs,
    while Optimized Softmax uses LUT/fixed-point IIs.
    """
    if "operations" not in df.columns:
        return

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    focus_df = df[df["section"].astype(str) == "Softmax"].copy()
    if focus_df.empty:
        return

    pivot = focus_df.pivot(index="section", columns="model", values="operations")
    ax = pivot.plot(kind="bar", figsize=(8, 5.5), width=0.72)

    ax.set_title(title or "Softmax modeled operations")
    ax.set_ylabel(ylabel or "Modeled row-wise softmax operations")
    ax.set_xlabel("Section")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=0)

    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(_compact_label(h) if pd.notna(h) and h != 0 else "")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_norm_softmax_macs_compare(
    df: pd.DataFrame,
    metric_or_out_path: str | Path = "math_macs",
    out_path: str | Path | None = None,
    *,
    metric: str | None = None,
    title: str | None = None,
    ylabel: str | None = None,
) -> None:
    """Compare only Norm and Softmax between baseline and optimized.

    Supports both call styles:

    1. New style:
       plot_norm_softmax_macs_compare(df, out_path, metric="math_macs")

    2. Old style:
       plot_norm_softmax_macs_compare(df, "cycles_total", out_path)
    """
    # Backward-compatible argument handling.
    if out_path is None:
        # New style: second positional argument is output path.
        out_path = metric_or_out_path
        metric_name = metric or "math_macs"
    else:
        # Old style: second positional argument is metric, third is output path.
        metric_name = metric or str(metric_or_out_path)

    if metric_name not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise KeyError(
            f"Metric column '{metric_name}' not found in DataFrame. "
            f"Available columns are: {available}"
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    focus_sections = ["Norm", "Softmax"]
    focus_df = df[df["section"].isin(focus_sections)].copy()
    if focus_df.empty:
        return

    pivot = focus_df.pivot(index="section", columns="model", values=metric_name)
    ax = pivot.plot(kind="bar", figsize=(8, 5.5), width=0.72)

    pretty_metric = metric_name.replace("_", " ")
    ax.set_title(title or f"Norm and Softmax {pretty_metric} by section")
    ax.set_ylabel(ylabel or pretty_metric)
    ax.set_xlabel("Section")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=0)

    # Add value labels on bars.
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append(_compact_label(h) if pd.notna(h) and h != 0 else "")
        ax.bar_label(container, labels=labels, padding=3, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_memory_access_compare(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    access = df[["model", "section", "dram_total_bytes", "bram_total_bytes"]].copy()
    access = access.melt(
        id_vars=["model", "section"],
        value_vars=["dram_total_bytes", "bram_total_bytes"],
        var_name="memory",
        value_name="bytes",
    )
    access["label"] = access["model"] + "\n" + access["memory"].str.replace("_total_bytes", "")
    pivot = access.pivot_table(index="section", columns="label", values="bytes", aggfunc="sum")
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title("DRAM / BRAM access by section")
    ax.set_ylabel("Access bytes")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_memory_usage_compare(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    usage = df[["model", "section", "dram_usage_bytes", "bram_usage_bytes"]].copy()
    usage = usage.melt(
        id_vars=["model", "section"],
        value_vars=["dram_usage_bytes", "bram_usage_bytes"],
        var_name="memory",
        value_name="bytes",
    )
    usage["label"] = usage["model"] + "\n" + usage["memory"].str.replace("_usage_bytes", "")
    pivot = usage.pivot_table(index="section", columns="label", values="bytes", aggfunc="max")
    ax = pivot.plot(kind="bar", figsize=(13, 6))
    ax.set_title("DRAM / BRAM peak usage by section")
    ax.set_ylabel("Peak usage bytes")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
