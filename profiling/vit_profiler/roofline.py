# from __future__ import annotations

# from pathlib import Path
# from typing import Optional

# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# def _compact_label(x: float) -> str:
#     if abs(x) >= 1e9:
#         return f"{x/1e9:.2f}G"
#     if abs(x) >= 1e6:
#         return f"{x/1e6:.2f}M"
#     if abs(x) >= 1e3:
#         return f"{x/1e3:.2f}K"
#     return f"{x:.2f}"


# def plot_roofline(df: pd.DataFrame, out_path: str | Path, title: str = "Roofline Model") -> None:
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     plot_df = df.dropna(subset=["operational_intensity", "performance_macs_per_cycle"]).copy()
#     if plot_df.empty:
#         return

#     fig, ax = plt.subplots(figsize=(10, 6.5))
#     xmax = max(float(plot_df["operational_intensity"].max()) * 1.20, 1.0)
#     xs = np.linspace(0, xmax, 300)

#     for model, sub in plot_df.groupby("model"):
#         peak = float(sub["peak_macs_per_cycle"].max())
#         bw = float(sub["peak_dram_bytes_per_cycle"].max())
#         ys = np.minimum(peak, xs * bw)
#         ax.plot(xs, ys, linewidth=2, label=f"{model} roof")

#     markers = {
#         "Baseline-B FP32 hardware-aware tiled": "o",
#         "Optimized INT8 hardware-aware": "s",
#     }
#     offsets = {
#         "Norm": (6, 8),
#         "QKV Projection": (6, -12),
#         "Attention Score": (6, 8),
#         "Softmax": (6, 8),
#         "Attention Value": (6, -12),
#         "Output Projection": (6, 8),
#         "MLP": (6, -12),
#         "Full MHSA": (6, 8),
#         "Full one block": (6, -12),
#     }

#     for model, sub in plot_df.groupby("model"):
#         ax.scatter(
#             sub["operational_intensity"],
#             sub["performance_macs_per_cycle"],
#             marker=markers.get(model, "o"),
#             s=72,
#             edgecolors="white",
#             linewidths=0.8,
#             label=f"{model} sections",
#         )
#         for _, row in sub.iterrows():
#             dx, dy = offsets.get(str(row["section"]), (6, 6))
#             if "Optimized" in str(model):
#                 dy = -dy
#             ax.annotate(
#                 str(row["section"]),
#                 (row["operational_intensity"], row["performance_macs_per_cycle"]),
#                 textcoords="offset points",
#                 xytext=(dx, dy),
#                 fontsize=8,
#             )

#     ax.set_xlabel("Operational intensity (MACs / DRAM byte)")
#     ax.set_ylabel("Performance (MACs / cycle)")
#     ax.set_title(title)
#     ax.grid(True, linestyle="--", linewidth=0.5)
#     ax.legend(fontsize=8)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=200)
#     plt.close(fig)


# def plot_bar_compare(df: pd.DataFrame, metric: str, out_path: str | Path, title: Optional[str] = None, ylabel: Optional[str] = None) -> None:
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     pivot = df.pivot(index="section", columns="model", values=metric)
#     ax = pivot.plot(kind="bar", figsize=(11, 5.5))
#     ax.set_title(title or metric)
#     ax.set_ylabel(ylabel or metric)
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
#     plt.xticks(rotation=35, ha="right")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def plot_memory_access_compare(df: pd.DataFrame, out_path: str | Path) -> None:
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     access = df[["model", "section", "dram_total_bytes", "bram_total_bytes"]].copy()
#     access = access.melt(id_vars=["model", "section"], value_vars=["dram_total_bytes", "bram_total_bytes"], var_name="memory", value_name="bytes")
#     access["label"] = access["model"] + "\n" + access["memory"].str.replace("_total_bytes", "")
#     pivot = access.pivot_table(index="section", columns="label", values="bytes", aggfunc="sum")
#     ax = pivot.plot(kind="bar", figsize=(13, 6))
#     ax.set_title("DRAM / BRAM access by section")
#     ax.set_ylabel("Access bytes")
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
#     plt.xticks(rotation=35, ha="right")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()


# def plot_memory_usage_compare(df: pd.DataFrame, out_path: str | Path) -> None:
#     out_path = Path(out_path)
#     out_path.parent.mkdir(parents=True, exist_ok=True)
#     usage = df[["model", "section", "dram_usage_bytes", "bram_usage_bytes"]].copy()
#     usage = usage.melt(id_vars=["model", "section"], value_vars=["dram_usage_bytes", "bram_usage_bytes"], var_name="memory", value_name="bytes")
#     usage["label"] = usage["model"] + "\n" + usage["memory"].str.replace("_usage_bytes", "")
#     pivot = usage.pivot_table(index="section", columns="label", values="bytes", aggfunc="max")
#     ax = pivot.plot(kind="bar", figsize=(13, 6))
#     ax.set_title("DRAM / BRAM peak usage by section")
#     ax.set_ylabel("Peak usage bytes")
#     ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
#     plt.xticks(rotation=35, ha="right")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()

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


def plot_roofline(
    df: pd.DataFrame,
    out_path: str | Path,
    title: str = "Roofline Model",
    exclude_sections: Optional[list[str]] = None,
) -> None:
    """Plot a DRAM roofline figure.

    By default, Norm is excluded because it is an elementwise/LUT-style
    operation and its operational intensity can be misleading in a DRAM
    roofline model. Pass exclude_sections=[] if you want to include every
    section.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if exclude_sections is None:
        exclude_sections = ["Norm"]

    plot_df = df.dropna(subset=["operational_intensity", "performance_macs_per_cycle"]).copy()
    if exclude_sections:
        plot_df = plot_df[~plot_df["section"].astype(str).isin(exclude_sections)].copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6.5))
    xmax = max(float(plot_df["operational_intensity"].max()) * 1.20, 1.0)
    xs = np.linspace(0, xmax, 300)

    for model, sub in plot_df.groupby("model"):
        peak = float(sub["peak_macs_per_cycle"].max())
        bw = float(sub["peak_dram_bytes_per_cycle"].max())
        ys = np.minimum(peak, xs * bw)
        ax.plot(xs, ys, linewidth=2, label=f"{model} roof")

    markers = {
        "Baseline-B FP32 hardware-aware tiled": "o",
        "Optimized INT8 hardware-aware": "s",
    }
    offsets = {
        "Norm": (6, 8),
        "QKV Projection": (6, -12),
        "Attention Score": (6, 8),
        "Softmax": (6, 8),
        "Attention Value": (6, -12),
        "Output Projection": (6, 8),
        "MLP": (6, -12),
        "Full MHSA": (6, 8),
        "Full one block": (6, -12),
    }

    for model, sub in plot_df.groupby("model"):
        ax.scatter(
            sub["operational_intensity"],
            sub["performance_macs_per_cycle"],
            marker=markers.get(model, "o"),
            s=72,
            edgecolors="white",
            linewidths=0.8,
            label=f"{model} sections",
        )
        for _, row in sub.iterrows():
            dx, dy = offsets.get(str(row["section"]), (6, 6))
            if "Optimized" in str(model):
                dy = -dy
            ax.annotate(
                str(row["section"]),
                (row["operational_intensity"], row["performance_macs_per_cycle"]),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=8,
            )

    ax.set_xlabel("Operational intensity (MACs / DRAM byte)")
    ax.set_ylabel("Performance (MACs / cycle)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bar_compare(df: pd.DataFrame, metric: str, out_path: str | Path, title: Optional[str] = None, ylabel: Optional[str] = None) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot(index="section", columns="model", values=metric)
    ax = pivot.plot(kind="bar", figsize=(11, 5.5))
    ax.set_title(title or metric)
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_memory_access_compare(df: pd.DataFrame, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    access = df[["model", "section", "dram_total_bytes", "bram_total_bytes"]].copy()
    access = access.melt(id_vars=["model", "section"], value_vars=["dram_total_bytes", "bram_total_bytes"], var_name="memory", value_name="bytes")
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
    usage = usage.melt(id_vars=["model", "section"], value_vars=["dram_usage_bytes", "bram_usage_bytes"], var_name="memory", value_name="bytes")
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
