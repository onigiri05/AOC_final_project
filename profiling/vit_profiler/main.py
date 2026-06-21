from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import HardwareConfig, EnergyConfig
from .model_parser import parse_timm_model, parse_checkpoint, save_specs
from .profiler import profile_block, make_group_summary
from .roofline import plot_roofline, plot_bar_compare, plot_memory_access_compare, plot_memory_usage_compare
from .optimization_analysis import export_optimization_analysis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ViT profiling package: Baseline-B FP32 hardware-aware tiled vs optimized INT8 checkpoint.")
    p.add_argument("--baseline-model", default="vit_small_patch16_224.augreg_in21k_ft_in1k", help="timm model name for Baseline-B")
    p.add_argument("--baseline-pretrained", action="store_true", help="load timm pretrained weights; normally not needed for shape parsing")
    p.add_argument("--optimized-checkpoint", default=None, help="path to optimized checkpoint such as rms_qat_best.pt")
    p.add_argument("--output-dir", "-o", default="outputs", help="output directory")
    p.add_argument("--clock-mhz", type=float, default=100.0, help="PL clock in MHz")
    p.add_argument("--dram-eff", type=float, default=0.50, help="effective DDR efficiency relative to 2.1 GB/s peak")
    p.add_argument("--bram-kb", type=float, default=560.0, help="effective BRAM capacity in KB")
    p.add_argument("--dsp-packing", type=float, default=2.0, help="effective INT8 MAC throughput multiplier")
    p.add_argument("--bram-service-bpc", type=float, default=32.0, help="local BRAM service bandwidth in bytes/cycle")
    p.add_argument("--leakage-w", type=float, default=0.20, help="leakage/static power assumption in W")
    p.add_argument("--no-plots", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    baseline_spec = parse_timm_model(args.baseline_model, pretrained=args.baseline_pretrained)
    if args.optimized_checkpoint:
        optimized_spec = parse_checkpoint(args.optimized_checkpoint, name=Path(args.optimized_checkpoint).name)
    else:
        optimized_spec = baseline_spec.__class__(**{**baseline_spec.to_dict(), "name": "optimized_default", "source": "baseline_spec_copy_no_checkpoint"})

    hw = HardwareConfig(
        clock_hz=args.clock_mhz * 1e6,
        dram_efficiency=args.dram_eff,
        bram_capacity_bytes=int(args.bram_kb * 1024),
        dsp_packing_factor=args.dsp_packing,
        bram_service_bytes_per_cycle=args.bram_service_bpc,
    )
    energy = EnergyConfig(leakage_power_w=args.leakage_w)

    save_specs({"baseline_b": baseline_spec, "optimized": optimized_spec}, out / "parsed_model_specs.json")
    with open(out / "hardware_config.json", "w", encoding="utf-8") as f:
        json.dump({"hardware": hw.to_dict(), "energy": energy.to_dict()}, f, indent=2)

    df = profile_block(baseline_spec, optimized_spec, hw, energy)
    df.to_csv(out / "profiling_results.csv", index=False)
    df.to_markdown(out / "profiling_results.md", index=False)

    group_df = make_group_summary(df)
    if not group_df.empty:
        group_df["latency_ms"] = group_df["cycles_total"] / hw.clock_hz * 1e3
        group_df.to_csv(out / "group_summary.csv", index=False)
        group_df.to_markdown(out / "group_summary.md", index=False)

    export_optimization_analysis(df, out)

    if not args.no_plots:
        plot_roofline(df, out / "roofline_sections.png", title="Section-level DRAM roofline")
        if not group_df.empty:
            plot_roofline(group_df, out / "roofline_groups.png", title="Group-level DRAM roofline")
        plot_bar_compare(df, "macs", out / "macs_by_section.png", title="MACs / effective operations by section", ylabel="MACs / modeled ops")
        plot_bar_compare(df, "dram_total_bytes", out / "dram_bytes_by_section.png", title="DRAM access bytes by section", ylabel="DRAM access bytes")
        plot_bar_compare(df, "bram_total_bytes", out / "bram_bytes_by_section.png", title="BRAM access bytes by section", ylabel="BRAM access bytes")
        plot_bar_compare(df, "dram_usage_bytes", out / "dram_usage_by_section.png", title="DRAM peak usage by section", ylabel="DRAM peak usage bytes")
        plot_bar_compare(df, "bram_usage_bytes", out / "bram_usage_by_section.png", title="BRAM peak usage by section", ylabel="BRAM peak usage bytes")
        plot_bar_compare(df, "cycles_total", out / "cycles_by_section.png", title="Cycles by section", ylabel="cycles")
        plot_bar_compare(df, "energy_total_uj", out / "energy_by_section.png", title="Energy by section", ylabel="energy (uJ)")
        plot_memory_access_compare(df, out / "dram_bram_access_by_section.png")
        plot_memory_usage_compare(df, out / "dram_bram_usage_by_section.png")

    print(f"Done. Results saved to: {out.resolve()}")
    print("Parsed model specs:")
    print(json.dumps({"baseline_b": baseline_spec.to_dict(), "optimized": optimized_spec.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
