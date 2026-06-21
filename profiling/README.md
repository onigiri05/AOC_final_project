# ViT Profiling Package v3

This package compares:

- **Baseline-B FP32 hardware-aware tiled**: same 16×16 tiling structure as the accelerator, but all tile operands/results access **DRAM** directly. It uses FP32, LayerNorm, standard softmax, no fusion, no ping-pong overlap, and no DSP packing.
- **Optimized INT8 hardware-aware**: INT8/QAT-style model with on-chip BRAM buffers, maximum weight reuse loop ordering, Streaming RMSNorm + LUT, Softmax LUT, operator fusion, ping-pong buffering, and DSP data packing.

The model is designed to match the updated project MD dataflow where GEMM loops use maximum weight reuse:

```text
for each N-tile:
    preload W[:, n_tile:n_tile+16] once
    for each M-tile:
        reuse the same W tile across token tiles
```

## Install

```bash
pip install torch timm pandas matplotlib tabulate
```

`timm` is optional if you are fine with the fallback ViT-Small/16 shape defaults.

## Run

```bash
python run_profile.py \
  --baseline-model vit_small_patch16_224.augreg_in21k_ft_in1k \
  --optimized-checkpoint ./rms_qat_best.pt \
  -o outputs_v3 \
  --clock-mhz 100 \
  --dram-eff 0.50
```

## Outputs

- `parsed_model_specs.json`
- `hardware_config.json`
- `profiling_results.csv/.md`
- `group_summary.csv/.md`
- `optimization_impact.csv/.md/.png`
- `roofline_sections.png`
- `roofline_groups.png`
- `macs_by_section.png`
- `dram_bytes_by_section.png`
- `bram_bytes_by_section.png`
- `dram_usage_by_section.png`
- `bram_usage_by_section.png`
- `dram_bram_access_by_section.png`
- `dram_bram_usage_by_section.png`
- `cycles_by_section.png`
- `energy_by_section.png`

## Notes

This is an analytical model, not a Vivado/PYNQ measurement. Replace the clock, leakage power, DRAM efficiency, and BRAM service bandwidth with measured values when available.

The DRAM roofline plots ignore points with zero DRAM traffic. On-chip-only sections should be interpreted using BRAM access/usage charts and cycles/energy tables.
