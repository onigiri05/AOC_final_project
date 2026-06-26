"""Cumulative 12-block DLAU replay with per-stage PT-calibrated scales."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
import torch

from compare_block11 import (
    HERE, dequant, linear_dynamic, metric, quant_u8, rescale, rms_dynamic, scale_of,
    G, make_reference, preprocess, cosine, save_attention_heatmaps, save_metric_line_plots,
)
PT_CANDIDATES = [
    (HERE / "../../PT_DIR/rms_qat_best.pt").resolve(),
    (HERE.parent / "golden_gen" / "rms_qat_best.pt").resolve(),
]
PT = next((path for path in PT_CANDIDATES if path.exists()), PT_CANDIDATES[0])
IMG_DIR = HERE/"Image/"


def capture_all(model: torch.nn.Module, image: Path) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
    refs: list[dict[str, np.ndarray]] = [dict() for _ in range(12)]
    hooks = []

    def save_out(block: int, name: str):
        return lambda _m, _i, out: refs[block].__setitem__(name, out.detach().cpu().numpy()[0])

    def save_in(block: int, name: str):
        return lambda _m, inp: refs[block].__setitem__(name, inp[0].detach().cpu().numpy()[0])

    for i, block in enumerate(model.blocks):
        if hasattr(block.attn, "fused_attn"):
            block.attn.fused_attn = False
        hooks += [block.register_forward_pre_hook(save_in(i, "input")),
                  block.norm1.register_forward_hook(save_out(i, "norm1")),
                  block.attn.qkv.register_forward_hook(save_out(i, "qkv")),
                  block.attn.attn_drop.register_forward_hook(save_out(i, "prob")),
                  block.attn.proj.register_forward_pre_hook(save_in(i, "attn_value")),
                  block.attn.proj.register_forward_hook(save_out(i, "projection")),
                  block.norm2.register_forward_pre_hook(save_in(i, "attn_residual")),
                  block.norm2.register_forward_hook(save_out(i, "norm2")),
                  block.mlp.act.register_forward_hook(save_out(i, "gelu")),
                  block.mlp.fc2.register_forward_hook(save_out(i, "fc2")),
                  block.register_forward_hook(save_out(i, "output"))]
    hooks.append(model.norm.register_forward_hook(save_out(11, "final_norm")))
    with torch.no_grad():
        logits = model(preprocess(image))[0].detach().cpu().numpy().astype(np.float64)
    for hook in hooks:
        hook.remove()
    return refs, logits


def run_block(state: dict[str, torch.Tensor], block: int, ref: dict[str, np.ndarray],
              xq: np.ndarray, xscale: float, exp_lut: np.ndarray,
              gelu_rom: np.ndarray, image: Path | None = None,
              heatmap_dir: Path | None = None,
              save_heads: bool = False) -> tuple[np.ndarray, float, dict[str, Any]]:
    a = lambda key: G.t(state, key).astype(np.float64)
    p = f"blocks.{block}"
    rows: list[dict[str, Any]] = [metric("input", ref["input"], xq, xscale)]
    shifts: dict[str, int] = {}

    n1, n1s, shifts["norm1_left"] = rms_dynamic(
        xq, ref["input"], a(p + ".norm1.weight"), xscale, ref["norm1"])
    rows.append(metric("norm1", ref["norm1"], n1, n1s))
    qkv, _, qkvs, _, shifts["qkv"] = linear_dynamic(
        n1, a(p + ".attn.qkv.weight"), a(p + ".attn.qkv.bias"),
        n1s, scale_of(ref["qkv"]))
    rows.append(metric("qkv", ref["qkv"], qkv, qkvs))

    q, k, v = np.split(qkv.astype(np.int64) - 128, 3, axis=1)
    scores = np.empty((G.HEAD_NUM, G.TOKEN_NUM, G.TOKEN_NUM), dtype=np.int64)
    for h in range(G.HEAD_NUM):
        sl = slice(h * G.HEAD_DIM, (h + 1) * G.HEAD_DIM)
        scores[h] = q[:, sl] @ k[:, sl].T
    qk_shift = max(0, min(63, int(np.rint(-math.log2(qkvs)))))
    old_q, old_k = G.SOFTMAX_Q_SHIFT, G.SOFTMAX_K_SHIFT
    G.SOFTMAX_Q_SHIFT = G.SOFTMAX_K_SHIFT = qk_shift
    prob = G.softmax_hw(scores, exp_lut)[:, :, :G.TOKEN_NUM]
    G.SOFTMAX_Q_SHIFT, G.SOFTMAX_K_SHIFT = old_q, old_k
    pref = ref["prob"]
    preal = prob.astype(np.float64) / 128.0
    pd = preal - pref
    rows.append({"layer": "softmax", "shape": list(pref.shape), "scale": 1/128,
                 "mae": float(np.mean(np.abs(pd))), "rmse": float(np.sqrt(np.mean(pd * pd))),
                 "max_abs_error": float(np.max(np.abs(pd))), "cosine": cosine(pref, preal),
                 "saturation": float(np.mean((prob == 0) | (prob == 127)))})
    heatmap = None
    if image is not None and heatmap_dir is not None:
        heatmap = save_attention_heatmaps(
            image, pref, prob, heatmap_dir, f"block{block:02d}", save_heads=save_heads)

    target_sv = scale_of(ref["attn_value"])
    heads = []
    svs = 0.0
    for h in range(G.HEAD_NUM):
        sl = slice(h * G.HEAD_DIM, (h + 1) * G.HEAD_DIM)
        vh = v[:, sl].T
        sv, _, svs, _, shifts["attention_value"] = linear_dynamic(
            prob[h], vh, None, 1/128, target_sv, zp=0, known_weight_scale=qkvs)
        heads.append(sv)
    attn = np.concatenate(heads, axis=1)
    rows.append(metric("attention_value", ref["attn_value"], attn, svs))

    proj, _, projs, _, shifts["projection"] = linear_dynamic(
        attn, a(p + ".attn.proj.weight"), a(p + ".attn.proj.bias"),
        svs, scale_of(ref["attn_residual"]))
    mid = G.residual_add(proj, rescale(xq, xscale, projs))
    rows.append(metric("projection", ref["projection"], proj, projs))
    rows.append(metric("attention_residual", ref["attn_residual"], mid, projs))

    n2, n2s, shifts["norm2_left"] = rms_dynamic(
        mid, ref["attn_residual"], a(p + ".norm2.weight"), projs, ref["norm2"])
    rows.append(metric("norm2", ref["norm2"], n2, n2s))
    _, psum, gelus, _, shifts["fc1"] = linear_dynamic(
        n2, a(p + ".mlp.fc1.weight"), a(p + ".mlp.fc1.bias"),
        n2s, scale_of(ref["gelu"]))
    idx = np.right_shift(psum & 0xFFFFFFFF, 8) & 0xFF
    gelu = G.requant_zp128(gelu_rom[idx], shifts["fc1"])
    rows.append(metric("gelu", ref["gelu"], gelu, gelus))

    fc2, _, outs, _, shifts["fc2"] = linear_dynamic(
        gelu, a(p + ".mlp.fc2.weight"), a(p + ".mlp.fc2.bias"),
        gelus, scale_of(ref["output"]))
    out = G.residual_add(fc2, rescale(mid, projs, outs))
    rows.append(metric("fc2", ref["fc2"], fc2, outs))
    rows.append(metric("output", ref["output"], out, outs))
    return out, outs, {"block": block, "input_scale": xscale, "output_scale": outs,
                       "output_cosine": rows[-1]["cosine"], "output_mae": rows[-1]["mae"],
                       "output_rmse": rows[-1]["rmse"], "output_abse": rows[-1]["max_abs_error"],
                       "output_saturation": rows[-1]["saturation"], "shifts": shifts,
                       "layers": rows, "attention_heatmap": heatmap}


def run_image(model: torch.nn.Module, state: dict[str, torch.Tensor], image: Path,
              exp: np.ndarray, gelu: np.ndarray,
              heatmap_dir: Path | None = None,
              save_heads: bool = False) -> dict[str, Any]:
    refs, pt_logits = capture_all(model, image)
    xscale = scale_of(refs[0]["input"])
    xq = quant_u8(refs[0]["input"], xscale)
    blocks = []
    for block in range(12):
        xq, xscale, result = run_block(
            state, block, refs[block], xq, xscale, exp, gelu,
            image=image, heatmap_dir=heatmap_dir, save_heads=save_heads)
        blocks.append(result)
    real = dequant(xq, xscale)
    gamma = G.t(state, "norm.weight").astype(np.float64)
    norm = real / np.sqrt(np.mean(real * real, axis=-1, keepdims=True) + 1e-6) * gamma
    hw_logits = norm[0] @ G.t(state, "head.weight").astype(np.float64).T + G.t(state, "head.bias")
    return {"image": str(image), "pt_top1": int(np.argmax(pt_logits)),
            "dlau_top1": int(np.argmax(hw_logits)),
            "top1_match": bool(np.argmax(pt_logits) == np.argmax(hw_logits)),
            "logit_cosine": cosine(pt_logits, hw_logits), "blocks": blocks}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PT)
    ap.add_argument("--images", type=Path, nargs="*", default=sorted((IMG_DIR).glob("*.jpg")))
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--no-heatmaps", action="store_true",
                    help="disable per-block CLS-attention heatmap PNG output")
    ap.add_argument("--save-head-heatmaps", action="store_true",
                    help="also dump each individual attention head for every block")
    args = ap.parse_args()
    model, state = make_reference(args.checkpoint)
    exp = G.load_hex_u16(G.EXP_LUT_HEX)
    gelu = G.load_gelu_rom(G.GELU_SV)
    heatmap_dir = None if args.no_heatmaps else args.out / "heatmaps" / "12blocks"
    samples = [
        run_image(model, state, image, exp, gelu, heatmap_dir, args.save_head_heatmaps)
        for image in args.images
    ]
    summary = {"scope": "cumulative blocks 0..11; exact PT block0 input",
               "scale_policy": "per-block/per-stage PT activation dynamic calibration",
               "images": len(samples),
               "top1_agreement": sum(s["top1_match"] for s in samples) / len(samples),
               "mean_logit_cosine": float(np.mean([s["logit_cosine"] for s in samples])),
               "mean_block11_output_cosine": float(np.mean([s["blocks"][-1]["output_cosine"] for s in samples]))}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "12blocks.json").write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n", encoding="utf-8")
    lines = ["# 12-block dynamic-scale comparison", "",
             f"- Images: {summary['images']}", f"- Top-1 agreement: {summary['top1_agreement']:.2%}",
             f"- Mean block11 output cosine: {summary['mean_block11_output_cosine']:.6f}",
             f"- Mean logit cosine: {summary['mean_logit_cosine']:.6f}", "",
             "| image | PT | DLAU | match | block11 cosine | logit cosine |",
             "|---|---:|---:|:---:|---:|---:|"]
    lines += [f"| {Path(s['image']).name} | {s['pt_top1']} | {s['dlau_top1']} | {'Y' if s['top1_match'] else 'N'} | {s['blocks'][-1]['output_cosine']:.6f} | {s['logit_cosine']:.6f} |" for s in samples]
    if not args.no_heatmaps:
        lines += ["", "## Attention heatmaps", "",
                  f"- Directory: `{(args.out / 'heatmaps' / '12blocks').as_posix()}`",
                  "- Maps are CLS-token attention over the 14x14 patch grid, averaged over 6 heads.",
                  "- Files are grouped by image and block: `block00` ... `block11`.",
                  "- Files per block: `pt_cls_attention.png`, `dlau_cls_attention.png`, "
                  "`abs_diff_attention.png`, `pt_overlay.png`, `dlau_overlay.png`."]
    lines += ["", "## Mean output error by block", "", "| block | cosine | MAE | RMSE | Max abs error|", "|---|---:|---:|---:|---:|"]
    block_labels: list[str] = []
    block_cosine: list[float] = []
    block_mae: list[float] = []
    block_rmse: list[float] = []
    for block in range(12):
        mean_cos = float(np.mean([s["blocks"][block]["output_cosine"] for s in samples]))
        mean_mae = float(np.mean([s["blocks"][block]["output_mae"] for s in samples]))
        mean_rmse = float(np.mean([s["blocks"][block]["output_rmse"] for s in samples]))
        mean_abse = float(np.mean([s["blocks"][block]["output_abse"] for s in samples]))
        block_labels.append(f"block{block}")
        block_cosine.append(mean_cos)
        block_mae.append(mean_mae)
        block_rmse.append(mean_rmse)
        lines.append(f"| {block} | {mean_cos:.6f} | {mean_mae:.6f} | {mean_rmse:.6f} | {mean_abse:.6f} |")
    plot_files = save_metric_line_plots(
        args.out, "12blocks_output", block_labels, block_cosine, block_mae, block_rmse)
    lines += ["", "## Metric line plots", ""]
    lines += [f"- Cosine: `{Path(plot_files['cosine']).as_posix()}`",
              f"- MAE: `{Path(plot_files['mae']).as_posix()}`",
              f"- RMSE: `{Path(plot_files['rmse']).as_posix()}`"]
    (args.out / "12blocks.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
