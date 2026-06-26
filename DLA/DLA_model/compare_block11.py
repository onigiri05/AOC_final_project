"""Block-11 comparison using PT-calibrated dynamic activation scales.

This keeps the integer operators/LUTs used by golden_final/golden_gen.py, but
replaces its fixed scale/shift policy with per-stage scales measured from the
corresponding PyTorch inference activation.  Block 11 starts from the exact
PyTorch block input, so errors from blocks 0..10 are intentionally excluded.
"""
from __future__ import annotations

import importlib.util
import argparse
import json
import math
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

HERE = Path(__file__).resolve().parent
PT_CANDIDATES = [
    (HERE / "../../PT_DIR/rms_qat_best.pt").resolve(),
    (HERE.parent / "golden_gen" / "rms_qat_best.pt").resolve(),
]
PT = next((path for path in PT_CANDIDATES if path.exists()), PT_CANDIDATES[0])
IMG_DIR = HERE/"Image/"

def load_algorithm():
    candidates = [
        HERE / "DLA_model.py",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])
    if not source.exists():
        raise FileNotFoundError(
            "DLA_model.py is missing near this script and in ../golden_final; "
            "no matching cached module exists"
        )
    spec = importlib.util.spec_from_file_location("dlau_golden", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DLA_model/DLA_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
G = load_algorithm()


def patch_algorithm_paths() -> None:
    """Keep other_path scripts runnable from any cwd without editing DLA_model.py."""
    exp_candidates = [
        getattr(G, "EXP_LUT_HEX", None),
        HERE.parent / "DLA" / "src" / "LUT" / "exp_lut_10bit_Q1_15_range12.hex",
    ]
    gelu_candidates = [
        getattr(G, "GELU_SV", None),
        HERE.parent / "DLA" / "src" / "PPU" / "GELU_Unit.sv",
    ]
    exp = next((Path(p) for p in exp_candidates if p is not None and Path(p).exists()), None)
    gelu = next((Path(p) for p in gelu_candidates if p is not None and Path(p).exists()), None)
    if exp is not None:
        G.EXP_LUT_HEX = exp
    if gelu is not None:
        G.GELU_SV = gelu


patch_algorithm_paths()

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight

class QuantLinear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = (self.weight.abs().max() / 127.0).detach().clamp_min(1e-9)
        w = torch.clamp(torch.round(self.weight / scale), -127, 127) * scale
        return F.linear(x, w, self.bias)

def make_reference(checkpoint: Path):
    model = timm.create_model("vit_small_patch16_224", pretrained=False, num_classes=1000)
    # Replace LayerNorm first, then load QAT state; replace Linear afterwards.
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(parent, name, RMSNorm(child.normalized_shape[0], child.eps))
    state = G.load_state(checkpoint)
    model.load_state_dict(state, strict=True)
    head = model.head
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child is not head:
                new = QuantLinear(child.in_features, child.out_features, child.bias is not None)
                new.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new.bias.data.copy_(child.bias.data)
                setattr(parent, name, new)
    return model.eval(), state

def preprocess(path: Path) -> torch.Tensor:
    arr = G.load_image_u8(path).astype(np.float32) / 255.0
    arr = (arr - G.IMAGENET_MEAN.astype(np.float32)) / G.IMAGENET_STD.astype(np.float32)
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0 if na == 0.0 and nb == 0.0 else 0.0
    return float(np.dot(a.reshape(-1), b.reshape(-1)) / (na * nb))

def scale_of(x: np.ndarray) -> float:
    return max(float(np.max(np.abs(x))) / 127.0, 1e-12)


def quant_u8(x: np.ndarray, scale: float) -> np.ndarray:
    return G.clamp_u8(np.rint(np.asarray(x) / scale).astype(np.int64) + 128)


def dequant(q: np.ndarray, scale: float) -> np.ndarray:
    return (q.astype(np.int64) - 128).astype(np.float64) * scale


def weight_q(w: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(float(np.max(np.abs(w))) / 127.0, 1e-12)
    return np.clip(np.rint(w / scale), -127, 127).astype(np.int64), scale


def choose_rshift(acc_scale: float, target_scale: float) -> int:
    return max(0, min(63, int(np.rint(math.log2(target_scale / acc_scale)))))


def linear_dynamic(xq: np.ndarray, w_out_in: np.ndarray, bias: np.ndarray | None,
                   input_scale: float, target_scale: float, zp: int = 128,
                   known_weight_scale: float | None = None) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    if known_weight_scale is None:
        wq, ws = weight_q(w_out_in)
    else:
        wq = np.asarray(w_out_in, dtype=np.int64)
        ws = float(known_weight_scale)
    acc_scale = input_scale * ws
    bq = np.zeros(wq.shape[0], dtype=np.int64) if bias is None else np.rint(bias / acc_scale).astype(np.int64)
    psum = (xq.astype(np.int64) - int(zp)) @ wq.T + bq
    shift = choose_rshift(acc_scale, target_scale)
    actual_scale = acc_scale * (2.0 ** shift)
    return G.requant_zp128(psum, shift), psum, actual_scale, ws, shift


def rms_dynamic(xq: np.ndarray, pt_input: np.ndarray, gamma: np.ndarray,
                input_scale: float, target: np.ndarray) -> tuple[np.ndarray, float, int]:
    gamma_q = np.clip(np.rint(gamma * (1 << 14)), -32768, 32767).astype(np.int64)
    rms = np.sqrt(np.mean(pt_input.astype(np.float64) ** 2, axis=-1) + 1e-6)
    inv = np.clip(np.rint(input_scale / rms * (1 << 14)), 0, 0xFFFF).astype(np.int64)
    target_scale = scale_of(target)
    left_shift = max(0, min(27, int(np.rint(-math.log2(target_scale)))))
    actual_scale = 2.0 ** (-left_shift)
    product = ((xq.astype(np.int64) - 128) * inv[:, None] * gamma_q[None, :])
    y = np.right_shift(product, 28 - left_shift)
    return G.clamp_u8(G.clamp_s8(y) + 128), actual_scale, left_shift


def rescale(q: np.ndarray, src: float, dst: float) -> np.ndarray:
    return quant_u8(dequant(q, src), dst)


def metric(name: str, ref: np.ndarray, q: np.ndarray, scale: float) -> dict[str, Any]:
    real = dequant(q, scale)
    delta = real - ref.astype(np.float64)
    return {"layer": name, "shape": list(ref.shape), "scale": scale,
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_error": float(np.max(np.abs(delta))),
            "cosine": cosine(ref.astype(np.float64), real),
            "saturation": float(np.mean((q == 0) | (q == 255)))}


def real_metric(name: str, ref: np.ndarray, real: np.ndarray, scale: float | None) -> dict[str, Any]:
    delta = real.astype(np.float64) - ref.astype(np.float64)
    return {"layer": name, "shape": list(ref.shape), "scale": scale,
            "mae": float(np.mean(np.abs(delta))),
            "rmse": float(np.sqrt(np.mean(delta * delta))),
            "max_abs_error": float(np.max(np.abs(delta))),
            "cosine": cosine(ref.astype(np.float64), real.astype(np.float64)),
            "saturation": None}


def _safe_stem(path: Path) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in path.stem)


def cls_attention_map(prob: np.ndarray) -> np.ndarray:
    """Return per-head CLS-to-patch attention maps as [head, 14, 14]."""
    arr = np.asarray(prob, dtype=np.float64)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 128.0
    patches = arr[:, 0, 1:G.TOKEN_NUM]
    grid = int(round(math.sqrt(G.TOKEN_NUM - 1)))
    if grid * grid != G.TOKEN_NUM - 1:
        raise ValueError(f"TOKEN_NUM={G.TOKEN_NUM} cannot form a square patch grid")
    return patches.reshape(arr.shape[0], grid, grid)


def _colorize_heatmap(hm: np.ndarray) -> np.ndarray:
    x = np.asarray(hm, dtype=np.float64)
    x = x - float(np.min(x))
    denom = float(np.max(x))
    if denom > 0:
        x = x / denom
    r = np.clip(255.0 * x, 0, 255)
    g = np.clip(255.0 * np.maximum(0.0, 1.0 - np.abs(x - 0.55) / 0.55), 0, 255)
    b = np.clip(255.0 * (1.0 - x), 0, 255)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def _resize_map(hm: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    img = Image.fromarray(_colorize_heatmap(hm), mode="RGB")
    resample = getattr(Image.Resampling, "BICUBIC", Image.BICUBIC)
    return np.asarray(img.resize(size, resample=resample), dtype=np.uint8)


def _resize_nearest(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    resample = getattr(Image.Resampling, "NEAREST", Image.NEAREST)
    return img.resize(size, resample=resample)


def _overlay(image: Path, hm: np.ndarray) -> Image.Image:
    base = Image.fromarray(G.load_image_u8(image).astype(np.uint8), mode="RGB")
    heat = _resize_map(hm, base.size)
    mixed = (0.55 * np.asarray(base, dtype=np.float64) + 0.45 * heat).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mixed, mode="RGB")


def attention_focus_summary(prob: np.ndarray, topk: int = 5) -> dict[str, Any]:
    maps = cls_attention_map(prob)
    mean_map = np.mean(maps, axis=0)
    flat_order = np.argsort(mean_map.reshape(-1))[::-1][:topk]
    grid = mean_map.shape[0]
    return {
        "head_count": int(maps.shape[0]),
        "grid": [int(grid), int(grid)],
        "top_patches": [
            {
                "rank": int(i + 1),
                "row": int(idx // grid),
                "col": int(idx % grid),
                "score": float(mean_map.reshape(-1)[idx]),
            }
            for i, idx in enumerate(flat_order)
        ],
        "entropy": float(-np.sum(mean_map * np.log(np.maximum(mean_map, 1e-12)))),
        "max_score": float(np.max(mean_map)),
    }


def save_attention_heatmaps(image: Path, pt_prob: np.ndarray, dlau_prob: np.ndarray,
                            outdir: Path, tag: str, save_heads: bool = False) -> dict[str, Any]:
    """Write PT/DLAU CLS-attention heatmaps and return focus statistics."""
    target = outdir / _safe_stem(image) / tag
    target.mkdir(parents=True, exist_ok=True)
    pt_maps = cls_attention_map(pt_prob)
    dlau_maps = cls_attention_map(dlau_prob)
    pt_mean = np.mean(pt_maps, axis=0)
    dlau_mean = np.mean(dlau_maps, axis=0)
    diff = np.abs(dlau_mean - pt_mean)

    _resize_nearest(Image.fromarray(_colorize_heatmap(pt_mean), mode="RGB"), (224, 224)).save(
        target / "pt_cls_attention.png")
    _resize_nearest(Image.fromarray(_colorize_heatmap(dlau_mean), mode="RGB"), (224, 224)).save(
        target / "dlau_cls_attention.png")
    _resize_nearest(Image.fromarray(_colorize_heatmap(diff), mode="RGB"), (224, 224)).save(
        target / "abs_diff_attention.png")
    _overlay(image, pt_mean).save(target / "pt_overlay.png")
    _overlay(image, dlau_mean).save(target / "dlau_overlay.png")
    if save_heads:
        head_dir = target / "heads"
        head_dir.mkdir(exist_ok=True)
        for h in range(pt_maps.shape[0]):
            _resize_nearest(Image.fromarray(_colorize_heatmap(pt_maps[h]), mode="RGB"), (224, 224)).save(
                head_dir / f"pt_head{h}.png")
            _resize_nearest(Image.fromarray(_colorize_heatmap(dlau_maps[h]), mode="RGB"), (224, 224)).save(
                head_dir / f"dlau_head{h}.png")
    return {
        "directory": str(target),
        "pt": attention_focus_summary(pt_prob),
        "dlau": attention_focus_summary(dlau_prob),
        "mean_abs_diff": float(np.mean(diff)),
        "cosine": cosine(pt_mean, dlau_mean),
    }


def save_line_plot(labels: list[str], values: list[float], ylabel: str,
                   title: str, path: Path, y_min: float | None = None,
                   y_max: float | None = None) -> None:
    """Write a compact line plot PNG using only Pillow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 960, 420
    left, right, top, bottom = 78, 28, 34, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    vals = [float(v) for v in values]
    if not vals:
        img.save(path)
        return
    lo = min(vals) if y_min is None else float(y_min)
    hi = max(vals) if y_max is None else float(y_max)
    if math.isclose(lo, hi):
        pad = 1.0 if hi == 0 else abs(hi) * 0.1
        lo -= pad
        hi += pad
    else:
        pad = (hi - lo) * 0.08
        lo = lo - pad if y_min is None else lo
        hi = hi + pad if y_max is None else hi

    def xy(i: int, v: float) -> tuple[int, int]:
        x = left + int(round(i * plot_w / max(1, len(vals) - 1)))
        y = top + int(round((hi - v) * plot_h / (hi - lo)))
        return x, y

    # Frame and grid.
    draw.rectangle([left, top, left + plot_w, top + plot_h], outline=(80, 80, 80))
    for t in range(6):
        y = top + int(round(t * plot_h / 5))
        val = hi - t * (hi - lo) / 5
        draw.line([left, y, left + plot_w, y], fill=(225, 225, 225))
        draw.text((8, y - 7), f"{val:.4g}", fill=(40, 40, 40))

    points = [xy(i, v) for i, v in enumerate(vals)]
    if len(points) == 1:
        x, y = points[0]
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(0, 96, 180))
    else:
        draw.line(points, fill=(0, 96, 180), width=3)
        for x, y in points:
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(0, 96, 180))

    # X labels: draw all for <=16 points, otherwise sparse.
    step = 1 if len(labels) <= 16 else max(1, int(math.ceil(len(labels) / 12)))
    for i, label in enumerate(labels):
        if i % step != 0 and i != len(labels) - 1:
            continue
        x, _ = xy(i, vals[i])
        draw.line([x, top + plot_h, x, top + plot_h + 5], fill=(80, 80, 80))
        text = str(label)
        if len(text) > 14:
            text = text[:13] + "…"
        draw.text((x - 28, top + plot_h + 10), text, fill=(40, 40, 40))

    draw.text((left, 8), title, fill=(20, 20, 20))
    draw.text((8, 10), ylabel, fill=(20, 20, 20))
    img.save(path)


def save_metric_line_plots(outdir: Path, prefix: str, labels: list[str],
                           cosine_values: list[float], mae_values: list[float],
                           rmse_values: list[float]) -> dict[str, str]:
    plot_dir = outdir / "plots"
    files = {
        "cosine": plot_dir / f"{prefix}_cosine.png",
        "mae": plot_dir / f"{prefix}_mae.png",
        "rmse": plot_dir / f"{prefix}_rmse.png",
    }
    save_line_plot(labels, cosine_values, "cosine", f"{prefix} cosine", files["cosine"], 0.0, 1.0)
    save_line_plot(labels, mae_values, "MAE", f"{prefix} MAE", files["mae"])
    save_line_plot(labels, rmse_values, "RMSE", f"{prefix} RMSE", files["rmse"])
    return {name: str(path) for name, path in files.items()}


def capture(model: torch.nn.Module, image: Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    block = model.blocks[11]
    if hasattr(block.attn, "fused_attn"):
        block.attn.fused_attn = False
    values: dict[str, np.ndarray] = {}

    def save_out(name: str):
        return lambda _m, _i, out: values.__setitem__(name, out.detach().cpu().numpy()[0])

    def save_in(name: str):
        return lambda _m, inp: values.__setitem__(name, inp[0].detach().cpu().numpy()[0])

    hooks = [block.register_forward_pre_hook(save_in("input")),
             block.norm1.register_forward_hook(save_out("norm1")),
             block.attn.qkv.register_forward_hook(save_out("qkv")),
             block.attn.attn_drop.register_forward_hook(save_out("prob")),
             block.attn.proj.register_forward_pre_hook(save_in("attn_value")),
             block.attn.proj.register_forward_hook(save_out("projection")),
             block.norm2.register_forward_pre_hook(save_in("attn_residual")),
             block.norm2.register_forward_hook(save_out("norm2")),
             block.mlp.fc1.register_forward_hook(save_out("fc1_linear")),
             block.mlp.act.register_forward_hook(save_out("gelu")),
             block.mlp.fc2.register_forward_hook(save_out("fc2")),
             block.register_forward_hook(save_out("output")),
             model.norm.register_forward_hook(save_out("final_norm"))]
    with torch.no_grad():
        logits = model(preprocess(image))[0].detach().cpu().numpy().astype(np.float64)
    for hook in hooks:
        hook.remove()
    return values, logits


def run_one(model: torch.nn.Module, state: dict[str, torch.Tensor], image: Path,
            exp_lut: np.ndarray, gelu_rom: np.ndarray,
            heatmap_dir: Path | None = None,
            save_heads: bool = False) -> dict[str, Any]:
    pt, pt_logits = capture(model, image)
    a = lambda key: G.t(state, key).astype(np.float64)
    p = "blocks.11"
    rows: list[dict[str, Any]] = []
    shifts: dict[str, int] = {}

    xscale = scale_of(pt["input"])
    xq = quant_u8(pt["input"], xscale)
    rows.append(metric("00_input", pt["input"], xq, xscale))

    n1, n1s, shifts["norm1_left"] = rms_dynamic(
        xq, pt["input"], a(p + ".norm1.weight"), xscale, pt["norm1"])
    rows.append(metric("01_norm1", pt["norm1"], n1, n1s))

    qkv, _, qkvs, _, shifts["qkv"] = linear_dynamic(
        n1, a(p + ".attn.qkv.weight"), a(p + ".attn.qkv.bias"),
        n1s, scale_of(pt["qkv"]))
    rows.append(metric("02_qkv", pt["qkv"], qkv, qkvs))
    q, k, v = np.split(qkv.astype(np.int64) - 128, 3, axis=1)
    scores = np.empty((G.HEAD_NUM, G.TOKEN_NUM, G.TOKEN_NUM), dtype=np.int64)
    for h in range(G.HEAD_NUM):
        sl = slice(h * G.HEAD_DIM, (h + 1) * G.HEAD_DIM)
        scores[h] = q[:, sl] @ k[:, sl].T
    old_q, old_k = G.SOFTMAX_Q_SHIFT, G.SOFTMAX_K_SHIFT
    G.SOFTMAX_Q_SHIFT = G.SOFTMAX_K_SHIFT = max(0, min(63, int(np.rint(-math.log2(qkvs)))))
    prob = G.softmax_hw(scores, exp_lut)[:, :, :G.TOKEN_NUM]
    G.SOFTMAX_Q_SHIFT, G.SOFTMAX_K_SHIFT = old_q, old_k
    prob_ref = pt["prob"]
    prob_delta = prob.astype(np.float64) / 128.0 - prob_ref
    rows.append({"layer": "03_softmax", "shape": list(prob_ref.shape), "scale": 1/128,
                 "mae": float(np.mean(np.abs(prob_delta))),
                 "rmse": float(np.sqrt(np.mean(prob_delta ** 2))),
                 "max_abs_error": float(np.max(np.abs(prob_delta))),
                 "cosine": cosine(prob_ref, prob.astype(np.float64) / 128.0),
                 "saturation": float(np.mean((prob == 0) | (prob == 127)))})
    heatmap = None
    if heatmap_dir is not None:
        heatmap = save_attention_heatmaps(
            image, prob_ref, prob, heatmap_dir, "block11", save_heads=save_heads)

    attn_parts = []
    sv_target = scale_of(pt["attn_value"])
    for h in range(G.HEAD_NUM):
        sl = slice(h * G.HEAD_DIM, (h + 1) * G.HEAD_DIM)
        vh = v[:, sl].T
        out, _, svs, _, svshift = linear_dynamic(
            prob[h], vh, None, 1/128, sv_target, zp=0, known_weight_scale=qkvs)
        attn_parts.append((out, svs, svshift))
    # All heads use the same scales and therefore the same shift.
    attn = np.concatenate([v_[0] for v_ in attn_parts], axis=1)
    attns = attn_parts[0][1]
    shifts["attention_value"] = attn_parts[0][2]
    rows.append(metric("04_attention_value", pt["attn_value"], attn, attns))

    proj, _, projs, _, shifts["projection"] = linear_dynamic(
        attn, a(p + ".attn.proj.weight"), a(p + ".attn.proj.bias"),
        attns, scale_of(pt["attn_residual"]))
    x_for_mid = rescale(xq, xscale, projs)
    mid = G.residual_add(proj, x_for_mid)
    rows.append(metric("05_projection", pt["projection"], proj, projs))
    rows.append(metric("06_attention_residual", pt["attn_residual"], mid, projs))

    n2, n2s, shifts["norm2_left"] = rms_dynamic(
        mid, pt["attn_residual"], a(p + ".norm2.weight"), projs, pt["norm2"])
    rows.append(metric("07_norm2", pt["norm2"], n2, n2s))

    _, fc1psum, fc1s, fc1ws, shifts["fc1"] = linear_dynamic(
        n2, a(p + ".mlp.fc1.weight"), a(p + ".mlp.fc1.bias"),
        n2s, scale_of(pt["gelu"]))
    rows.append(real_metric("08_fc1_linear", pt["fc1_linear"],
                            fc1psum.astype(np.float64) * n2s * fc1ws,
                            n2s * fc1ws))
    idx = np.right_shift(fc1psum & 0xFFFFFFFF, 8) & 0xFF
    gelu = G.requant_zp128(gelu_rom[idx], shifts["fc1"])
    rows.append(metric("09_gelu", pt["gelu"], gelu, fc1s))

    fc2, _, fc2s, _, shifts["fc2"] = linear_dynamic(
        gelu, a(p + ".mlp.fc2.weight"), a(p + ".mlp.fc2.bias"),
        fc1s, scale_of(pt["output"]))
    mid_for_out = rescale(mid, projs, fc2s)
    out = G.residual_add(fc2, mid_for_out)
    rows.append(metric("10_fc2", pt["fc2"], fc2, fc2s))
    block_output_metric = metric("11_block11_output", pt["output"], out, fc2s)
    rows.append(block_output_metric)

    out_real = dequant(out, fc2s)
    gamma = a("norm.weight")
    final_norm = out_real / np.sqrt(np.mean(out_real ** 2, axis=-1, keepdims=True) + 1e-6) * gamma
    rows.append(real_metric("12_final_norm", pt["final_norm"], final_norm, None))
    hw_logits = final_norm[0] @ a("head.weight").T + a("head.bias")
    rows.append(real_metric("13_classifier_logits", pt_logits, hw_logits, None))
    return {"image": str(image), "pt_top1": int(np.argmax(pt_logits)),
            "dlau_top1": int(np.argmax(hw_logits)),
            "top1_match": bool(np.argmax(pt_logits) == np.argmax(hw_logits)),
            "logit_cosine": cosine(pt_logits, hw_logits),
            "block11_cosine": block_output_metric["cosine"], "shifts": shifts,
            "scales": {r["layer"]: r["scale"] for r in rows}, "layers": rows,
            "attention_heatmap": heatmap}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, default=PT)
    ap.add_argument("--images", type=Path, nargs="*",
                    default=sorted((IMG_DIR).glob("*.jpg")))
    ap.add_argument("--out", type=Path, default=HERE / "results")
    ap.add_argument("--no-heatmaps", action="store_true",
                    help="disable CLS-attention heatmap PNG output")
    ap.add_argument("--save-head-heatmaps", action="store_true",
                    help="also dump each individual attention head")
    args = ap.parse_args()
    model, state = make_reference(args.checkpoint)
    exp = G.load_hex_u16(G.EXP_LUT_HEX)
    gelu = G.load_gelu_rom(G.GELU_SV)
    heatmap_dir = None if args.no_heatmaps else args.out / "heatmaps" / "block11"
    samples = [
        run_one(model, state, image, exp, gelu, heatmap_dir, args.save_head_heatmaps)
        for image in args.images
    ]
    summary = {"scope": "block11 only; exact PT block11 input",
               "scale_policy": "per-stage PT activation dynamic calibration",
               "images": len(samples),
               "top1_agreement": sum(s["top1_match"] for s in samples) / len(samples),
               "mean_logit_cosine": float(np.mean([s["logit_cosine"] for s in samples])),
               "mean_block11_cosine": float(np.mean([s["block11_cosine"] for s in samples]))}
    args.out.mkdir(parents=True, exist_ok=True)
    stage_labels: list[str] = []
    stage_cosine: list[float] = []
    stage_mae: list[float] = []
    stage_rmse: list[float] = []
    (args.out / "block11.json").write_text(
        json.dumps({"summary": summary, "samples": samples}, indent=2) + "\n", encoding="utf-8")
    lines = ["# Block11 dynamic-scale comparison", "",
             f"- Images: {summary['images']}", f"- Top-1 agreement: {summary['top1_agreement']:.2%}",
             f"- Mean block11 cosine: {summary['mean_block11_cosine']:.6f}",
             f"- Mean logit cosine: {summary['mean_logit_cosine']:.6f}", "",
             "| image | PT | DLAU | match | block11 cosine | logit cosine |",
             "|---|---:|---:|:---:|---:|---:|"]
    lines += [f"| {Path(s['image']).name} | {s['pt_top1']} | {s['dlau_top1']} | {'Y' if s['top1_match'] else 'N'} | {s['block11_cosine']:.6f} | {s['logit_cosine']:.6f} |" for s in samples]
    if not args.no_heatmaps:
        lines += ["", "## Attention heatmaps", "",
                  f"- Directory: `{(args.out / 'heatmaps' / 'block11').as_posix()}`",
                  "- Maps are CLS-token attention over the 14x14 patch grid, averaged over 6 heads.",
                  "- Files per image: `pt_cls_attention.png`, `dlau_cls_attention.png`, "
                  "`abs_diff_attention.png`, `pt_overlay.png`, `dlau_overlay.png`."]
    lines += ["", "## Mean error by stage", "",
              "| stage | cosine | MAE | RMSE |", "|---|---:|---:|---:|"]
    for layer in [row["layer"] for row in samples[0]["layers"]]:
        values = [next(row for row in sample["layers"] if row["layer"] == layer)
                  for sample in samples]
        mean_cos = float(np.mean([row["cosine"] for row in values]))
        mean_mae = float(np.mean([row["mae"] for row in values]))
        mean_rmse = float(np.mean([row["rmse"] for row in values]))
        stage_labels.append(layer)
        stage_cosine.append(mean_cos)
        stage_mae.append(mean_mae)
        stage_rmse.append(mean_rmse)
        lines.append(
            f"| {layer} | {mean_cos:.6f} "
            f"| {mean_mae:.6f} "
            f"| {mean_rmse:.6f} |"
        )
    plot_files = save_metric_line_plots(
        args.out, "block11_stage", stage_labels, stage_cosine, stage_mae, stage_rmse)
    lines += ["", "## Metric line plots", ""]
    lines += [f"- Cosine: `{Path(plot_files['cosine']).as_posix()}`",
              f"- MAE: `{Path(plot_files['mae']).as_posix()}`",
              f"- RMSE: `{Path(plot_files['rmse']).as_posix()}`"]
    (args.out / "block11.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
