#!/usr/bin/env python3
"""Generate real-image, real-checkpoint vectors for the current one-block RTL ViT design.

This generator uses the real QAT checkpoint tensors and the real input image.
It emits the same file names consumed by vit.ipynb / ViT_System_Core:
image, pos, cls, gamma, patch/transformer weight tiles, bias tiles, and x_out.

Scope: the RTL currently instantiates one ViT block, so this script exports one
selected block (default block 0), not all 12 timm blocks.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

IMG_H = 224
IMG_W = 224
IMG_C = 3
PATCH_SIZE = 16
EMBED_DIM = 384
HEAD_NUM = 6
HEAD_DIM = 64
FFN_CHANNEL_NUM = 1536
SOFTMAX_COLS = 208
TOKEN_TILE = 8
CHANNEL_TILE = 8
PATCH_COUNT = (IMG_H // PATCH_SIZE) * (IMG_W // PATCH_SIZE)
TOKEN_NUM = PATCH_COUNT + 1
PATCH_ELEMS = PATCH_SIZE * PATCH_SIZE * IMG_C
XOUT_ELEMS = TOKEN_NUM * EMBED_DIM

CHANNEL_TILE_NUM = EMBED_DIM // CHANNEL_TILE
QKV_CHANNEL_TILE_NUM = CHANNEL_TILE_NUM * 3
HEAD_DIM_TILE_NUM = HEAD_DIM // CHANNEL_TILE
SCORE_TILE_NUM = SOFTMAX_COLS // CHANNEL_TILE
FFN_CHANNEL_TILE_NUM = FFN_CHANNEL_NUM // CHANNEL_TILE
PATCH_K_TILE_NUM = (PATCH_ELEMS + CHANNEL_TILE - 1) // CHANNEL_TILE
TILE_ROW_WORDS = CHANNEL_TILE // 4
WEIGHT_TILE_WORDS = CHANNEL_TILE * TILE_ROW_WORDS
BIAS_TILE_WORDS = CHANNEL_TILE

W_QKV_BASE = 0x00000
B_QKV_BASE = 0x00000
W_OUT_BASE = 0x02000
B_OUT_BASE = 0x00100
W_FC1_BASE = 0x04000
B_FC1_BASE = 0x00200
W_FC2_BASE = 0x08000
B_FC2_BASE = 0x00300

SHIFT_QKV = 2
SHIFT_ATTN_V = 2
SHIFT_OUT_PROJ = 1
SHIFT_FC1 = 0
SHIFT_FC2 = 1
SOFTMAX_Q_SHIFT = 4
SOFTMAX_K_SHIFT = 4

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)

DEFAULT_HARDWARE_ROOT = Path(r"C:\成大貓\大四\AOC\炸彈惡魔\AOC_final_project-main\hardware")


def find_hardware_root() -> Path:
    candidates = []
    env_root = None
    try:
        import os
        env_root = os.environ.get("VIT_HARDWARE_ROOT")
    except Exception:
        env_root = None
    if env_root:
        candidates.append(Path(env_root))
    here = Path(__file__).resolve()
    candidates.extend(here.parents)
    candidates.append(DEFAULT_HARDWARE_ROOT)
    for cand in candidates:
        if (cand / "Streaming_RMSNorm_Unit").exists() and (cand / "PPU" / "GELU_Unit.sv").exists():
            return cand
    raise FileNotFoundError(
        "Cannot find hardware root. Set VIT_HARDWARE_ROOT to the hardware folder, "
        "for example C:\\成大貓\\大四\\AOC\\炸彈惡魔\\AOC_final_project-main\\hardware"
    )


HARDWARE_ROOT = find_hardware_root()
INV_LUT_HEX = HARDWARE_ROOT / "Streaming_RMSNorm_Unit" / "hardware_export" / "rmsnorm_inv_sqrt_lut" / "rms_inv_sqrt_lut_A_global_10bit_Q2_14.hex"
EXP_LUT_HEX = HARDWARE_ROOT / "softmax_FPGA_package" / "softmax_FPGA_package" / "exp_lut_10bit_Q1_15_range12.hex"
GELU_SV = HARDWARE_ROOT / "PPU" / "GELU_Unit.sv"


def pack_u8_to_u32(data: np.ndarray | list[int]) -> list[int]:
    arr = np.asarray(data, dtype=np.uint8).reshape(-1)
    words: list[int] = []
    for i in range(0, arr.size, 4):
        w = 0
        for lane, b in enumerate(arr[i : i + 4]):
            w |= int(b) << (8 * lane)
        words.append(w & 0xFFFFFFFF)
    return words


def write_words(path: Path, words: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for w in words:
            f.write(f"{int(w) & 0xFFFFFFFF:08x}\n")


def load_hex_u16(path: Path) -> np.ndarray:
    vals = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            vals.append(int(s, 16) & 0xFFFF)
    return np.asarray(vals, dtype=np.int64)


def load_gelu_rom(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="ignore")
    rom = np.zeros(256, dtype=np.int64)
    for m in re.finditer(r"gelu_rom\[(\d+)\]\s*=\s*8'h([0-9a-fA-F]{2})", text):
        rom[int(m.group(1))] = int(m.group(2), 16)
    if not np.any(rom):
        raise RuntimeError(f"Could not parse GELU ROM from {path}")
    return rom


def resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    w, h = img.size
    if w < h:
        return img.resize((short_side, int(round(h * short_side / w))), Image.Resampling.BILINEAR)
    return img.resize((int(round(w * short_side / h)), short_side), Image.Resampling.BILINEAR)


def load_image_u8(image_path: Path) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    img = resize_short_side(img, 256)
    w, h = img.size
    left = (w - IMG_W) // 2
    top = (h - IMG_H) // 2
    img = img.crop((left, top, left + IMG_W, top + IMG_H))
    arr = np.asarray(img, dtype=np.uint8)
    if arr.shape != (IMG_H, IMG_W, IMG_C):
        raise RuntimeError(f"Bad image shape {arr.shape}")
    return arr


def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state: Any = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if not isinstance(state, dict):
        raise RuntimeError("checkpoint is not a state dict")
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        cleaned[nk] = v
    return cleaned


def t(state: dict[str, torch.Tensor], key: str) -> np.ndarray:
    if key not in state:
        raise KeyError(key)
    return state[key].detach().cpu().numpy().astype(np.float64)


def clamp_u8(v: np.ndarray) -> np.ndarray:
    return np.clip(v, 0, 255).astype(np.uint8)


def clamp_s8(v: np.ndarray) -> np.ndarray:
    return np.clip(v, -128, 127).astype(np.int64)


def quant_u8_zp128(x: np.ndarray, scale: float) -> np.ndarray:
    return clamp_u8(np.rint(x / scale).astype(np.int64) + 128)


def quant_s8(x: np.ndarray, scale: float) -> tuple[np.ndarray, int]:
    raw = np.rint(x / scale).astype(np.int64)
    clipped = clamp_s8(raw)
    return clipped, int(np.count_nonzero(raw != clipped))


def i8_to_u8(v: np.ndarray) -> np.ndarray:
    return (v.astype(np.int64) & 0xFF).astype(np.uint8)


def requant_zp128(psum: np.ndarray, shift: int) -> np.ndarray:
    shifted = np.right_shift(psum.astype(np.int64), shift)
    return clamp_u8(clamp_s8(shifted) + 128)


def residual_add(a_q: np.ndarray, b_q: np.ndarray) -> np.ndarray:
    return clamp_u8(a_q.astype(np.int64) + b_q.astype(np.int64) - 128)


def rms_inv_from_sum(sum_sq: np.ndarray, inv_lut: np.ndarray) -> np.ndarray:
    idx = ((sum_sq.astype(np.int64) * 620 + (1 << 15)) >> 16)
    idx = np.clip(idx, 0, 1023)
    return inv_lut[idx]


def rmsnorm_hw(x_q: np.ndarray, gamma_q: np.ndarray, inv_lut: np.ndarray) -> np.ndarray:
    x_s = x_q.astype(np.int64) - 128
    sum_sq = np.sum(x_s * x_s, axis=1)
    inv = rms_inv_from_sum(sum_sq, inv_lut).astype(np.int64)
    prod = x_s * inv[:, None] * gamma_q.astype(np.int64)[None, :]
    y_s = np.right_shift(prod, 28)
    return clamp_u8(clamp_s8(y_s) + 128)


def linear_hw(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray, shift: int, act_zp: int = 128) -> tuple[np.ndarray, np.ndarray]:
    x_s = x_q.astype(np.int64) - int(act_zp)
    psum = x_s @ w_q.astype(np.int64).T + b_q.astype(np.int64)[None, :]
    return requant_zp128(psum, shift), psum


def qkv_hw(x_q: np.ndarray, w_q: np.ndarray, b_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qkv_q, psum = linear_hw(x_q, w_q, b_q, SHIFT_QKV, act_zp=128)
    q = qkv_q[:, :EMBED_DIM]
    k = qkv_q[:, EMBED_DIM : 2 * EMBED_DIM]
    v = qkv_q[:, 2 * EMBED_DIM :]
    return q, k, v, psum


def softmax_hw(scores: np.ndarray, exp_lut: np.ndarray) -> np.ndarray:
    # scores: [heads, tokens, tokens] int64 Q*K opsums.
    out = np.zeros((HEAD_NUM, TOKEN_NUM, SOFTMAX_COLS), dtype=np.uint8)
    shift = SOFTMAX_Q_SHIFT + SOFTMAX_K_SHIFT + 2
    for h in range(HEAD_NUM):
        for r in range(TOKEN_NUM):
            row = np.zeros(SOFTMAX_COLS, dtype=np.int64)
            row[:TOKEN_NUM] = scores[h, r]
            scaled = np.right_shift(row, 3)
            valid = scaled[:TOKEN_NUM]
            max_score = int(np.max(valid))
            diff = scaled - max_score
            mag = np.maximum(-diff, 0).astype(np.int64)
            idx = np.right_shift(mag * 341, shift)
            idx = np.clip(idx, 0, 1023)
            expv = exp_lut[idx].astype(np.int64)
            expv[TOKEN_NUM:] = 0
            denom = int(np.sum(expv))
            if denom > 0:
                att = (expv * 128 + (denom >> 1)) // denom
                att = np.clip(att, 0, 127)
                out[h, r] = att.astype(np.uint8)
    return out


def gelu_requant_hw(psum: np.ndarray, gelu_rom: np.ndarray) -> np.ndarray:
    idx = np.right_shift(psum.astype(np.int64) & 0xFFFFFFFF, 8) & 0xFF
    gelu = gelu_rom[idx]
    return requant_zp128(gelu, SHIFT_FC1)


def pack_weight_matrix(weight_k_n: np.ndarray, k_tiles: int, n_tiles: int, base: int, storage_words: list[int]) -> None:
    # weight_k_n is signed int8 shaped [K, N].  Address is 32-bit word address.
    for n_tile in range(n_tiles):
        for k_tile in range(k_tiles):
            for k_inner in range(CHANNEL_TILE):
                k_idx = k_tile * CHANNEL_TILE + k_inner
                for word_sel in range(TILE_ROW_WORDS):
                    word = 0
                    for lane in range(4):
                        n_idx = n_tile * CHANNEL_TILE + word_sel * 4 + lane
                        val = 0
                        if k_idx < weight_k_n.shape[0] and n_idx < weight_k_n.shape[1]:
                            val = int(weight_k_n[k_idx, n_idx]) & 0xFF
                        word |= val << (8 * lane)
                    addr = (base +
                            n_tile * (k_tiles * WEIGHT_TILE_WORDS) +
                            k_tile * WEIGHT_TILE_WORDS +
                            k_inner * TILE_ROW_WORDS +
                            word_sel)
                    if addr >= len(storage_words):
                        raise IndexError(f"weight addr {addr} >= {len(storage_words)}")
                    storage_words[addr] = word


def pack_bias_vector(bias: np.ndarray, base: int, storage_words: list[int]) -> None:
    for i, val in enumerate(bias.astype(np.int64).reshape(-1)):
        addr = base + i
        if addr >= len(storage_words):
            raise IndexError(f"bias addr {addr} >= {len(storage_words)}")
        storage_words[addr] = int(val) & 0xFFFFFFFF


def max_external_weight_tile_id() -> int:
    phases = [
        (W_QKV_BASE, CHANNEL_TILE_NUM, QKV_CHANNEL_TILE_NUM),
        (W_OUT_BASE, CHANNEL_TILE_NUM, CHANNEL_TILE_NUM),
        (W_FC1_BASE, CHANNEL_TILE_NUM, FFN_CHANNEL_TILE_NUM),
        (W_FC2_BASE, FFN_CHANNEL_TILE_NUM, CHANNEL_TILE_NUM),
    ]
    max_tile = 0
    for base, k_tiles, n_tiles in phases:
        for n in range(n_tiles):
            last_word = base + n * (k_tiles * WEIGHT_TILE_WORDS) + k_tiles * WEIGHT_TILE_WORDS - 1
            max_tile = max(max_tile, last_word // WEIGHT_TILE_WORDS)
    return max_tile


def max_external_bias_tile_id() -> int:
    phases = [
        (B_QKV_BASE, QKV_CHANNEL_TILE_NUM),
        (B_OUT_BASE, CHANNEL_TILE_NUM),
        (B_FC1_BASE, FFN_CHANNEL_TILE_NUM),
        (B_FC2_BASE, CHANNEL_TILE_NUM),
    ]
    max_tile = 0
    for base, n_tiles in phases:
        for n in range(n_tiles):
            max_tile = max(max_tile, (base + n * BIAS_TILE_WORDS + BIAS_TILE_WORDS - 1) // BIAS_TILE_WORDS)
    return max_tile


def patches_from_image(image_u8: np.ndarray) -> np.ndarray:
    patches = []
    for py in range(IMG_H // PATCH_SIZE):
        for px in range(IMG_W // PATCH_SIZE):
            patch = image_u8[py*PATCH_SIZE:(py+1)*PATCH_SIZE, px*PATCH_SIZE:(px+1)*PATCH_SIZE, :]
            patches.append(patch.reshape(-1))
    return np.asarray(patches, dtype=np.int64)


def export_case(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    state = load_state(Path(args.checkpoint))
    image_u8 = load_image_u8(Path(args.image))
    patches_u8 = patches_from_image(image_u8)

    inv_lut = load_hex_u16(INV_LUT_HEX)
    exp_lut = load_hex_u16(EXP_LUT_HEX)
    gelu_rom = load_gelu_rom(GELU_SV)

    block = int(args.block_index)
    prefix = f"blocks.{block}"

    cls = t(state, "cls_token").reshape(TOKEN_NUM - PATCH_COUNT, EMBED_DIM)[0]
    pos = t(state, "pos_embed").reshape(TOKEN_NUM, EMBED_DIM)
    patch_w = t(state, "patch_embed.proj.weight")
    patch_b = t(state, "patch_embed.proj.bias")
    norm1_gamma = t(state, f"{prefix}.norm1.weight")
    norm2_gamma = t(state, f"{prefix}.norm2.weight")
    w_qkv = t(state, f"{prefix}.attn.qkv.weight")
    b_qkv = t(state, f"{prefix}.attn.qkv.bias")
    w_out = t(state, f"{prefix}.attn.proj.weight")
    b_out = t(state, f"{prefix}.attn.proj.bias")
    w_fc1 = t(state, f"{prefix}.mlp.fc1.weight")
    b_fc1 = t(state, f"{prefix}.mlp.fc1.bias")
    w_fc2 = t(state, f"{prefix}.mlp.fc2.weight")
    b_fc2 = t(state, f"{prefix}.mlp.fc2.bias")

    # Fold ImageNet normalization into patch projection so RTL can consume raw uint8 RGB.
    patch_w_eff = np.zeros((PATCH_ELEMS, EMBED_DIM), dtype=np.float64)
    patch_b_eff = patch_b.astype(np.float64).copy()
    for y in range(PATCH_SIZE):
        for x in range(PATCH_SIZE):
            for c in range(IMG_C):
                k = (y * PATCH_SIZE + x) * IMG_C + c
                w_oc = patch_w[:, c, y, x]
                patch_w_eff[k, :] = w_oc / (255.0 * IMAGENET_STD[c])
                patch_b_eff -= w_oc * (IMAGENET_MEAN[c] / IMAGENET_STD[c])

    patch_float = patches_u8.astype(np.float64) @ patch_w_eff + patch_b_eff[None, :]
    x_float = np.zeros((TOKEN_NUM, EMBED_DIM), dtype=np.float64)
    x_float[0] = cls + pos[0]
    x_float[1:] = patch_float + pos[1:]
    x_scale = float(max(np.max(np.abs(x_float)) / 120.0, 1e-6))

    patch_min_w_scale = float(np.max(np.abs(patch_w_eff)) / 127.0) if np.max(np.abs(patch_w_eff)) else 1e-9
    patch_shift = int(args.patch_shift) if args.patch_shift is not None else 0
    patch_w_scale = max(x_scale / (1 << patch_shift), patch_min_w_scale)
    if patch_w_scale != x_scale / (1 << patch_shift):
        # Keep RTL simple: adjust x_scale to the actual patch output scale.
        x_scale = patch_w_scale * (1 << patch_shift)

    patch_w_q, patch_w_clips = quant_s8(patch_w_eff, patch_w_scale)
    patch_b_q = np.rint(patch_b_eff / patch_w_scale).astype(np.int64)
    patch_psum = patches_u8.astype(np.int64) @ patch_w_q.astype(np.int64) + patch_b_q[None, :]
    patch_q = requant_zp128(patch_psum, patch_shift)
    pos_q = quant_u8_zp128(pos, x_scale)
    cls_q = quant_u8_zp128(cls, x_scale)
    x_q = np.zeros((TOKEN_NUM, EMBED_DIM), dtype=np.uint8)
    x_q[0] = residual_add(cls_q[None, :], pos_q[0:1])[0]
    x_q[1:] = residual_add(patch_q, pos_q[1:])

    gamma1_q = np.clip(np.rint(norm1_gamma * (1 << 14)), -32768, 32767).astype(np.int64)
    gamma2_q = np.clip(np.rint(norm2_gamma * (1 << 14)), -32768, 32767).astype(np.int64)

    x_norm = rmsnorm_hw(x_q, gamma1_q, inv_lut)

    # Scales chosen to match hard-coded RTL shifts and residual scale assumptions.
    qkv_scale = 2.0 ** -4
    qkv_w_scale = qkv_scale / (1 << SHIFT_QKV)
    qkv_w_q, qkv_w_clips = quant_s8(w_qkv, qkv_w_scale)
    qkv_b_q = np.rint(b_qkv / qkv_w_scale).astype(np.int64)
    q_q, k_q, v_q, _ = qkv_hw(x_norm, qkv_w_q, qkv_b_q)

    q_s = q_q.astype(np.int64) - 128
    k_s = k_q.astype(np.int64) - 128
    v_s = v_q.astype(np.int64) - 128
    scores = np.zeros((HEAD_NUM, TOKEN_NUM, TOKEN_NUM), dtype=np.int64)
    for h in range(HEAD_NUM):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        scores[h] = q_s[:, sl] @ k_s[:, sl].T
    attn_q = softmax_hw(scores, exp_lut)[:, :, :TOKEN_NUM]

    o_heads = np.zeros((TOKEN_NUM, EMBED_DIM), dtype=np.uint8)
    for h in range(HEAD_NUM):
        sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
        psum = attn_q[h].astype(np.int64) @ v_s[:, sl].astype(np.int64)
        o_heads[:, sl] = requant_zp128(psum, SHIFT_ATTN_V)
    o_scale = (1.0 / 128.0) * qkv_scale * (1 << SHIFT_ATTN_V)

    out_w_scale = x_scale / (o_scale * (1 << SHIFT_OUT_PROJ))
    out_w_q, out_w_clips = quant_s8(w_out, out_w_scale)
    out_b_q = np.rint(b_out / (o_scale * out_w_scale)).astype(np.int64)
    out_q, _ = linear_hw(o_heads, out_w_q, out_b_q, SHIFT_OUT_PROJ, act_zp=128)
    x_mid = residual_add(out_q, x_q)

    x_mid_norm = rmsnorm_hw(x_mid, gamma2_q, inv_lut)

    fc1_w_scale = float(args.fc1_weight_scale)
    fc1_w_q, fc1_w_clips = quant_s8(w_fc1, fc1_w_scale)
    fc1_b_q = np.rint(b_fc1 / fc1_w_scale).astype(np.int64)
    _, fc1_psum = linear_hw(x_mid_norm, fc1_w_q, fc1_b_q, SHIFT_FC1, act_zp=128)
    gelu_q = gelu_requant_hw(fc1_psum, gelu_rom)

    fc2_w_scale = x_scale / (1.0 * (1 << SHIFT_FC2))
    fc2_w_q, fc2_w_clips = quant_s8(w_fc2, fc2_w_scale)
    fc2_b_q = np.rint(b_fc2 / fc2_w_scale).astype(np.int64)
    fc2_q, _ = linear_hw(gelu_q, fc2_w_q, fc2_b_q, SHIFT_FC2, act_zp=128)
    x_out = residual_add(fc2_q, x_mid)

    patch_weight_tiles = PATCH_K_TILE_NUM * CHANNEL_TILE_NUM
    patch_bias_tiles = CHANNEL_TILE_NUM
    patch_weight_words = [0] * (patch_weight_tiles * WEIGHT_TILE_WORDS)
    patch_bias_words = [0] * (patch_bias_tiles * BIAS_TILE_WORDS)
    pack_weight_matrix(patch_w_q, PATCH_K_TILE_NUM, CHANNEL_TILE_NUM, 0, patch_weight_words)
    pack_bias_vector(patch_b_q, 0, patch_bias_words)

    trans_weight_tiles = max_external_weight_tile_id() + 1
    trans_bias_tiles = max_external_bias_tile_id() + 1
    trans_weight_words = [0] * (trans_weight_tiles * WEIGHT_TILE_WORDS)
    trans_bias_words = [0] * (trans_bias_tiles * BIAS_TILE_WORDS)
    pack_weight_matrix(qkv_w_q.T, CHANNEL_TILE_NUM, QKV_CHANNEL_TILE_NUM, W_QKV_BASE, trans_weight_words)
    pack_bias_vector(qkv_b_q, B_QKV_BASE, trans_bias_words)
    pack_weight_matrix(out_w_q.T, CHANNEL_TILE_NUM, CHANNEL_TILE_NUM, W_OUT_BASE, trans_weight_words)
    pack_bias_vector(out_b_q, B_OUT_BASE, trans_bias_words)
    pack_weight_matrix(fc1_w_q.T, CHANNEL_TILE_NUM, FFN_CHANNEL_TILE_NUM, W_FC1_BASE, trans_weight_words)
    pack_bias_vector(fc1_b_q, B_FC1_BASE, trans_bias_words)
    pack_weight_matrix(fc2_w_q.T, FFN_CHANNEL_TILE_NUM, CHANNEL_TILE_NUM, W_FC2_BASE, trans_weight_words)
    pack_bias_vector(fc2_b_q, B_FC2_BASE, trans_bias_words)

    gamma_words = [int(v) & 0xFFFF for v in np.concatenate([gamma1_q, gamma2_q])]
    x_out_words = [int(v) for v in x_out.reshape(-1)]
    files = {
        "image.hex": pack_u8_to_u32(image_u8.reshape(-1)),
        "raw_image.hex": pack_u8_to_u32(image_u8.reshape(-1)),
        "pos.hex": pack_u8_to_u32(pos_q.reshape(-1)),
        "position.hex": pack_u8_to_u32(pos_q.reshape(-1)),
        "cls.hex": pack_u8_to_u32(cls_q.reshape(-1)),
        "gamma.hex": gamma_words,
        "patch_weight.hex": patch_weight_words,
        "patch_bias.hex": patch_bias_words,
        "transformer_weight.hex": trans_weight_words,
        "transformer_bias.hex": trans_bias_words,
        "weight.hex": trans_weight_words,
        "bias.hex": trans_bias_words,
        "x_out.hex": x_out_words,
        "x_out_packed.hex": pack_u8_to_u32(x_out.reshape(-1)),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, words in files.items():
        write_words(out_dir / name, words)

    manifest: dict[str, Any] = {
        "kind": "rtl_real_model_block0" if block == 0 else f"rtl_real_model_block{block}",
        "description": "Real checkpoint vectors for the current one-block integer RTL dataflow.",
        "checkpoint": str(args.checkpoint),
        "image_source": str(args.image),
        "block_index": block,
        "rtl_scope": "one ViT block; current RTL does not iterate all 12 blocks",
        "patch_requant_shift": patch_shift,
        "tile_layout": {
            "token_tile": TOKEN_TILE,
            "channel_tile": CHANNEL_TILE,
            "tile_row_words": TILE_ROW_WORDS,
            "weight_tile_words": WEIGHT_TILE_WORDS,
            "bias_tile_words": BIAS_TILE_WORDS,
        },
        "fixed_rtl_shifts": {
            "qkv": SHIFT_QKV,
            "attn_v": SHIFT_ATTN_V,
            "out_proj": SHIFT_OUT_PROJ,
            "fc1": SHIFT_FC1,
            "fc2": SHIFT_FC2,
            "softmax_q": SOFTMAX_Q_SHIFT,
            "softmax_k": SOFTMAX_K_SHIFT,
        },
        "scales": {
            "x_scale": x_scale,
            "patch_w_scale": patch_w_scale,
            "qkv_out_scale": qkv_scale,
            "qkv_w_scale": qkv_w_scale,
            "attn_v_out_scale": o_scale,
            "out_proj_w_scale": out_w_scale,
            "fc1_w_scale": fc1_w_scale,
            "fc2_w_scale": fc2_w_scale,
        },
        "clip_counts": {
            "patch_w": patch_w_clips,
            "qkv_w": qkv_w_clips,
            "out_proj_w": out_w_clips,
            "fc1_w": fc1_w_clips,
            "fc2_w": fc2_w_clips,
        },
        "word_counts": {name: len(words) for name, words in files.items()},
        "expected_value": "x_out.hex contains one uint8 activation per 32-bit word low byte",
        "compare_mode": "exact_rtl_integer_model",
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.save(out_dir / "x_out.npy", x_out)
    np.save(out_dir / "x_mid.npy", x_mid)
    np.save(out_dir / "x_after_patch.npy", x_q)
    np.savez(
        out_dir / f"stage_golden_block{int(block)}.npz",
        x_after_patch=x_q.astype(np.uint8),
        x_norm1=x_norm.astype(np.uint8),
        o_attn=o_heads.astype(np.uint8),
        x_mid=x_mid.astype(np.uint8),
        x_mid_norm=x_mid_norm.astype(np.uint8),
        gelu=gelu_q.astype(np.uint8),
        x_out=x_out.astype(np.uint8),
        x_scale=np.asarray([x_scale], dtype=np.float32),
        patch_shift=np.asarray([int(patch_shift)], dtype=np.int32),
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=r"C:\Users\angelliu.LAPTOP-3NTJHQPG\Downloads\golden_gen\rms_qat_best.pt")
    ap.add_argument("--image", default=r"C:\Users\angelliu.LAPTOP-3NTJHQPG\Downloads\golden_gen\n01440764_10194.jpg")
    ap.add_argument("--block-index", type=int, default=0)
    ap.add_argument("--patch-shift", type=int, default=None)
    ap.add_argument("--fc1-weight-scale", type=float, default=1.0 / 1024.0)
    ap.add_argument(
        "--single-block-only",
        action="store_true",
        help="Only export --block-index into the root case directory. By default, also export blocks/block_00..block_11 for Python-driven 12-block hardware replay.",
    )
    ap.add_argument(
        "--out",
        action="append",
        default=None,
        help="Output directory. Default: Downloads/golden_gen/hex/case_vit_real_model and, when present, Z:/jupyter_notebooks/VIT_fulldesign/golden_gen/hex/case_vit_real_model.",
    )
    args = ap.parse_args()

    if args.out is None:
        default_outs = [Path(__file__).resolve().parent / "hex" / "case_vit_real_model"]
        z_case = Path(r"Z:\jupyter_notebooks\VIT_fulldesign\golden_gen\hex\case_vit_real_model")
        if z_case.parent.exists():
            default_outs.append(z_case)
        args.out = [str(p) for p in default_outs]

    print("Hardware root:", HARDWARE_ROOT)
    print("Image:", args.image)
    print("Checkpoint:", args.checkpoint)

    for out in args.out:
        out_path = Path(out)
        manifest = export_case(args, out_path)
        print(f"[OK] wrote {manifest['kind']} to {out}")
        print(f"     patch_shift={manifest['patch_requant_shift']} x_scale={manifest['scales']['x_scale']:.8g}")
        print(f"     clips={manifest['clip_counts']}")
        print(f"     x_out_words={manifest['word_counts']['x_out.hex']}")

        if not args.single_block_only:
            original_block = int(args.block_index)
            for block_idx in range(12):
                args.block_index = block_idx
                block_dir = out_path / "blocks" / f"block_{block_idx:02d}"
                block_manifest = export_case(args, block_dir)
                print(
                    f"     [block {block_idx:02d}] wrote gamma/transformer vectors "
                    f"to {block_dir} ({block_manifest['word_counts']['transformer_weight.hex']} weight words)"
                )
            args.block_index = original_block


if __name__ == "__main__":
    main()
