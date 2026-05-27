#!/usr/bin/env python3
"""
export_weights.py — Export ViT-Small/16 weights from timm to float32 binary.

Usage:
    pip install timm torch torchvision
    python export_weights.py --out weights/

Output directory layout:
    patch_embed_weight.bin  [384, 768]  fp32 row-major
    patch_embed_bias.bin    [384]
    cls_token.bin           [384]
    pos_embed.bin           [197, 384]
    block_00_norm1_w.bin    [384]  (×12 blocks, norm1/norm2, qkv/proj, fc1/fc2)
    ...
    norm_w.bin              [384]
    norm_b.bin              [384]
    head_w.bin              [1000, 384]
    head_b.bin              [1000]
    manifest.txt            shape/size summary
"""

import argparse
import os
import numpy as np


def save(out_dir, name, tensor):
    arr = tensor.detach().cpu().float().numpy()
    path = os.path.join(out_dir, name + ".bin")
    arr.tofile(path)
    return arr.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",   default="weights", help="Output directory")
    ap.add_argument("--model", default="vit_small_patch16_224")
    args = ap.parse_args()

    try:
        import timm, torch
    except ImportError:
        print("Run: pip install timm torch")
        raise SystemExit(1)

    os.makedirs(args.out, exist_ok=True)
    manifest = []

    def dump(name, t):
        shape = save(args.out, name, t)
        size  = int(np.prod(shape))
        manifest.append(f"{name}: shape={list(shape)} bytes={size * 4}")
        print(f"  {name}: {list(shape)}")

    print(f"Loading {args.model} (pretrained=True)...")
    model = timm.create_model(args.model, pretrained=True).eval()

    # Patch embedding: weight [384,3,16,16] → flatten to [384,768]
    pe = model.patch_embed
    dump("patch_embed_weight", pe.proj.weight.reshape(384, -1))
    dump("patch_embed_bias",   pe.proj.bias)

    # CLS token [1,1,384] → [384]; position embedding [1,197,384] → [197,384]
    dump("cls_token", model.cls_token.squeeze())
    dump("pos_embed",  model.pos_embed.squeeze(0))

    # Transformer blocks
    for i, blk in enumerate(model.blocks):
        p = f"block_{i:02d}"
        dump(f"{p}_norm1_w",   blk.norm1.weight)
        dump(f"{p}_norm1_b",   blk.norm1.bias)
        dump(f"{p}_qkv_w",     blk.attn.qkv.weight)   # [1152, 384]
        dump(f"{p}_qkv_b",     blk.attn.qkv.bias)
        dump(f"{p}_proj_w",    blk.attn.proj.weight)   # [384, 384]
        dump(f"{p}_proj_b",    blk.attn.proj.bias)
        dump(f"{p}_norm2_w",   blk.norm2.weight)
        dump(f"{p}_norm2_b",   blk.norm2.bias)
        dump(f"{p}_mlp_fc1_w", blk.mlp.fc1.weight)    # [1536, 384]
        dump(f"{p}_mlp_fc1_b", blk.mlp.fc1.bias)
        dump(f"{p}_mlp_fc2_w", blk.mlp.fc2.weight)    # [384, 1536]
        dump(f"{p}_mlp_fc2_b", blk.mlp.fc2.bias)

    # Final norm + classification head
    dump("norm_w", model.norm.weight)
    dump("norm_b", model.norm.bias)
    dump("head_w", model.head.weight)  # [1000, 384]
    dump("head_b", model.head.bias)

    with open(os.path.join(args.out, "manifest.txt"), "w") as f:
        f.write("\n".join(manifest) + "\n")

    total_bytes = sum(int(m.split("bytes=")[1]) for m in manifest)
    print(f"\nDone. {len(manifest)} tensors → '{args.out}/'")
    print(f"Total size: {total_bytes / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
