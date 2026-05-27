#!/usr/bin/env python3
"""
preprocess_image.py — Resize + normalize an image for ViT-Small/16 inference.

Output: float32 binary [3, 224, 224] channel-first (C,H,W), ImageNet normalization.

Usage:
    python preprocess_image.py cat.jpg cat.bin
    python preprocess_image.py cat.jpg cat.bin --label 281   # expected class (optional)
"""

import argparse
import sys
import numpy as np


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess(path):
    try:
        from PIL import Image
    except ImportError:
        print("Run: pip install Pillow")
        raise SystemExit(1)

    img = Image.open(path).convert("RGB")

    # Center crop to square then resize to 224×224
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((224, 224), Image.BICUBIC)

    arr = np.array(img, dtype=np.float32) / 255.0          # [224,224,3]  HWC
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std  = np.array(IMAGENET_STD,  dtype=np.float32)
    arr  = (arr - mean) / std
    arr  = arr.transpose(2, 0, 1)                           # CHW
    assert arr.shape == (3, 224, 224)
    return arr


def verify_with_pytorch(img_path, expected=None):
    """Optional: run timm model to get ground-truth top-5."""
    try:
        import timm, torch
        from timm.data import resolve_data_config, create_transform
    except ImportError:
        return

    model = timm.create_model("vit_small_patch16_224", pretrained=True).eval()
    cfg   = resolve_data_config({}, model=model)
    tfm   = create_transform(**cfg)
    from PIL import Image
    x = tfm(Image.open(img_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)[0]
    probs  = torch.softmax(logits, dim=0)
    top5   = probs.topk(5)
    print("\n[PyTorch reference top-5]")
    for prob, idx in zip(top5.values.tolist(), top5.indices.tolist()):
        marker = " ← expected" if expected is not None and idx == expected else ""
        print(f"  class {idx:4d}  prob={prob:.4f}{marker}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image",  help="Input image path")
    ap.add_argument("output", help="Output .bin path (float32 CHW)")
    ap.add_argument("--label", type=int, default=None,
                    help="Expected ImageNet class index (for verification)")
    args = ap.parse_args()

    arr = preprocess(args.image)
    arr.tofile(args.output)
    print(f"Saved {arr.shape} float32 → {args.output}  ({arr.nbytes} bytes)")

    verify_with_pytorch(args.image, args.label)


if __name__ == "__main__":
    main()
