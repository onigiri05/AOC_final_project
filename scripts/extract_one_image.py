#!/usr/bin/env python3
"""
extract_one_image.py — 從 ImageNet 資料集（多種格式）萃取一張圖片並預處理。

支援格式：
  - HuggingFace Dataset (load_from_disk) —— IMAGENET_VAL_HF 通常是這種
  - 普通圖片目錄 (JPEG/PNG 檔案)

用法：
  python scripts/extract_one_image.py --ds <資料集路徑> [--idx 0] [--out test.bin]

輸出：
  <out>.bin   float32 CHW [3,224,224]，ImageNet 正規化
  印出 class index 和 class name，供 tb_vit 的 CLASS 參數使用

之後在 PowerShell：
  docker cp test.bin c95c3ccbbddf:/tmp/test.bin

在 Docker 容器內：
  ./test/testbench/vit/build/tb_vit "$(pwd)/weights/" /tmp/test.bin <class_index>
"""

import argparse
import os
import sys
import numpy as np

# ImageNet 1k class names（前 20 + 常用類別，完整版太長）
IMAGENET_CLASSES = {
    0: "tench",           1: "goldfish",         2: "great_white_shark",
    3: "tiger_shark",     4: "hammerhead",        5: "electric_ray",
    6: "stingray",        7: "cock",              8: "hen",
    9: "ostrich",         281: "tabby_cat",       282: "tiger_cat",
    283: "Persian_cat",   284: "Siamese_cat",     285: "Egyptian_cat",
    386: "elephant",      388: "panda",           954: "banana",
    955: "jackfruit",     970: "alp",             980: "volcano",
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_pil(pil_img):
    """PIL Image → float32 [3,224,224] CHW, ImageNet-normalized."""
    from PIL import Image
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    s = min(w, h)
    pil_img = pil_img.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2))
    pil_img = pil_img.resize((224, 224), Image.BICUBIC)
    arr = np.array(pil_img, dtype=np.float32) / 255.0   # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)                        # CHW


def load_from_hf(ds_path, idx):
    """HuggingFace datasets (load_from_disk) 格式。"""
    try:
        from datasets import load_from_disk
    except ImportError:
        print("[ERR] 請安裝: pip install datasets")
        raise SystemExit(1)

    print(f"[INFO] 載入 HuggingFace Dataset: {ds_path}")
    ds = load_from_disk(ds_path)

    # 可能有多個 split
    if hasattr(ds, "keys"):
        split = "validation" if "validation" in ds else list(ds.keys())[0]
        print(f"[INFO] 使用 split: {split}  (共 {len(ds[split])} 筆)")
        ds = ds[split]
    else:
        print(f"[INFO] 共 {len(ds)} 筆")

    item  = ds[idx]
    image = item.get("image") or item.get("img")
    label = item.get("label", -1)

    if image is None:
        print("[ERR] 找不到 'image' 或 'img' 欄位，資料集欄位：", list(item.keys()))
        raise SystemExit(1)

    return image, int(label)


def load_from_dir(dir_path, idx):
    """普通圖片目錄格式（JPEG/PNG 依字母排序，按 idx 選）。"""
    from PIL import Image
    exts = (".jpeg", ".jpg", ".png", ".JPEG", ".JPG")
    files = sorted(
        f for f in os.listdir(dir_path)
        if os.path.splitext(f)[1] in exts
    )
    if not files:
        # 嘗試遞迴找（subdirectory 格式）
        files = []
        for root, _, fnames in os.walk(dir_path):
            for f in fnames:
                if os.path.splitext(f)[1].lower() in (".jpeg", ".jpg", ".png"):
                    files.append(os.path.join(root, f))
        files = sorted(files)

    if not files:
        print(f"[ERR] 在 {dir_path} 找不到圖片檔案")
        raise SystemExit(1)

    print(f"[INFO] 找到 {len(files)} 張圖片，使用第 {idx} 張")
    path = files[idx]
    print(f"[INFO] 圖片路徑: {path}")
    img = Image.open(path).convert("RGB")
    # 目錄格式無法自動得到 label，需用 ground truth 檔案
    return img, -1


def main():
    ap = argparse.ArgumentParser(
        description="從 ImageNet 資料集萃取一張圖片並預處理成 float32 binary")
    ap.add_argument("--ds",  required=True,
                    help="資料集路徑（HuggingFace disk 格式或圖片目錄）")
    ap.add_argument("--idx", type=int, default=0,
                    help="資料集中的圖片索引（預設 0）")
    ap.add_argument("--out", default="test_imagenet.bin",
                    help="輸出 binary 路徑（預設 test_imagenet.bin）")
    args = ap.parse_args()

    # 判斷格式
    ds_path  = args.ds
    hf_marker = os.path.join(ds_path, "dataset_info.json")
    arrow_dir = os.path.join(ds_path, "data")

    if os.path.exists(hf_marker) or os.path.exists(arrow_dir):
        pil_img, label = load_from_hf(ds_path, args.idx)
    else:
        pil_img, label = load_from_dir(ds_path, args.idx)

    # Preprocess
    arr = preprocess_pil(pil_img)
    assert arr.shape == (3, 224, 224), f"Wrong shape: {arr.shape}"

    arr.tofile(args.out)
    size_kb = arr.nbytes / 1024

    class_name = IMAGENET_CLASSES.get(label, f"class_{label}")
    print(f"\n[OK] 已儲存: {args.out}  ({size_kb:.0f} KB)")
    print(f"     Class index : {label}  ({class_name})")
    print(f"     影像 shape  : {arr.shape}  dtype=float32 CHW")
    print()
    print("下一步 (PowerShell):")
    print(f'  docker cp "{os.path.abspath(args.out)}" c95c3ccbbddf:/tmp/test_imagenet.bin')
    print()
    print("Docker 容器內:")
    print(f'  ./test/testbench/vit/build/tb_vit "$(pwd)/weights/" /tmp/test_imagenet.bin {label}')


if __name__ == "__main__":
    main()
