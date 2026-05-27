#!/usr/bin/env python3
"""
batch_preprocess.py — 從 ImageNet 驗證集批次萃取 N 張圖片並預處理。

支援三種格式（自動偵測）：

  ① 編號資料夾格式 (你的資料集！)
       IMAGENET_VAL_HF/
         00000/  ← 資料夾名稱 = class index
           ILSVRC2012_val_XXXXXXXX_n01440764.JPEG
         00001/
         ...
       → 自動從每個類別資料夾各取 1 張圖片（跨類別均勻取樣）
       → 資料夾名稱直接作為 ground-truth label

  ② HuggingFace Dataset (dataset_info.json / data/ 目錄存在)
       → 從 HF 格式逐筆讀取

  ③ 一般圖片目錄（無 ground-truth label）
       → os.walk 遞迴找 JPEG/PNG 檔案

用法：
  python scripts/batch_preprocess.py --ds "C:\\完整路徑\\IMAGENET_VAL_HF" --n 50 --out batch_50/

  !! 注意：必須使用完整絕對路徑，不能縮寫成 C:\\...\\

輸出：
  batch_50/img_0000.bin  float32 CHW [3,224,224]，ImageNet 正規化
  ...
  batch_50/labels.txt    每行一個 ground-truth class index
  batch_50/summary.txt   摘要

Docker 工作流程：
  docker cp batch_50/ <container>:/tmp/batch_50
  ./test/testbench/vit/build/tb_vit weights/ --batch /tmp/batch_50/
  ./test/testbench/vit/build/tb_vit weights/ --batch /tmp/batch_50/ --hw
"""

import argparse
import os
import sys
import numpy as np

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

EXTS = {".jpeg", ".jpg", ".png", ".JPEG", ".JPG", ".PNG"}


def preprocess_pil(pil_img):
    """PIL Image → float32 CHW [3,224,224], ImageNet-normalized."""
    from PIL import Image
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    s = min(w, h)
    pil_img = pil_img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    pil_img = pil_img.resize((224, 224), Image.BICUBIC)
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)  # CHW


# ── 格式 ①：編號資料夾（資料夾名稱 = class index）────────────────────────────

def detect_numbered_dirs(ds_path):
    """
    檢查是否為編號資料夾格式。
    條件：ds_path 內存在至少 10 個全數字命名的子資料夾（如 00000, 00001...）。
    回傳已排序的子資料夾清單，或 None（不符合格式）。
    """
    try:
        entries = os.listdir(ds_path)
    except PermissionError:
        return None
    numeric_dirs = sorted(
        e for e in entries
        if os.path.isdir(os.path.join(ds_path, e)) and e.isdigit()
    )
    return numeric_dirs if len(numeric_dirs) >= 10 else None


def load_from_numbered_dirs(ds_path, numeric_dirs, start_class, n_classes):
    """
    從編號資料夾逐類別取 1 張圖片。
    yield (pil_img, label)，label = int(資料夾名稱)。
    """
    from PIL import Image

    classes_to_use = numeric_dirs[start_class: start_class + n_classes]
    print(f"[INFO] 偵測到編號資料夾格式（{len(numeric_dirs)} 個類別）")
    print(f"[INFO] 使用類別 {int(classes_to_use[0])} ~ {int(classes_to_use[-1])}"
          f"（共 {len(classes_to_use)} 個類別，各取 1 張）")

    for class_dir in classes_to_use:
        label     = int(class_dir)
        cls_path  = os.path.join(ds_path, class_dir)
        imgs      = sorted(f for f in os.listdir(cls_path)
                           if os.path.splitext(f)[1] in EXTS)
        if not imgs:
            print(f"[WARN] class {label:4d} ({class_dir}) 無圖片，跳過")
            continue
        img_path  = os.path.join(cls_path, imgs[0])
        pil_img   = Image.open(img_path).convert("RGB")
        yield pil_img, label


# ── 格式 ②：HuggingFace Dataset ──────────────────────────────────────────────

def detect_hf(ds_path):
    return (os.path.exists(os.path.join(ds_path, "dataset_info.json")) or
            os.path.exists(os.path.join(ds_path, "data")) or
            os.path.exists(os.path.join(ds_path, "dataset_dict.json")))


def load_from_hf(ds_path, start, count):
    try:
        from datasets import load_from_disk
    except ImportError:
        print("[ERR] 請安裝: pip install datasets")
        raise SystemExit(1)

    print(f"[INFO] 載入 HuggingFace Dataset: {ds_path}")
    ds = load_from_disk(ds_path)
    if hasattr(ds, "keys"):
        split = "validation" if "validation" in ds else list(ds.keys())[0]
        print(f"[INFO] split={split}  共 {len(ds[split])} 筆")
        ds = ds[split]
    else:
        print(f"[INFO] 共 {len(ds)} 筆")

    end = min(start + count, len(ds))
    for idx in range(start, end):
        item  = ds[idx]
        image = item.get("image") or item.get("img")
        label = int(item.get("label", -1))
        if image is None:
            print(f"[WARN] idx={idx} 無 image 欄位，跳過")
            continue
        yield image, label


# ── 格式 ③：一般圖片目錄（無 label）─────────────────────────────────────────

def load_from_flat_dir(ds_path, start, count):
    from PIL import Image

    files = []
    for root, _, fnames in os.walk(ds_path):
        for f in fnames:
            if os.path.splitext(f)[1] in EXTS:
                files.append(os.path.join(root, f))
    files.sort()

    if not files:
        print(f"[ERR] 在 {ds_path} 找不到任何 JPEG/PNG 圖片")
        print("[提示] 確認路徑正確、檔案副檔名為 .jpg/.jpeg/.png/.JPEG/.JPG/.PNG")
        raise SystemExit(1)

    end = min(start + count, len(files))
    print(f"[INFO] 找到 {len(files)} 張圖片（遞迴），使用 {start}..{end-1}")
    for idx in range(start, end):
        img = Image.open(files[idx]).convert("RGB")
        # 嘗試從父資料夾名稱推斷 label（若為全數字）
        parent = os.path.basename(os.path.dirname(files[idx]))
        label  = int(parent) if parent.isdigit() else -1
        yield img, label


# ── 主程式 ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="批次萃取 ImageNet 圖片並預處理成 float32 binary",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ds",    required=True,
                    help='資料集完整路徑，例：'
                         '"C:\\Users\\User\\Desktop\\...\\IMAGENET_VAL_HF"')
    ap.add_argument("--n",     type=int, default=50,
                    help="萃取張數（預設 50）")
    ap.add_argument("--start", type=int, default=0,
                    help="起始類別索引（預設 0）")
    ap.add_argument("--out",   default="batch_images",
                    help="輸出目錄（預設 batch_images/）")
    args = ap.parse_args()

    # ── 路徑檢查 ──
    ds_path = os.path.abspath(args.ds)
    if not os.path.exists(ds_path):
        print(f"[ERR] 找不到資料集路徑：{ds_path}")
        print()
        print("[提示] 常見錯誤：")
        print("  ✗ --ds C:\\...\\IMAGENET_VAL_HF   （不能用 ... 縮寫）")
        print("  ✓ --ds \"C:\\Users\\User\\Desktop\\AOC_Final\\炸彈惡魔\\提報後重新規劃\\第二階段\\IMAGENET_VAL_HF\"")
        raise SystemExit(1)

    print(f"[INFO] 資料集路徑：{ds_path}")
    os.makedirs(args.out, exist_ok=True)

    # ── 格式偵測 ──
    numeric_dirs = detect_numbered_dirs(ds_path)
    if numeric_dirs:
        print("[INFO] 格式：① 編號資料夾（資料夾名稱 = class index）")
        source = load_from_numbered_dirs(ds_path, numeric_dirs, args.start, args.n)
    elif detect_hf(ds_path):
        print("[INFO] 格式：② HuggingFace Dataset")
        source = load_from_hf(ds_path, args.start, args.n)
    else:
        print("[INFO] 格式：③ 一般圖片目錄（遞迴搜尋）")
        source = load_from_flat_dir(ds_path, args.start, args.n)

    # ── 預處理 ──
    labels    = []
    processed = 0
    failed    = 0

    for pil_img, label in source:
        try:
            arr = preprocess_pil(pil_img)
            assert arr.shape == (3, 224, 224)
        except Exception as e:
            print(f"[WARN] 圖片 {processed + failed} 預處理失敗: {e}")
            failed += 1
            continue

        bin_path = os.path.join(args.out, f"img_{processed:04d}.bin")
        arr.tofile(bin_path)
        labels.append(label)
        processed += 1

        if processed % 10 == 0 or processed == 1:
            print(f"[INFO] {processed}/{args.n}  class={label}")

    # ── 寫入 labels.txt ──
    with open(os.path.join(args.out, "labels.txt"), "w") as f:
        for lb in labels:
            f.write(f"{lb}\n")

    # ── 寫入 summary.txt ──
    known    = sum(1 for lb in labels if lb >= 0)
    size_mb  = processed * 3 * 224 * 224 * 4 / 1024 / 1024
    with open(os.path.join(args.out, "summary.txt"), "w") as f:
        f.write(f"images     : {processed}\n")
        f.write(f"failed     : {failed}\n")
        f.write(f"with_label : {known}\n")
        f.write(f"total_size : {size_mb:.1f} MB\n")
        f.write(f"source     : {ds_path}\n")
        f.write(f"start      : {args.start}\n")

    out_abs = os.path.abspath(args.out)
    out_name = os.path.basename(args.out.rstrip("/\\"))

    print(f"\n[OK] 完成：{processed} 張圖片 → {args.out}/")
    print(f"     容量  ：{size_mb:.1f} MB")
    print(f"     已知標籤：{known}/{processed}")
    print()
    print("下一步 (PowerShell)：")
    print(f'  docker cp "{out_abs}" c95c3ccbbddf:/tmp/{out_name}')
    print()
    print("Docker 容器內（SW 模式，快）：")
    print(f'  ./test/testbench/vit/build/tb_vit weights/ --batch /tmp/{out_name}/')
    print()
    print("Docker 容器內（HW 模式，慢，建議 ≤10 張）：")
    print(f'  ./test/testbench/vit/build/tb_vit weights/ --batch /tmp/{out_name}/ --hw')


if __name__ == "__main__":
    main()
