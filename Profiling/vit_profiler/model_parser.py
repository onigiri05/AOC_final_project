from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from .config import ViTModelSpec


def _safe_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default


def parse_timm_model(model_name: str, pretrained: bool = False) -> ViTModelSpec:
    """Parse timm ViT shapes. Falls back to ViT-Small/16 defaults if timm is unavailable."""
    try:
        import timm  # type: ignore
        model = timm.create_model(model_name, pretrained=pretrained)
        embed_dim = _safe_int(getattr(model, "embed_dim", 384), 384)
        num_heads = _safe_int(getattr(model.blocks[0].attn, "num_heads", 6), 6)
        mlp_dim = _safe_int(model.blocks[0].mlp.fc1.out_features, 4 * embed_dim)
        qkv_out_dim = _safe_int(model.blocks[0].attn.qkv.out_features, 3 * embed_dim)
        patch_size = getattr(model.patch_embed, "patch_size", (16, 16))
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        img_size = getattr(model.patch_embed, "img_size", (224, 224))
        if isinstance(img_size, tuple):
            img_size = img_size[0]
        patch_tokens = (img_size // patch_size) ** 2
        has_cls = getattr(model, "cls_token", None) is not None
        tokens = patch_tokens + (1 if has_cls else 0)
        num_blocks = len(model.blocks)
        num_classes = _safe_int(getattr(model.head, "out_features", 1000), 1000)
        return ViTModelSpec(
            name=model_name,
            image_size=img_size,
            patch_size=patch_size,
            tokens=tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            mlp_dim=mlp_dim,
            num_blocks=num_blocks,
            qkv_out_dim=qkv_out_dim,
            num_classes=num_classes,
            has_cls_token=has_cls,
            source="parsed_from_timm",
        )
    except Exception as e:
        print(f"[WARN] Could not parse timm model '{model_name}': {e}")
        print("[WARN] Falling back to default ViT-Small/16 config.")
        return ViTModelSpec(name=model_name, source="fallback_default_timm_parse_failed")


def _extract_state_dict(obj: Any) -> Mapping[str, Any]:
    if isinstance(obj, Mapping):
        for key in ["state_dict", "model", "model_state_dict", "net"]:
            if key in obj and isinstance(obj[key], Mapping):
                return obj[key]
        return obj
    return {}


def _shape_of(sd: Mapping[str, Any], suffixes: list[str]) -> tuple[int, ...] | None:
    for k, v in sd.items():
        if any(k.endswith(s) for s in suffixes):
            shape = getattr(v, "shape", None)
            if shape is not None:
                return tuple(int(x) for x in shape)
    return None


def parse_checkpoint(path: str | Path, name: str | None = None) -> ViTModelSpec:
    """Parse an optimized checkpoint by state_dict tensor shapes. Falls back safely."""
    path = Path(path)
    try:
        import torch  # type: ignore
        obj = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(obj)
        qkv = _shape_of(sd, ["attn.qkv.weight", "qkv.weight"])
        fc1 = _shape_of(sd, ["mlp.fc1.weight", "fc1.weight"])
        pos = _shape_of(sd, ["pos_embed"])
        patch = _shape_of(sd, ["patch_embed.proj.weight", "proj.weight"])
        head = _shape_of(sd, ["head.weight"])

        embed_dim = qkv[1] if qkv and len(qkv) == 2 else 384
        qkv_out_dim = qkv[0] if qkv and len(qkv) == 2 else 3 * embed_dim
        mlp_dim = fc1[0] if fc1 and len(fc1) == 2 else 4 * embed_dim
        tokens = pos[1] if pos and len(pos) == 3 else 197
        patch_size = patch[-1] if patch and len(patch) == 4 else 16
        image_size = int(((tokens - 1) ** 0.5) * patch_size) if tokens > 1 else 224
        num_classes = head[0] if head and len(head) == 2 else 1000
        num_heads = 6 if embed_dim == 384 else max(1, embed_dim // 64)
        return ViTModelSpec(
            name=name or path.name,
            image_size=image_size,
            patch_size=patch_size,
            tokens=tokens,
            embed_dim=embed_dim,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
            mlp_dim=mlp_dim,
            qkv_out_dim=qkv_out_dim,
            num_classes=num_classes,
            source=f"parsed_from_checkpoint:{path.name}",
        )
    except Exception as e:
        print(f"[WARN] Could not torch.load checkpoint '{path}': {e}")
        print("[WARN] Falling back to default ViT-Small/16 config.")
        return ViTModelSpec(name=name or path.name, source="fallback_default_checkpoint_parse_failed")


def save_specs(specs: Dict[str, ViTModelSpec], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v.to_dict() for k, v in specs.items()}, f, indent=2)
