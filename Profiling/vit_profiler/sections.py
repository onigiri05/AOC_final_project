from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import ViTModelSpec


@dataclass(frozen=True)
class SectionShape:
    name: str
    kind: str


def build_vit_block_sections(spec: ViTModelSpec) -> List[SectionShape]:
    return [
        SectionShape("Norm", "elementwise"),
        SectionShape("QKV Projection", "gemm"),
        SectionShape("Attention Score", "gemm"),
        SectionShape("Softmax", "softmax"),
        SectionShape("Attention Value", "gemm"),
        SectionShape("Output Projection", "gemm_elementwise"),
        SectionShape("MLP", "gemm_elementwise"),
    ]


def build_group_definitions() -> Dict[str, List[str]]:
    return {
        "Full MHSA": [
            "QKV Projection",
            "Attention Score",
            "Softmax",
            "Attention Value",
            "Output Projection",
        ],
        "Full one block": [
            "Norm",
            "QKV Projection",
            "Attention Score",
            "Softmax",
            "Attention Value",
            "Output Projection",
            "MLP",
        ],
    }
