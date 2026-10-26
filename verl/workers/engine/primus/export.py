"""Hugging Face export names for Primus modules.

Standard Qwen3-30B-A3B weights keep Megatron-Bridge's native names. Primus-Turbo's
packed expert parameter is storage for the per-expert views and is not an HF tensor.
"""

from __future__ import annotations


def normalize_export_name(name: str) -> str:
    if ".mlp.experts.linear_fc" in name and name.endswith(".weights"):
        raise RuntimeError(
            f"{name} is a Primus-Turbo packed expert parameter. Export the per-expert weight views instead."
        )
    return name
