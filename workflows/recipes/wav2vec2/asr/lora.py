"""
Minimal LoRA implementation compatible with torch.nn.Linear and fairseq2.nn.Linear.

Usage:
    from lora import LoraConfig, get_lora_model, print_trainable_parameters

    config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"])
    model = get_lora_model(model, config)
    print_trainable_parameters(model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LoraConfig:
    r: int
    lora_alpha: float
    target_modules: list[str]
    lora_dropout: float = 0.0


def get_linear_dims(layer: nn.Module) -> tuple[int, int]:
    """Return (in_features, out_features) for torch or fairseq2 Linear layers."""
    if hasattr(layer, "in_features") and hasattr(layer, "out_features"):
        return layer.in_features, layer.out_features
    if hasattr(layer, "input_dim") and hasattr(layer, "output_dim"):
        return layer.input_dim, layer.output_dim
    raise TypeError(
        f"{type(layer).__name__} has no recognisable in/out dimension attributes. "
        "Expected 'in_features'/'out_features' (torch) or 'input_dim'/'output_dim' (fairseq2)."
    )


class LoraLinear(nn.Module):
    """
    LoRA adapter wrapping any linear layer.

    Adds the low-rank update  Δy = (x @ A.T @ B.T) * (alpha / r)  on top of
    the frozen base layer output.  B is zero-initialised so the adapter is a
    no-op at the start of training.
    """

    def __init__(self, base: nn.Module, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.scaling = alpha / r

        in_dim, out_dim = get_linear_dims(base)

        self.lora_A = nn.Parameter(torch.zeros(r, in_dim, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return result + lora_out


def get_lora_model(model: nn.Module, config: LoraConfig) -> nn.Module:
    """
    Freeze the entire model, then replace every module whose leaf name appears
    in config.target_modules with a LoraLinear wrapper.

    Modifies model in-place and returns it.
    """
    for param in model.parameters():
        param.requires_grad = False

    target_set = set(config.target_modules)
    replaced = []

    for name, module in list(model.named_modules()):
        if not name:
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in target_set:
            continue

        *parent_parts, attr = name.split(".")
        parent = model
        for part in parent_parts:
            parent = getattr(parent, part)

        wrapped = LoraLinear(
            module,
            r=config.r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        )
        setattr(parent, attr, wrapped)
        replaced.append(name)

    if not replaced:
        raise ValueError(
            f"No modules matched target_modules={config.target_modules}. "
            f"Available leaf names: {sorted({n.rsplit('.', 1)[-1] for n, _ in model.named_modules() if n})}"
        )

    return model


def print_trainable_parameters(model: nn.Module) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable:,} || "
        f"total params: {total:,} || "
        f"trainable%: {100 * trainable / total:.4f}%"
    )
