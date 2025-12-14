from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from pathlib import Path

from load_and_compute_loss import load_llm_asr_300m

try:
    # Fairseq2 Linear (used inside this repo)
    from fairseq2.nn import Linear as FsLinear  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    FsLinear = nn.Linear  # type: ignore


@dataclass
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout_p: float = 0.0
    # Module name fragments to decide where to inject LoRA.
    # You can override this from your training script.
    target_keywords: Sequence[str] = (
        "self_attn",
        "ffn",
    )


class LoRALinear(nn.Module):
    """
    Simple LoRA wrapper around a Linear layer.

    Forward: y = base(x) + scale * B(A(dropout(x)))
    where A: in_dim -> r, B: r -> out_dim
    """

    def __init__(
        self,
        base_linear: nn.Module,
        r: int,
        alpha: float,
        dropout_p: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(base_linear, (nn.Linear, FsLinear)):
            raise TypeError(
                f"LoRALinear expects a Linear-like module, got {type(base_linear)}"
            )

        if r <= 0:
            raise ValueError("LoRA rank r must be > 0.")

        self.base = base_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.dropout = nn.Dropout(p=dropout_p) if dropout_p > 0.0 else nn.Identity()

        in_features = base_linear.input_dim if hasattr(base_linear, "input_dim") else base_linear.in_features  # type: ignore[attr-defined]
        out_features = base_linear.output_dim if hasattr(base_linear, "output_dim") else base_linear.out_features  # type: ignore[attr-defined]

        # LoRA matrices (A: in->r, B: r->out).
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        # Initialize following standard LoRA convention:
        # - A ~ N(0, 0.01)
        # - B = 0
        nn.init.normal_(self.lora_A.weight, std=0.01)
        nn.init.zeros_(self.lora_B.weight)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return base_out + lora_out


def _get_parent_module(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    """
    Given a dotted module name from model.named_modules(), return its parent
    module and the final attribute name.
    """
    parts = module_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora(
    model: nn.Module,
    *,
    config: LoraConfig | None = None,
    freeze_base: bool = True,
    extra_target_keywords: Iterable[str] | None = None,
) -> None:
    """
    In-place: walk the model and wrap selected Linear layers with LoRALinear.

    Selection is done heuristically based on module name fragments
    (e.g. "self_attn", "ffn" for the LLaMA decoder in this repo).

    After calling this, you typically want to construct your optimizer from
    the parameters with requires_grad=True only, e.g.:

        optimizer = AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
        )
    """
    if config is None:
        config = LoraConfig()

    # If requested, freeze all parameters in the model first.
    # New LoRA parameters created below will be trainable by default.
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    target_keywords = list(config.target_keywords)
    if extra_target_keywords is not None:
        target_keywords.extend(extra_target_keywords)

    # Work on a snapshot of named_modules to avoid modifying while iterating.
    named_modules = list(model.named_modules())

    for name, module in named_modules:
        if not isinstance(module, (nn.Linear, FsLinear)):
            continue

        if not any(kw in name for kw in target_keywords):
            continue

        parent, child_name = _get_parent_module(model, name)

        wrapped = LoRALinear(
            base_linear=module,
            r=config.r,
            alpha=config.alpha,
            dropout_p=config.dropout_p,
            freeze_base=freeze_base,
        )

        setattr(parent, child_name, wrapped)


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    """
    Convenience helper to get all LoRA trainable parameters from a model.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # Only return parameters that are actually trainable.
            # When `freeze_base=True`, this excludes the wrapped base Linear
            # weights and biases, and only returns LoRA matrices.
            for p in module.parameters():
                if p.requires_grad:
                    yield p

def load_llm_asr_300m_with_lora(
    lora_checkpoint_path: str | Path,
    *,
    base_checkpoint_path: str | Path,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    # 1) Load the original base model + tokenizer
    model, tokenizer = load_llm_asr_300m(
        checkpoint_path=base_checkpoint_path,
        device=device,
        dtype=dtype,
    )

    # 2) Inject LoRA exactly as in training
    lora_config = LoraConfig(
        r=LORA_R,
        alpha=LORA_ALPHA,
        dropout_p=LORA_DROPOUT_P,
        target_keywords=LORA_TARGET_KEYWORDS,
    )
    inject_lora(model, config=lora_config, freeze_base=True)

    # 3) Load LoRA‑finetuned weights
    ckpt = torch.load(lora_checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict") or ckpt
    model.load_state_dict(state_dict, strict=True)  # use strict=False if you save LoRA‑only

    model.eval()
    return model, tokenizer


LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT_P = 0.1
LORA_TARGET_KEYWORDS = ("llama_decoder.layers",)
#USAGE: model, tokenizer = load_llm_asr_300m_with_lora(lora_checkpoint_path, base_checkpoint_path=base_checkpoint_path, device=device, dtype=dtype) 