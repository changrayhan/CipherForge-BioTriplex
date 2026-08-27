"""
PeftModel Key Remapping Utilities.

Solves the peft-0.19.1 + Llama-3.1-8B-Instruct key mismatch bug where
the trainer saves adapter keys as:

    base_model.model.layers.{i}.self_attn.q_proj.lora_A.weight

but PeftModel.from_pretrained resolves them under:

    base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.default.weight

All three locations that independently re-implement this fix are consolidated here.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterator, MutableMapping

import torch

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Constants                                                                  #
# --------------------------------------------------------------------------- #
PREFIX_LEGACY = "base_model.model.layers."
PREFIX_LIVE   = "base_model.model.model.layers."
ADAPTER_SUFFIX = ".default"


# --------------------------------------------------------------------------- #
#  Core remapping                                                             #
# --------------------------------------------------------------------------- #
def remap_lora_keys(
    state_dict: MutableMapping[str, torch.Tensor],
    *,
    in_place: bool = False,
) -> Dict[str, torch.Tensor]:
    """Strip extra ``model.`` prefix and add ``.default`` adapter-name suffix.

    Transforms disk keys → live module paths:

        base_model.model.layers.N.self_attn.q_proj.lora_A.weight
      → base_model.model.model.layers.N.self_attn.q_proj.lora_A.default.weight

    Args:
        state_dict: Mapping from key names to tensors (e.g. a safetensors
            file loaded with ``safetensors.torch.load_file``).
        in_place: If True, modify ``state_dict`` in place and return it.
            If False (default), return a new dict leaving the input unchanged.

    Returns:
        Remapped state dict.
    """
    result = state_dict if in_place else {}
    target = result if in_place else {}

    for key, value in state_dict.items():
        new_key = key

        # Step 1 — fix the doubled "model.model." nesting
        if PREFIX_LEGACY in new_key:
            new_key = new_key.replace(PREFIX_LEGACY, PREFIX_LIVE)

        # Step 2 — add ".default" adapter suffix to LoRA weight keys
        if "lora_A.weight" in new_key:
            new_key = new_key.replace("lora_A.weight", "lora_A" + ADAPTER_SUFFIX + ".weight")
        if "lora_B.weight" in new_key:
            new_key = new_key.replace("lora_B.weight", "lora_B" + ADAPTER_SUFFIX + ".weight")

        target[new_key] = value

    if target is not state_dict:
        logger.debug("Remapped %d keys", len(target))

    return target


def remap_with_fallbacks(
    state_dict: MutableMapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Remap with multiple fallback prefix variants.

    Tries in order:
      1. ``base_model.model.layers.`` → ``base_model.model.model.layers.`` + add ``.default``
      2. ``base_model.model.model.layers.`` → ``base_model.model.model.layers.`` + add ``.default``

    The second variant is needed when the disk keys already have the correct
    ``model.layers.`` prefix but still need the ``.default`` suffix.
    """
    # Fast path: most common case
    result = remap_lora_keys(state_dict)
    if any(PREFIX_LEGACY in k for k in result):
        # At least one key still has the legacy prefix — try variant 2
        result2 = _remap_variant2(state_dict)
        # Merge: prefer result2 for keys that changed, result for keys unchanged
        merged = dict(result)
        merged.update(result2)
        result = merged
    return result


def _remap_variant2(
    state_dict: MutableMapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Add .default suffix even when prefix already has ``model.model.``."""
    result = {}
    for key, value in state_dict.items():
        new_key = key
        if "lora_A.weight" in new_key and "lora_A.default.weight" not in new_key:
            new_key = new_key.replace("lora_A.weight", "lora_A" + ADAPTER_SUFFIX + ".weight")
        if "lora_B.weight" in new_key and "lora_B.default.weight" not in new_key:
            new_key = new_key.replace("lora_B.weight", "lora_B" + ADAPTER_SUFFIX + ".weight")
        result[new_key] = value
    return result


# --------------------------------------------------------------------------- #
#  Conv1D orientation fix                                                     #
# --------------------------------------------------------------------------- #
def remap_conv1d_to_linear(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Fix Conv1D → Linear weight orientation for Llama compatibility.

    Some checkpoints store LoRA B weights in ``(r, in_features)`` Conv1D layout
    but the model expects ``(in_features, r)`` Linear layout.  This swaps the
    first two dimensions of any tensor that looks like a LoRA B weight.

    Detection heuristic: key contains ``lora_B`` and tensor is 2-D with
    the first dimension ≤ 64 (LoRA rank typical range).
    """
    result = {}
    for key, value in state_dict.items():
        if "lora_B" in key and value.ndim == 2 and value.shape[0] <= 64:
            logger.debug("Transposing Conv1D lora_B key: %s  %s → %s",
                         key, value.shape, tuple(reversed(value.shape)))
            result[key] = value.T
        else:
            result[key] = value
    return result


# --------------------------------------------------------------------------- #
#  Convenience loaders                                                        #
# --------------------------------------------------------------------------- #
def load_and_remap_safetensors(
    path: str,
    *,
    remap_fallbacks: bool = True,
    fix_conv1d: bool = True,
) -> Dict[str, torch.Tensor]:
    """Load a safetensors file and remap LoRA keys.

    Args:
        path: Path to a ``.safetensors`` file.
        remap_fallbacks: If True, use ``remap_with_fallbacks``; else ``remap_lora_keys``.
        fix_conv1d: If True, apply Conv1D orientation fix.

    Returns:
        Remapped state dict.
    """
    from safetensors.torch import load_file

    sd = load_file(path, device="cpu")
    sd = remap_with_fallbacks(sd) if remap_fallbacks else remap_lora_keys(sd)
    if fix_conv1d:
        sd = remap_conv1d_to_linear(sd)
    return sd


def load_adapter_and_apply_to_model(
    model: "torch.nn.Module",
    adapter_path: str,
    *,
    adapter_name: str = "default",
) -> None:
    """Load a LoRA adapter checkpoint and inject it into a model.

    Handles the key remapping automatically so the LoRA weights land in the
    correct ``.lora_A.default.weight`` / ``.lora_B.default.weight`` locations
    that ``PeftModel`` uses at runtime.

    Args:
        model: Live ``PeftModel`` instance.
        adapter_path: Directory containing ``adapter_model.safetensors``.
        adapter_name: Name to register the adapter under.
    """
    import warnings
    from pathlib import Path

    adapter_path = Path(adapter_path)
    safetensor_path = adapter_path / "adapter_model.safetensors"
    if not safetensor_path.exists():
        raise FileNotFoundError(f"adapter_model.safetensors not found in {adapter_path}")

    sd = load_and_remap_safetensors(str(safetensor_path))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        logger.warning("Missing keys during adapter injection (%d): %s", len(missing), missing[:5])
    if unexpected:
        logger.debug("Unexpected keys ignored (%d): %s", len(unexpected), unexpected[:5])
