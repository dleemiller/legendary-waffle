"""Utility functions for SwipeTransformer."""

import os
from pathlib import Path
from typing import Any

import torch


def configure_hf_env(
    hf_home: str | Path | None,
    *,
    offline: bool = False,
    disable_telemetry: bool = True,
    overwrite: bool = False,
    set_hub_cache: bool = True,
) -> Path | None:
    """Configure HuggingFace cache and offline mode via environment variables.

    Important: environment variables must be set *before* importing `datasets` / `transformers`
    for cache locations and offline flags to take effect.
    """

    def _set(key: str, value: str) -> None:
        if overwrite:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)

    hf_home_path: Path | None
    if hf_home is not None:
        hf_home_path = Path(hf_home)
        _set("HF_HOME", str(hf_home_path))
        _set("HF_DATASETS_CACHE", str(hf_home_path / "datasets"))
        _set("HF_MODULES_CACHE", str(hf_home_path / "modules"))
        if set_hub_cache:
            _set("HF_HUB_CACHE", str(hf_home_path / "hub"))
    else:
        hf_home_path = None

    if disable_telemetry:
        _set("HF_HUB_DISABLE_TELEMETRY", "1")

    if offline:
        _set("HF_HUB_OFFLINE", "1")
        _set("HF_DATASETS_OFFLINE", "1")
        _set("TRANSFORMERS_OFFLINE", "1")

    return hf_home_path


def extract_character_logits(
    char_logits: torch.Tensor, path_len: int, char_len: int
) -> torch.Tensor:
    """
    Extract character prediction logits from full sequence.

    The sequence structure is: [CLS] + path + [SEP] + chars
    This function extracts just the character portion.

    Args:
        char_logits: Full sequence logits [batch, full_seq_len, vocab_size]
        path_len: Length of path sequence
        char_len: Length of character sequence

    Returns:
        Character logits [batch, char_len, vocab_size]
    """
    char_start = 1 + path_len + 1  # Skip [CLS], path, [SEP]
    char_end = char_start + char_len
    return char_logits[:, char_start:char_end, :]


def batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """
    Transfer batch tensors to device.

    Args:
        batch: Dictionary containing batch data
        device: Target device

    Returns:
        Batch dictionary with tensors moved to device
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
