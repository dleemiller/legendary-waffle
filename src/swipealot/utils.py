"""Utility functions for SwipeTransformer."""

from typing import Any

import torch


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
