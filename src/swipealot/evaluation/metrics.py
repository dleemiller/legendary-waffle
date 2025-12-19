"""Backwards-compatible re-exports for evaluation functions/types.

The original monolithic module was split into smaller per-task modules for readability.
Keep this as a stable import path for scripts/tests.
"""

from __future__ import annotations

from .length import LengthEvalResult, evaluate_length_dataset
from .masked_text import (
    BlindReconstructionMetrics,
    FullReconstructionMetrics,
    MaskedEvalResult,
    MaskedPredictionMetrics,
    evaluate_blind_reconstruction_two_pass,
    evaluate_full_reconstruction_100pct,
    evaluate_masked_prediction_30pct,
    evaluate_masked_tokens,
)
from .path import PathReconstructionMetrics, evaluate_path_reconstruction_masked_mse

__all__ = [
    # masked text
    "MaskedEvalResult",
    "MaskedPredictionMetrics",
    "FullReconstructionMetrics",
    "BlindReconstructionMetrics",
    "evaluate_masked_tokens",
    "evaluate_masked_prediction_30pct",
    "evaluate_full_reconstruction_100pct",
    "evaluate_blind_reconstruction_two_pass",
    # length
    "LengthEvalResult",
    "evaluate_length_dataset",
    # path
    "PathReconstructionMetrics",
    "evaluate_path_reconstruction_masked_mse",
]
