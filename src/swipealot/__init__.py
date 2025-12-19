"""Swipe keyboard transformer model.

Note: SwipeTransformerModel is now available from swipealot.huggingface
The native PyTorch version has been archived.
"""

__version__ = "0.1.0"

import importlib
from typing import Any

# Keep package import lightweight (important for CLI tools that want to configure HF caches
# before importing transformers / datasets). Public symbols remain available via lazy import.

__all__ = [
    "Config",
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "CharacterTokenizer",
    "SwipeDataset",
    "SwipeTrainer",
    "SwipeLoss",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Configuration
    "Config": ("swipealot.config", "Config"),
    "ModelConfig": ("swipealot.config", "ModelConfig"),
    "DataConfig": ("swipealot.config", "DataConfig"),
    "TrainingConfig": ("swipealot.config", "TrainingConfig"),
    # Data
    "CharacterTokenizer": ("swipealot.data", "CharacterTokenizer"),
    "SwipeDataset": ("swipealot.data", "SwipeDataset"),
    # Training
    "SwipeTrainer": ("swipealot.training", "SwipeTrainer"),
    "SwipeLoss": ("swipealot.training", "SwipeLoss"),
}


def __getattr__(name: str) -> Any:  # pragma: no cover
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + list(_LAZY_EXPORTS.keys()))
