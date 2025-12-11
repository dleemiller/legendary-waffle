"""SwipeDataset and coordinate preprocessing utilities."""

from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

from .tokenizer import CharacterTokenizer


def normalize_coordinates(
    data_points: list[dict], canvas_width: float, canvas_height: float
) -> list[dict]:
    """
    Normalize swipe coordinates and timestamps.

    Args:
        data_points: List of dicts with 'x', 'y', 't' keys
        canvas_width: Canvas width for normalization
        canvas_height: Canvas height for normalization

    Returns:
        List of normalized coordinate dicts
    """
    if not data_points:
        return []

    # Extract timestamps for normalization
    timestamps = [p["t"] for p in data_points]
    t_min = min(timestamps)
    t_max = max(timestamps)
    t_range = t_max - t_min if t_max > t_min else 1.0

    normalized = []
    for point in data_points:
        # x and y are already normalized to [0,1] in the dataset
        # But sometimes they go slightly outside bounds, so clamp them
        x_norm = max(0.0, min(1.0, point["x"]))
        y_norm = max(0.0, min(1.0, point["y"]))

        # Normalize timestamp to [0, 1]
        t_norm = (point["t"] - t_min) / t_range

        normalized.append({"x": x_norm, "y": y_norm, "t": t_norm})

    return normalized


def rotate_to_portrait(data_points: list[dict], orientation: str) -> list[dict]:
    """
    Rotate normalized coordinates into a canonical portrait frame.

    Observed orientation values in the dataset: "portrait-primary" and
    "landscape-primary" (others possible per ScreenOrientation API:
    "portrait-secondary", "landscape-secondary"). We map them explicitly:
      - portrait-primary:        no rotation
      - portrait-secondary:      180° rotation
      - landscape-primary:       90° counter-clockwise (device rotated CW)
      - landscape-secondary:     90° clockwise (device rotated CCW)
    Unknown values fall back to no rotation.
    """
    if not data_points:
        return []

    orient = (orientation or "").lower()

    # Precompute rotation functions on normalized coordinates
    def rot_none(pt):
        return pt["x"], pt["y"]

    def rot_180(pt):
        return 1.0 - pt["x"], 1.0 - pt["y"]

    def rot_cw(pt):
        # 90° clockwise: (x, y) -> (y, 1 - x)
        return pt["y"], 1.0 - pt["x"]

    def rot_ccw(pt):
        # 90° counter-clockwise: (x, y) -> (1 - y, x)
        return 1.0 - pt["y"], pt["x"]

    if orient == "portrait-primary":
        rotate_fn = rot_none
    elif orient == "portrait-secondary":
        rotate_fn = rot_180
    elif orient == "landscape-primary":
        # Device rotated clockwise; rotate CCW to restore portrait
        rotate_fn = rot_ccw
    elif orient == "landscape-secondary":
        # Device rotated counter-clockwise; rotate CW to restore portrait
        rotate_fn = rot_cw
    else:
        rotate_fn = rot_none

    rotated = []
    for pt in data_points:
        x_new, y_new = rotate_fn(pt)
        rotated.append({"x": float(x_new), "y": float(y_new), "t": pt["t"]})

    return rotated


def sample_path_points(data_points: list[dict], max_len: int) -> tuple:
    """
    Sample or pad path points to fixed length using linear interpolation.

    Args:
        data_points: List of coordinate dicts
        max_len: Target length

    Returns:
        Tuple of (sampled_points, mask) where mask indicates valid (1) vs padding (0)
    """
    num_points = len(data_points)

    if num_points == max_len:
        points = data_points
        mask = [1] * max_len
    elif num_points < max_len:
        # Pad with zeros
        points = data_points + [{"x": 0.0, "y": 0.0, "t": 0.0}] * (max_len - num_points)
        mask = [1] * num_points + [0] * (max_len - num_points)
    else:
        # Downsample using linear interpolation
        # Extract coordinates as arrays
        x_coords = np.array([p["x"] for p in data_points])
        y_coords = np.array([p["y"] for p in data_points])
        t_coords = np.array([p["t"] for p in data_points])

        # Original indices (parameter for interpolation)
        original_indices = np.arange(num_points)

        # Target indices for interpolation (evenly spaced)
        target_indices = np.linspace(0, num_points - 1, max_len)

        # Interpolate each coordinate independently
        x_interp = np.interp(target_indices, original_indices, x_coords)
        y_interp = np.interp(target_indices, original_indices, y_coords)
        t_interp = np.interp(target_indices, original_indices, t_coords)

        # Reconstruct points
        points = [
            {"x": float(x), "y": float(y), "t": float(t)}
            for x, y, t in zip(x_interp, y_interp, t_interp)
        ]
        mask = [1] * max_len

    # Convert to numpy arrays
    coords = np.array([[p["x"], p["y"], p["t"]] for p in points], dtype=np.float32)
    mask = np.array(mask, dtype=np.int64)

    return coords, mask


class SwipeDataset(Dataset):
    """Dataset for swipe keyboard data."""

    def __init__(
        self,
        split: str = "train",
        max_path_len: int = 64,
        max_word_len: int = 38,
        tokenizer: CharacterTokenizer | None = None,
        dataset_name: str = "futo-org/swipe.futo.org",
        max_samples: int | None = None,
    ):
        """
        Initialize swipe dataset.

        Args:
            split: Dataset split ('train', 'validation', 'test')
            max_path_len: Maximum path length (points will be sampled/padded)
            max_word_len: Maximum word length in characters
            tokenizer: Character tokenizer (will be created if None)
            dataset_name: HuggingFace dataset name
            max_samples: Optional limit on number of samples (for debugging)
        """
        self.max_path_len = max_path_len
        self.max_word_len = max_word_len

        # Load dataset
        print(f"Loading dataset: {dataset_name}, split: {split}")
        if max_samples:
            self.dataset = load_dataset(dataset_name, split=f"{split}[:{max_samples}]")
        else:
            self.dataset = load_dataset(dataset_name, split=split)
        print(f"Loaded {len(self.dataset)} samples")

        # Initialize tokenizer
        if tokenizer is None:
            print("Building tokenizer from dataset...")
            self.tokenizer = CharacterTokenizer.from_dataset(self.dataset)
            print(f"Vocabulary size: {self.tokenizer.vocab_size}")
        else:
            self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.dataset[idx]

        # Process swipe path
        data_points = sample["data"]
        canvas_width = sample.get("canvas_width", 1.0)
        canvas_height = sample.get("canvas_height", 1.0)
        orientation = sample.get("orientation", "portrait")

        # Normalize coordinates
        normalized_points = normalize_coordinates(data_points, canvas_width, canvas_height)

        # XX Rotate into a canonical portrait frame if needed
        # data is already scaled and rotated
        # normalized_points = rotate_to_portrait(normalized_points, orientation)

        # Sample/pad to fixed length
        path_coords, path_mask = sample_path_points(normalized_points, self.max_path_len)

        # Process word
        word = sample["word"]
        char_tokens = self.tokenizer.encode(word)

        # Add EOS token after the word
        char_tokens = char_tokens + [self.tokenizer.eos_token_id]

        # Pad character tokens (EOS + padding)
        if len(char_tokens) < self.max_word_len:
            char_tokens = char_tokens + [self.tokenizer.pad_token_id] * (
                self.max_word_len - len(char_tokens)
            )
        else:
            # If word+EOS is too long, truncate word and ensure EOS at the end
            char_tokens = char_tokens[: self.max_word_len - 1] + [self.tokenizer.eos_token_id]

        # Create character mask (1 for real tokens + EOS, 0 for padding)
        char_mask = [1 if token != self.tokenizer.pad_token_id else 0 for token in char_tokens]

        return {
            "path_coords": torch.tensor(path_coords, dtype=torch.float32),  # [max_path_len, 3]
            "char_tokens": torch.tensor(char_tokens, dtype=torch.long),  # [max_word_len]
            "path_mask": torch.tensor(path_mask, dtype=torch.long),  # [max_path_len]
            "char_mask": torch.tensor(char_mask, dtype=torch.long),  # [max_word_len]
            "word": word,  # Keep original word for evaluation
        }
