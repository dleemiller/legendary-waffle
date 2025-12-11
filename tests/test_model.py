"""Quick test to verify model can be instantiated and run."""

import torch

from swipealot.config import Config
from swipealot.models import SwipeTransformerModel
from swipealot.training import SwipeLoss


def test_model():
    """Test basic model functionality."""
    print("Testing model instantiation and forward pass...")

    # Create simple config
    config = Config()
    config.model.vocab_size = 100

    # Create model
    model = SwipeTransformerModel(config.model)
    print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dummy inputs
    batch_size = 2
    path_coords = torch.randn(batch_size, config.model.max_path_len, 3)
    char_tokens = torch.randint(0, 100, (batch_size, config.model.max_char_len))
    attention_mask = torch.ones(
        batch_size, 1 + config.model.max_path_len + 1 + config.model.max_char_len
    )

    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(path_coords, char_tokens, attention_mask)

    print("✓ Forward pass successful")
    print(f"  - char_logits shape: {outputs['char_logits'].shape}")
    if "path_coords_pred" in outputs:
        print(f"  - path_coords_pred shape: {outputs['path_coords_pred'].shape}")

    # Test loss
    loss_fn = SwipeLoss()
    # Create char_labels: some positions are -100 (ignore), others are valid token IDs
    char_labels = torch.randint(0, 100, (batch_size, config.model.max_char_len))
    # Randomly set some positions to -100 (ignored in loss)
    mask = torch.rand(batch_size, config.model.max_char_len) > 0.85
    char_labels[mask] = -100

    batch = {
        "path_coords": path_coords,
        "char_tokens": char_tokens,
        "char_labels": char_labels,
        "path_labels": torch.randn(batch_size, config.model.max_path_len, 3).clamp(
            0, 1
        ),  # [0, 1] range
        "path_mask_indices": torch.randint(0, 2, (batch_size, config.model.max_path_len)),
    }
    losses = loss_fn(outputs, batch)
    print("✓ Loss computation successful")
    print(f"  - total_loss: {losses['total_loss'].item():.4f}")
    print(f"  - char_loss: {losses['char_loss'].item():.4f}")
    if "path_loss" in losses:
        print(f"  - path_loss: {losses['path_loss'].item():.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
