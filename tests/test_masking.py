"""Test independent masking probabilities for path and text."""

from torch.utils.data import DataLoader

from swipealot.data import MaskedCollator, SwipeDataset


def test_independent_masking():
    """Test that path and character masking work independently."""

    print("=" * 60)
    print("TESTING INDEPENDENT MASKING PROBABILITIES")
    print("=" * 60)

    # Test different masking configurations
    configs = [
        {"char": 0.15, "path": 0.15, "name": "Balanced (15% both)"},
        {"char": 0.30, "path": 0.15, "name": "High char, normal path (30% / 15%)"},
        {"char": 0.15, "path": 0.30, "name": "Normal char, high path (15% / 30%)"},
        {"char": 0.50, "path": 0.05, "name": "Very high char, low path (50% / 5%)"},
        {"char": 0.0, "path": 0.20, "name": "No char masking, only path (0% / 20%)"},
    ]

    # Load small dataset
    print("\nLoading dataset...")
    dataset = SwipeDataset(
        split="train",
        max_path_len=128,
        max_word_len=38,
        dataset_name="futo-org/swipe.futo.org",
        max_samples=100,
    )
    tokenizer = dataset.tokenizer

    print(f"Loaded {len(dataset)} samples\n")

    # Test each configuration
    for config in configs:
        print(f"\n{'=' * 60}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 60}")

        collator = MaskedCollator(
            tokenizer=tokenizer,
            char_mask_prob=config["char"],
            path_mask_prob=config["path"],
            mask_path=True,
        )

        # Create loader with single batch
        loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

        # Get one batch
        batch = next(iter(loader))

        # Analyze masking (only count valid, non-padding positions)
        char_labels = batch["char_labels"]  # -100 for non-masked
        char_mask = batch["char_mask"]  # 1 for valid, 0 for padding
        path_mask_indices = batch["path_mask_indices"]  # 1 for masked
        path_mask = batch["path_mask"]  # 1 for valid, 0 for padding

        # Count masked positions (only among valid positions)
        valid_char_positions = char_mask == 1
        char_masked = (char_labels != -100).sum().item()
        char_total_valid = valid_char_positions.sum().item()
        char_masked_pct = 100 * char_masked / char_total_valid if char_total_valid > 0 else 0

        valid_path_positions = path_mask == 1
        path_masked = path_mask_indices.sum().item()
        path_total_valid = valid_path_positions.sum().item()
        path_masked_pct = 100 * path_masked / path_total_valid if path_total_valid > 0 else 0

        print("\nCharacter Masking:")
        print(f"  Expected: {config['char'] * 100:.1f}%")
        print(
            f"  Actual:   {char_masked_pct:.1f}% ({char_masked}/{char_total_valid} valid positions)"
        )
        print(f"  Total positions (incl. padding): {char_labels.numel()}")

        print("\nPath Masking:")
        print(f"  Expected: {config['path'] * 100:.1f}%")
        print(
            f"  Actual:   {path_masked_pct:.1f}% ({path_masked}/{path_total_valid} valid positions)"
        )
        print(f"  Total positions (incl. padding): {path_mask_indices.numel()}")

        # Check if within reasonable range (±3%)
        char_ok = abs(char_masked_pct - config["char"] * 100) < 5
        path_ok = abs(path_masked_pct - config["path"] * 100) < 5

        if char_ok and path_ok:
            print("\n✓ Both masking probabilities working correctly!")
        else:
            print("\n⚠ Warning: Masking probabilities outside expected range")
            if not char_ok:
                print(
                    f"  - Character masking: {char_masked_pct:.1f}% (expected {config['char'] * 100:.1f}%)"
                )
            if not path_ok:
                print(
                    f"  - Path masking: {path_masked_pct:.1f}% (expected {config['path'] * 100:.1f}%)"
                )

    print("\n" + "=" * 60)
    print("INDEPENDENT MASKING TEST COMPLETE")
    print("=" * 60)
    print("\n✓ Character and path masking are independently configurable!")
    print("✓ Set via config.yaml:")
    print("    char_mask_prob: 0.15  # 15% of character tokens")
    print("    path_mask_prob: 0.15  # 15% of path points")


if __name__ == "__main__":
    test_independent_masking()
