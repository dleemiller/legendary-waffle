"""Evaluation script for swipe keyboard model."""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from swipealot.data import MaskedCollator, SwipeDataset
from swipealot.models import SwipeTransformerModel
from swipealot.training import CharacterAccuracy, WordAccuracy
from swipealot.utils import batch_to_device, extract_character_logits


def evaluate_model(model, test_loader, tokenizer, device):
    """
    Evaluate model on test set.

    Args:
        model: Trained model
        test_loader: Test data loader
        tokenizer: Character tokenizer
        device: Device to run on

    Returns:
        Dictionary of metrics
    """
    model.eval()

    char_accuracy = CharacterAccuracy()
    word_accuracy = WordAccuracy(tokenizer)

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move to device
            batch = batch_to_device(batch, device)

            # Forward pass
            outputs = model(
                path_coords=batch["path_coords"],
                char_tokens=batch["char_tokens"],
                attention_mask=batch["attention_mask"],
            )

            # Extract character predictions
            char_logits = outputs["char_logits"]
            path_len = batch["path_coords"].shape[1]
            char_logits_subset = extract_character_logits(
                char_logits, path_len, batch["char_tokens"].shape[1]
            )

            # Get predictions
            pred_tokens = char_logits_subset.argmax(dim=-1)

            # Decode predictions
            for i, pred in enumerate(pred_tokens):
                pred_word = tokenizer.decode(pred.cpu().tolist())
                target_word = batch["words"][i]
                all_predictions.append(pred_word.strip())
                all_targets.append(target_word.strip())

            # Update metrics (only on non-padding positions)
            # Use the masking-aware labels so padding is ignored
            char_labels = batch["char_labels"]
            char_accuracy.update(char_logits_subset, char_labels)
            word_accuracy.update(char_logits_subset, batch["words"])

    # Compute final metrics
    char_acc = char_accuracy.compute()
    word_acc = word_accuracy.compute()

    results = {
        "char_accuracy": char_acc,
        "word_accuracy": word_acc,
        "num_samples": len(all_predictions),
    }

    return results, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate swipe keyboard model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (train/validation/test)",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument("--show_examples", action="store_true", help="Show example predictions")
    args = parser.parse_args()

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build tokenizer
    print("\nBuilding tokenizer...")
    sample_dataset = SwipeDataset(
        split="train",
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        dataset_name=config.data.dataset_name,
        max_samples=10000,
    )
    tokenizer = sample_dataset.tokenizer

    # Create test dataset
    print(f"\nLoading {args.split} dataset...")
    test_dataset = SwipeDataset(
        split=args.split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=args.num_samples,
    )

    # For evaluation we leave swipe paths untouched and mask all characters.
    collator = MaskedCollator(
        tokenizer=tokenizer,
        char_mask_prob=1.0,  # mask every character token (including EOS)
        path_mask_prob=0.0,  # never mask path points
        mask_path=False,  # keep paths unmasked
    )

    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collator
    )

    # Create model
    print("\nCreating model...")
    model = SwipeTransformerModel(checkpoint["config"].model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Evaluate
    print("\nEvaluating...")
    results, predictions, targets = evaluate_model(model, test_loader, tokenizer, device)

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"Number of samples: {results['num_samples']}")
    print(f"Character Accuracy: {results['char_accuracy']:.4f}")
    print(f"Word Accuracy: {results['word_accuracy']:.4f}")
    print("=" * 60)

    # Show examples
    if args.show_examples:
        print("\nExample Predictions:")
        print("-" * 60)
        for i in range(min(20, len(predictions))):
            correct = "✓" if predictions[i] == targets[i] else "✗"
            print(f"{correct} Target: '{targets[i]}' | Predicted: '{predictions[i]}'")


if __name__ == "__main__":
    main()
