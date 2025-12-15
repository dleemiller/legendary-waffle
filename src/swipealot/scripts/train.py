"""Main training script for swipe keyboard model."""

import argparse
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from swipealot.config import Config
from swipealot.data import (
    MaskedCollator,
    PairwiseMaskedCollator,
    SwipeDataset,
    ValidationCollator,
    vocab_hash,
)
from swipealot.models import SwipeTransformerModel
from swipealot.training import SwipeLoss, SwipeTrainer


def main():
    parser = argparse.ArgumentParser(description="Train swipe keyboard model")
    parser.add_argument(
        "--config", type=str, default="configs/tiny.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--load-weights",
        type=str,
        default=None,
        help="Path to checkpoint to load weights from (for fine-tuning, starts fresh training)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use small subset of data for debugging"
    )
    parser.add_argument(
        "--char-weights",
        type=str,
        default=None,
        help="Path to precomputed character frequency weights (.pt)",
    )
    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = Config.from_yaml(args.config)

    # Create unique run name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_name = args.config.split("/")[-1].replace(".yaml", "")
    run_name = f"{config_name}_{timestamp}"

    # Update log and checkpoint directories with unique run name
    config.training.log_dir = f"{config.training.log_dir}/{run_name}"
    config.training.checkpoint_dir = f"{config.training.checkpoint_dir}/{run_name}"
    print(f"Run name: {run_name}")
    print(f"Logs: {config.training.log_dir}")
    print(f"Checkpoints: {config.training.checkpoint_dir}")

    # Set device
    device = torch.device(config.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build tokenizer from training data
    print("\nBuilding tokenizer...")
    # Load small sample to build vocabulary
    sample_dataset = SwipeDataset(
        split=config.data.train_split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        dataset_name=config.data.dataset_name,
        max_samples=10000,  # Use subset for vocab building
    )
    tokenizer = sample_dataset.tokenizer
    config.model.vocab_size = tokenizer.vocab_size
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Optional inverse-log character frequency weights (pre-computed only)
    char_freq_weights = None
    if config.training.use_char_freq_weights:
        weights_path = args.char_weights or config.training.char_weights_path
        if not weights_path:
            raise ValueError(
                "use_char_freq_weights is True but no char_weights_path provided. "
                "Run scripts/compute_char_weights.py first."
            )

        print(f"\nLoading precomputed character weights from: {weights_path}")
        loaded = torch.load(weights_path, map_location=device, weights_only=False)
        if isinstance(loaded, dict) and "weights" in loaded:
            char_freq_weights = loaded["weights"]

            # Optional vocab consistency check
            if "vocab" in loaded:
                saved_vocab = loaded["vocab"]
                current_vocab = [tokenizer.id_to_char[i] for i in range(tokenizer.vocab_size)]
                if saved_vocab != current_vocab:
                    raise ValueError(
                        "Tokenizer vocabulary does not match saved weight vocabulary; regenerate weights or align tokenizer"
                    )
            if "vocab_hash" in loaded:
                current_hash = vocab_hash(tokenizer)
                if loaded["vocab_hash"] != current_hash:
                    raise ValueError(
                        "Tokenizer hash mismatch: weights were built with a different tokenizer ordering"
                    )
        else:
            char_freq_weights = loaded

        if char_freq_weights.shape[0] != tokenizer.vocab_size:
            raise ValueError(
                f"Loaded weights vocab ({char_freq_weights.shape[0]}) != tokenizer vocab ({tokenizer.vocab_size})"
            )
        char_freq_weights = char_freq_weights.to(device)
        print(
            f"Loaded weights tensor shape: {tuple(char_freq_weights.shape)} (mean {char_freq_weights.mean():.4f})"
        )
    else:
        print("\nCharacter frequency weights: disabled (use_char_freq_weights=False)")

    # Create datasets
    print("\nLoading datasets...")
    max_samples = 1000 if args.debug else None

    train_dataset = SwipeDataset(
        split=config.data.train_split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=max_samples,
    )

    val_dataset = SwipeDataset(
        split=config.data.val_split,
        max_path_len=config.data.max_path_len,
        max_word_len=config.data.max_char_len,
        tokenizer=tokenizer,
        dataset_name=config.data.dataset_name,
        max_samples=max_samples // 10 if max_samples else 1000,  # Use 1k samples for validation
    )

    # Create data loaders
    print("\nCreating data loaders...")
    if config.training.use_pairwise_masking:
        train_collator = PairwiseMaskedCollator(
            tokenizer=tokenizer,
            mask_path=config.data.mask_path,
            modality_prob=config.training.pairwise_modality_prob,
            zero_text_attention_prob=config.training.pairwise_zero_text_attention_prob,
        )
        # Use unmasked validation for true accuracy metrics
        val_collator = ValidationCollator(tokenizer=tokenizer)
        modality_pct = int(config.training.pairwise_modality_prob * 100)
        inverted_pct = 100 - modality_pct
        print(
            f"Using pairwise masking for training ({inverted_pct}% inverted, {modality_pct}% modality)"
        )
        print("Using unmasked evaluation for validation")
    else:
        train_collator = MaskedCollator(
            tokenizer=tokenizer,
            char_mask_prob=config.data.char_mask_prob,
            path_mask_prob=config.data.path_mask_prob,
            mask_path=config.data.mask_path,
            mask_vocab_only=config.data.mask_vocab_only,
        )
        val_collator = train_collator

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=train_collator,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=val_collator,
        pin_memory=True,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Create model
    print("\nCreating model...")
    model = SwipeTransformerModel(config.model)
    model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create learning rate scheduler with cosine annealing and warmup
    import math

    from torch.optim.lr_scheduler import LambdaLR

    # Calculate total training steps
    total_steps = len(train_loader) * config.training.num_epochs

    def lr_lambda(current_step: int):
        """Linear warmup then cosine annealing to min_lr."""
        if current_step < config.training.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, config.training.warmup_steps))
        # Cosine annealing from 1.0 to min_lr_ratio
        progress = float(current_step - config.training.warmup_steps) / float(
            max(1, total_steps - config.training.warmup_steps)
        )
        min_lr_ratio = config.training.min_learning_rate / config.training.learning_rate
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    scheduler = LambdaLR(optimizer, lr_lambda)
    print(
        f"Learning rate schedule: warmup for {config.training.warmup_steps} steps, then cosine annealing from {config.training.learning_rate:.2e} to {config.training.min_learning_rate:.2e} over {total_steps} total steps"
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    resume_metrics = None
    resume_global_step = None
    resume_scaler_state = None

    if args.resume and args.load_weights:
        raise ValueError(
            "Cannot specify both --resume and --load-weights. Use --resume to continue training, or --load-weights for fine-tuning."
        )

    if args.resume:
        print(f"\nLoading checkpoint from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        resume_metrics = checkpoint.get("metrics")
        resume_global_step = checkpoint.get("global_step")
        resume_scaler_state = checkpoint.get("scaler_state_dict")
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"Resuming from epoch {start_epoch}")

    elif args.load_weights:
        print(f"\nLoading model weights for fine-tuning from: {args.load_weights}")
        checkpoint = torch.load(args.load_weights, map_location=device, weights_only=False)
        # Use strict=False to allow loading when model architecture differs (e.g., missing heads)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint["model_state_dict"], strict=False
        )
        if unexpected_keys:
            print(
                f"Ignored {len(unexpected_keys)} unexpected keys (likely disabled heads): {unexpected_keys[:3]}..."
            )
        if missing_keys:
            print(f"Warning: {len(missing_keys)} missing keys: {missing_keys[:3]}...")
        print(
            "Loaded pretrained weights. Starting fresh training from epoch 0 with new optimizer/scheduler."
        )

    # Create loss function
    loss_fn = SwipeLoss(
        char_weight=config.training.char_loss_weight,
        path_weight=config.training.path_loss_weight,
        length_weight=config.training.length_loss_weight,
        focal_gamma=config.training.focal_gamma if config.training.use_focal_loss else 0.0,
        char_class_weights=char_freq_weights,
        contrastive_weight=config.training.contrastive_weight,
        contrastive_temperature=config.training.contrastive_temperature,
        matryoshka_dims=config.training.matryoshka_dims,
        matryoshka_weights=config.training.matryoshka_weights,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SwipeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config,  # Pass full Config object, not just config.training
        tokenizer=tokenizer,
    )

    # Restore trainer state (global step, scaler, best accuracy) when resuming
    if args.resume:
        if resume_global_step is not None:
            trainer.global_step = resume_global_step
        if resume_metrics and "char_accuracy" in resume_metrics:
            trainer.best_accuracy = float(resume_metrics["char_accuracy"])
        if trainer.scaler and resume_scaler_state is not None:
            trainer.scaler.load_state_dict(resume_scaler_state)

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train(num_epochs=config.training.num_epochs, start_epoch=start_epoch)


if __name__ == "__main__":
    main()
