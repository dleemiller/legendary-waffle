"""Evaluate a HuggingFace SwipeALot checkpoint via the customer-facing APIs.

This script intentionally loads the checkpoint using:
- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`

so evaluation reflects how the model will be used after export.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from swipealot.evaluation.metrics import (
    evaluate_blind_reconstruction_two_pass,
    evaluate_full_reconstruction_100pct,
    evaluate_length_dataset,
    evaluate_masked_prediction_30pct,
    evaluate_masked_tokens,
    evaluate_path_reconstruction_masked_mse,
)
from swipealot.utils import configure_hf_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a SwipeALot HuggingFace checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a HuggingFace checkpoint directory (e.g. .../checkpoint-6000 or .../final)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="futo-org/swipe.futo.org",
        help="HuggingFace dataset name (default: futo-org/swipe.futo.org)",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split (default: test)")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples to evaluate (0 = full split; default: 5000)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF cache root (respects default HF env when omitted)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable HF offline mode (sets HF_HUB_OFFLINE / HF_DATASETS_OFFLINE / TRANSFORMERS_OFFLINE)",
    )
    parser.add_argument(
        "--mask-mode",
        choices=["random", "all", "none"],
        default="random",
        help="Text masking mode (default: random)",
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.3,
        help="Mask probability for masked-text metrics when --mask-mode=random (default: 0.3)",
    )
    parser.add_argument(
        "--path-mse-dims",
        type=str,
        default="0,1",
        help="Comma-separated path dims for masked MSE in --model-card (default: 0,1 for x/y)",
    )
    parser.add_argument(
        "--mask-eos",
        action="store_true",
        help="Also mask EOS tokens (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=0,
        help="Print up to N word mismatches with masked character diffs (default: 0)",
    )
    parser.add_argument(
        "--show-top-errors",
        type=int,
        default=0,
        help="Print top N masked-token error pairs + most-masked tokens (default: 0)",
    )
    parser.add_argument(
        "--model-card",
        action="store_true",
        help="Print model-card style metrics table (masked pred, full recon, blind recon, length, path MSE)",
    )
    parser.add_argument(
        "--skip-length",
        action="store_true",
        help="Skip length evaluation (default: False)",
    )

    args = parser.parse_args()

    if args.hf_home is not None or args.offline:
        configure_hf_env(
            args.hf_home,
            offline=bool(args.offline),
            disable_telemetry=args.hf_home is not None,
            overwrite=False,
            set_hub_cache=False,
        )

    from datasets import load_dataset
    from transformers import AutoModel, AutoProcessor

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(str(ckpt), trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)

    if int(args.n_samples) <= 0:
        dataset = load_dataset(args.dataset, split=str(args.split))
    else:
        dataset = load_dataset(args.dataset, split=f"{args.split}[:{args.n_samples}]")

    if args.model_card:
        if args.path_mse_dims.strip():
            mse_dims = [int(x) for x in args.path_mse_dims.split(",") if x.strip() != ""]
        else:
            mse_dims = None

        masked30 = evaluate_masked_prediction_30pct(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            mask_prob=float(args.mask_prob),
            seed=args.seed,
        )
        full100 = evaluate_full_reconstruction_100pct(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        blind = evaluate_blind_reconstruction_two_pass(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        length = evaluate_length_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        path = evaluate_path_reconstruction_masked_mse(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            mask_prob=float(args.mask_prob),
            mse_dims=mse_dims,
            seed=args.seed,
        )

        print("\n## Performance Metrics\n")
        print(f"Evaluated on {len(dataset):,} {args.split} samples:\n")
        print("| Task | Metric | Score |")
        print("|------|--------|-------|")
        print(f"| Masked Prediction (30%) | Character Accuracy | {masked30.char_accuracy:.1%} |")
        print(f"|  | Top-3 Accuracy | {masked30.top3_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {masked30.word_accuracy:.1%} |")
        print(f"| Full Reconstruction (100%) | Character Accuracy | {full100.char_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {full100.word_accuracy:.1%} |")
        print(f"| Blind Reconstruction (2-pass) | Character Accuracy | {blind.char_accuracy:.1%} |")
        print(f"|  | Word Accuracy | {blind.word_accuracy:.1%} |")
        print(f"| Length Prediction | Exact Accuracy | {length.acc_exact_rounded:.1%} |")
        print(f"|  | Within ±1 | {length.acc_within_1:.1%} |")
        print(f"|  | Within ±2 | {length.acc_within_2:.1%} |")
        dim_label = (
            "x/y"
            if mse_dims == [0, 1]
            else ("all" if not mse_dims else ",".join(map(str, mse_dims)))
        )
        print(f"| Path Reconstruction | MSE (masked; dims={dim_label}) | {path.masked_mse:.6f} |")
        return

    dataset_items = list(dataset)
    masked = evaluate_masked_tokens(
        model=model,
        processor=processor,
        dataset_items=dataset_items,
        device=device,
        batch_size=args.batch_size,
        mask_mode=args.mask_mode,
        mask_prob=args.mask_prob,
        mask_eos=args.mask_eos,
        seed=args.seed,
        show_mismatches=args.show_mismatches,
        show_top_errors=args.show_top_errors,
    )

    print("\n" + "=" * 72)
    print("Masked token evaluation")
    print("=" * 72)
    print(
        f"masked_token_accuracy: {masked.masked_token_accuracy:.4f} (n={masked.masked_token_count})"
    )
    if masked.masked_token_count_vocab:
        print(
            f"masked_token_acc(vocab): {masked.masked_token_accuracy_vocab:.4f} "
            f"(n={masked.masked_token_count_vocab})"
        )
    print(f"word_accuracy(recon):  {masked.word_accuracy_reconstruct:.4f} (n={masked.word_count})")
    print(f"word_accuracy(argmax): {masked.word_accuracy_argmax:.4f} (n={masked.word_count})")

    if not args.skip_length:
        length = evaluate_length_dataset(
            model=model,
            processor=processor,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
        )
        print("\n" + "=" * 72)
        print("Length evaluation (regression)")
        print("=" * 72)
        print(f"mae:             {length.mae:.4f}")
        print(f"rmse:            {length.rmse:.4f}")
        print(f"acc_exact_round: {length.acc_exact_rounded:.4f}")
        print(f"acc_within_1:    {length.acc_within_1:.4f}")
        print(f"acc_within_2:    {length.acc_within_2:.4f}")


if __name__ == "__main__":
    main()
