"""Evaluate a HuggingFace SwipeALot checkpoint via the customer-facing APIs.

This script intentionally loads the checkpoint using:
- `AutoModel.from_pretrained(..., trust_remote_code=True)`
- `AutoProcessor.from_pretrained(..., trust_remote_code=True)`

so evaluation reflects how the model will be used after export.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def _swipable_length(word: str) -> int:
    """Match training: count only alphanumeric characters (letters + digits)."""
    return sum(1 for c in word.lower() if c.isalpha() or c.isdigit())


@dataclass
class MaskedEvalResult:
    masked_token_accuracy: float
    masked_token_count: int
    word_accuracy: float
    word_count: int


@dataclass
class LengthEvalResult:
    mae: float
    rmse: float
    acc_exact_rounded: float
    acc_within_1: float
    acc_within_2: float
    count: int


def _iter_batches(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def evaluate_masked_tokens(
    *,
    model,
    processor,
    dataset_items: list[dict],
    device: torch.device,
    batch_size: int,
    mask_mode: str,
    mask_prob: float,
    mask_eos: bool,
    seed: int,
) -> MaskedEvalResult:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(getattr(tokenizer, "eos_token_id", -1))

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    masked_correct = 0
    masked_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_batches(dataset_items, batch_size),
            total=(len(dataset_items) + batch_size - 1) // batch_size,
            desc="Masked token eval",
        ):
            words = [ex["word"] for ex in batch]
            paths = [ex["data"] for ex in batch]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]  # [B, char_len]
            labels = input_ids.clone()

            maskable = input_ids.ne(pad_id)
            if not mask_eos and eos_id >= 0:
                maskable = maskable & input_ids.ne(eos_id)

            if mask_mode == "none":
                masked_positions = torch.zeros_like(maskable, dtype=torch.bool)
            elif mask_mode == "all":
                masked_positions = maskable
            elif mask_mode == "random":
                r = torch.rand(maskable.shape, generator=rng, device="cpu")
                masked_positions = (r < float(mask_prob)) & maskable.cpu()
                masked_positions = masked_positions.to(device)
            else:
                raise ValueError(f"Unknown mask_mode: {mask_mode}")

            labels[~masked_positions] = -100
            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_positions] = mask_id

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                labels=labels,
                return_dict=True,
            )
            logits = outputs.char_logits  # [B, char_len, V] (text segment only)
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)
            masked = labels.ne(-100)
            if masked.any():
                masked_correct += int((preds[masked] == labels[masked]).sum().item())
                masked_total += int(masked.sum().item())

            # Word accuracy: decode the full prediction up to EOS of the *target* sequence.
            preds_cpu = preds.detach().cpu().numpy()
            input_ids_cpu = input_ids.detach().cpu().numpy()
            for i, target_word in enumerate(words):
                target_ids = input_ids_cpu[i].tolist()
                try:
                    eos_pos = target_ids.index(eos_id) if eos_id >= 0 else len(target_ids)
                except ValueError:
                    eos_pos = len(target_ids)
                pred_ids = preds_cpu[i][:eos_pos].tolist()
                pred_word = tokenizer.decode(pred_ids).strip().lower()
                if pred_word == target_word.strip().lower():
                    word_correct += 1
                word_total += 1

    return MaskedEvalResult(
        masked_token_accuracy=(masked_correct / masked_total if masked_total else 0.0),
        masked_token_count=masked_total,
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        word_count=word_total,
    )


def evaluate_length(
    *,
    model,
    processor,
    dataset_items: list[dict],
    device: torch.device,
    batch_size: int,
) -> LengthEvalResult:
    preds: list[float] = []
    targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            _iter_batches(dataset_items, batch_size),
            total=(len(dataset_items) + batch_size - 1) // batch_size,
            desc="Length eval",
        ):
            words = [ex["word"] for ex in batch]
            paths = [ex["data"] for ex in batch]

            inputs = processor(path_coords=paths, text=None, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }
            outputs = model(**inputs, return_dict=True)

            length_logits = getattr(outputs, "length_logits", None)
            if length_logits is None:
                raise RuntimeError(
                    "Model did not return `length_logits`; is predict_length disabled?"
                )

            length_pred = length_logits.reshape(-1).detach().cpu().numpy().astype(np.float64)
            preds.extend(length_pred.tolist())
            targets.extend([_swipable_length(w) for w in words])

    pred_arr = np.array(preds, dtype=np.float64)
    tgt_arr = np.array(targets, dtype=np.float64)
    err = pred_arr - tgt_arr

    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0

    pred_round = np.round(pred_arr).clip(min=0.0)
    acc_exact = float(np.mean(pred_round == tgt_arr)) if len(err) else 0.0
    acc_w1 = float(np.mean(np.abs(pred_round - tgt_arr) <= 1.0)) if len(err) else 0.0
    acc_w2 = float(np.mean(np.abs(pred_round - tgt_arr) <= 2.0)) if len(err) else 0.0

    return LengthEvalResult(
        mae=mae,
        rmse=rmse,
        acc_exact_rounded=acc_exact,
        acc_within_1=acc_w1,
        acc_within_2=acc_w2,
        count=len(targets),
    )


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
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples (default: 5000)",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
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
        help="Mask probability when --mask-mode=random (default: 0.3)",
    )
    parser.add_argument(
        "--mask-eos",
        action="store_true",
        help="Also mask EOS tokens (default: False)",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0)")
    parser.add_argument(
        "--skip-length",
        action="store_true",
        help="Skip length evaluation (default: False)",
    )

    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists() or not ckpt.is_dir():
        raise SystemExit(f"Checkpoint directory not found: {ckpt}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(str(ckpt), trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(str(ckpt), trust_remote_code=True)

    dataset = load_dataset(args.dataset, split=f"{args.split}[:{args.n_samples}]")
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
    )

    print("\n" + "=" * 72)
    print("Masked token evaluation")
    print("=" * 72)
    print(
        f"masked_token_accuracy: {masked.masked_token_accuracy:.4f} (n={masked.masked_token_count})"
    )
    print(f"word_accuracy:         {masked.word_accuracy:.4f} (n={masked.word_count})")

    if not args.skip_length:
        length = evaluate_length(
            model=model,
            processor=processor,
            dataset_items=dataset_items,
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
