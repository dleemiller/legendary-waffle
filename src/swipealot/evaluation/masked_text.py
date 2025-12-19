"""Masked-text evaluation metrics (masked token prediction and reconstruction)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from swipealot.evaluation.common import (
    iter_batches,
    iter_dataset_batches,
    safe_token,
    vocab_start_id,
)
from swipealot.text_utils import swipable_text


@dataclass
class MaskedEvalResult:
    masked_token_accuracy: float
    masked_token_count: int
    masked_token_accuracy_vocab: float
    masked_token_count_vocab: int
    word_accuracy_reconstruct: float
    word_accuracy_argmax: float
    word_count: int


@dataclass
class MaskedPredictionMetrics:
    char_accuracy: float
    top3_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


@dataclass
class FullReconstructionMetrics:
    char_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


@dataclass
class BlindReconstructionMetrics:
    """Two-pass ('blind reconstruction'): predict length from path, then reconstruct masked text."""

    char_accuracy: float
    word_accuracy: float
    n_chars: int
    n_words: int


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
    show_mismatches: int = 0,
    show_top_errors: int = 0,
) -> MaskedEvalResult:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(getattr(tokenizer, "eos_token_id", -1))

    v_start = vocab_start_id(tokenizer)
    if v_start is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    masked_correct = 0
    masked_total = 0

    masked_correct_vocab = 0
    masked_total_vocab = 0

    word_correct = 0
    word_correct_argmax = 0
    word_total = 0

    mismatches: list[dict] = []
    error_pair_counts: dict[tuple[int, int], int] = {}
    true_token_counts: dict[int, int] = {}
    true_token_correct: dict[int, int] = {}

    model.eval()
    with torch.no_grad():
        for raw_batch in tqdm(
            iter_batches(dataset_items, batch_size),
            total=(len(dataset_items) + batch_size - 1) // batch_size,
            desc="Masked token eval",
        ):
            words = [item["word"] for item in raw_batch]
            paths = [item["data"] for item in raw_batch]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]  # [B, char_len]
            labels = input_ids.clone()

            if mask_mode == "none":
                masked_positions = torch.zeros_like(input_ids, dtype=torch.bool)
            else:
                maskable = input_ids.ne(pad_id)
                if not mask_eos and eos_id >= 0:
                    maskable = maskable & input_ids.ne(eos_id)

                if mask_mode == "all":
                    masked_positions = maskable
                elif mask_mode == "random":
                    r = torch.rand(maskable.shape, generator=rng, device="cpu")
                    masked_positions = (r < float(mask_prob)) & maskable.cpu()
                    masked_positions = masked_positions.to(device)
                else:
                    raise ValueError(f"Unknown mask_mode={mask_mode!r}")

            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_positions] = mask_id

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)

            masked = masked_positions & labels.ne(pad_id)
            if masked.any():
                masked_correct += int((preds[masked] == labels[masked]).sum().item())
                masked_total += int(masked.sum().item())

            masked_vocab = masked & labels.ge(int(v_start))
            if masked_vocab.any():
                masked_correct_vocab += int(
                    (preds[masked_vocab] == labels[masked_vocab]).sum().item()
                )
                masked_total_vocab += int(masked_vocab.sum().item())

            if show_top_errors > 0 and masked.any():
                true_ids = labels[masked].detach().cpu().numpy().astype(np.int64)
                pred_ids = preds[masked].detach().cpu().numpy().astype(np.int64)
                for t, p in zip(true_ids.tolist(), pred_ids.tolist(), strict=False):
                    true_token_counts[t] = true_token_counts.get(t, 0) + 1
                    if p == t:
                        true_token_correct[t] = true_token_correct.get(t, 0) + 1
                    else:
                        key = (t, p)
                        error_pair_counts[key] = error_pair_counts.get(key, 0) + 1

            preds_cpu = preds.detach().cpu().numpy()
            input_ids_cpu = input_ids.detach().cpu().numpy()
            masked_positions_cpu = masked_positions.detach().cpu().numpy()

            for i, target_word in enumerate(words):
                target_ids = input_ids_cpu[i].tolist()
                try:
                    eos_pos = target_ids.index(eos_id) if eos_id >= 0 else len(target_ids)
                except ValueError:
                    eos_pos = len(target_ids)

                target_swipable = swipable_text(target_word)

                recon_ids = target_ids[:eos_pos]
                pred_seq = preds_cpu[i][:eos_pos].tolist()
                mask_seq = masked_positions_cpu[i][:eos_pos].tolist()
                for pos, is_masked in enumerate(mask_seq):
                    if is_masked:
                        recon_ids[pos] = int(pred_seq[pos])

                pred_swipable = swipable_text(tokenizer.decode(recon_ids))
                if pred_swipable == target_swipable:
                    word_correct += 1

                pred_argmax_swipable = swipable_text(tokenizer.decode(pred_seq))
                if pred_argmax_swipable == target_swipable:
                    word_correct_argmax += 1

                word_total += 1

                if show_mismatches > 0 and pred_swipable != target_swipable:
                    diffs = []
                    for pos, is_masked in enumerate(mask_seq):
                        if not is_masked:
                            continue
                        true_id = int(target_ids[pos])
                        pred_id = int(pred_seq[pos])
                        if pred_id != true_id:
                            diffs.append(
                                {
                                    "pos": pos,
                                    "true": safe_token(tokenizer, true_id),
                                    "pred": safe_token(tokenizer, pred_id),
                                }
                            )

                    mismatches.append(
                        {
                            "target": target_word,
                            "target_swipable": target_swipable,
                            "pred_reconstruct_swipable": pred_swipable,
                            "pred_argmax_swipable": pred_argmax_swipable,
                            "n_masked": int(sum(mask_seq)),
                            "n_wrong_masked": len(diffs),
                            "wrong_masked": diffs,
                        }
                    )

    if show_mismatches > 0 and mismatches:
        mismatches.sort(key=lambda d: (d["n_wrong_masked"], d["n_masked"]), reverse=True)
        print("\n" + "=" * 72)
        print(f"Mismatches (showing up to {min(show_mismatches, len(mismatches))})")
        print("=" * 72)
        for ex in mismatches[:show_mismatches]:
            print(f"target:    {ex['target']!r}  (swipable={ex['target_swipable']!r})")
            print(f"pred(re):  {ex['pred_reconstruct_swipable']!r}")
            print(f"pred(arg): {ex['pred_argmax_swipable']!r}")
            print(
                f"masked: {ex['n_masked']}  wrong_masked: {ex['n_wrong_masked']}  details: {ex['wrong_masked']}"
            )
            print("-" * 72)

    if show_top_errors > 0 and error_pair_counts:
        top_pairs = sorted(error_pair_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :show_top_errors
        ]
        top_tokens = sorted(true_token_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :show_top_errors
        ]
        print("\n" + "=" * 72)
        print(f"Top masked-token errors (up to {len(top_pairs)})")
        print("=" * 72)
        for (true_id, pred_id), n in top_pairs:
            print(
                f"{n:6d}  true={safe_token(tokenizer, true_id):>8s}  "
                f"pred={safe_token(tokenizer, pred_id):>8s}"
            )

        print("\n" + "=" * 72)
        print(f"Most-masked tokens + accuracy (up to {len(top_tokens)})")
        print("=" * 72)
        for tok_id, n in top_tokens:
            correct = true_token_correct.get(tok_id, 0)
            acc = correct / n if n else 0.0
            print(f"{n:6d}  acc={acc:6.3f}  token={safe_token(tokenizer, tok_id)}")

    return MaskedEvalResult(
        masked_token_accuracy=(masked_correct / masked_total if masked_total else 0.0),
        masked_token_count=masked_total,
        masked_token_accuracy_vocab=(
            masked_correct_vocab / masked_total_vocab if masked_total_vocab else 0.0
        ),
        masked_token_count_vocab=masked_total_vocab,
        word_accuracy_reconstruct=(word_correct / word_total if word_total else 0.0),
        word_accuracy_argmax=(word_correct_argmax / word_total if word_total else 0.0),
        word_count=word_total,
    )


def evaluate_masked_prediction_30pct(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
    mask_prob: float,
    seed: int,
) -> MaskedPredictionMetrics:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    v_start = vocab_start_id(tokenizer)
    if v_start is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    char_correct = 0
    char_top3 = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Masked prediction (30%)",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            maskable = input_ids.ne(pad_id)
            r = torch.rand(maskable.shape, generator=rng, device="cpu")
            masked_positions = (r < float(mask_prob)) & maskable.cpu()
            masked_positions = masked_positions.to(device)

            evaluable = labels.ge(int(v_start)) & maskable
            eval_mask = masked_positions & evaluable

            masked_input_ids = input_ids.clone()
            masked_input_ids[masked_positions] = mask_id

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)

            if eval_mask.any():
                char_correct += int((preds[eval_mask] == labels[eval_mask]).sum().item())
                char_total += int(eval_mask.sum().item())

                top3 = logits.topk(k=3, dim=-1).indices
                hit_top3 = (top3 == labels.unsqueeze(-1)).any(dim=-1)
                char_top3 += int(hit_top3[eval_mask].sum().item())

            per_word_total = eval_mask.sum(dim=1)
            if per_word_total.any():
                per_word_correct = ((preds == labels) & eval_mask).sum(dim=1)
                word_mask = per_word_total > 0
                word_total += int(word_mask.sum().item())
                word_correct += int(
                    (per_word_correct[word_mask] == per_word_total[word_mask]).sum().item()
                )

    return MaskedPredictionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        top3_accuracy=(char_top3 / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )


def evaluate_full_reconstruction_100pct(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> FullReconstructionMetrics:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)

    v_start = vocab_start_id(tokenizer)
    if v_start is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Full reconstruction (100%)",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            input_ids = inputs["input_ids"]
            labels = input_ids.clone()

            maskable = input_ids.ne(pad_id)
            masked_input_ids = input_ids.clone()
            masked_input_ids[maskable] = mask_id

            evaluable = labels.ge(int(v_start)) & maskable

            outputs = model(
                path_coords=inputs["path_coords"],
                input_ids=masked_input_ids,
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")

            preds = logits.argmax(dim=-1)

            if evaluable.any():
                char_correct += int((preds[evaluable] == labels[evaluable]).sum().item())
                char_total += int(evaluable.sum().item())

            per_word_total = evaluable.sum(dim=1)
            if per_word_total.any():
                per_word_correct = ((preds == labels) & evaluable).sum(dim=1)
                word_mask = per_word_total > 0
                word_total += int(word_mask.sum().item())
                word_correct += int(
                    (per_word_correct[word_mask] == per_word_total[word_mask]).sum().item()
                )

    return FullReconstructionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )


def evaluate_blind_reconstruction_two_pass(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> BlindReconstructionMetrics:
    tokenizer = processor.tokenizer
    pad_id = int(tokenizer.pad_token_id)
    mask_id = int(tokenizer.mask_token_id)
    eos_id = int(tokenizer.eos_token_id)

    v_start = vocab_start_id(tokenizer)
    if v_start is None:
        raise RuntimeError("Could not infer vocab_start_id for tokenizer.")

    max_path_len = int(processor.max_path_len)
    max_char_len = int(processor.max_char_len)
    char_start = 1 + max_path_len + 1

    char_correct = 0
    char_total = 0
    word_correct = 0
    word_total = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Blind reconstruction (2-pass)",
        ):
            words = batch["word"]
            paths = batch["data"]

            text_inputs = processor.encode_text(words, return_tensors="pt")
            text_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in text_inputs.items()
            }
            labels_full = text_inputs["input_ids"]
            alnum_mask = (
                labels_full.ge(int(v_start)) & labels_full.ne(pad_id) & labels_full.ne(eos_id)
            )

            path_inputs = processor.encode_path(paths, return_tensors="pt")
            path_inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in path_inputs.items()
            }
            length_out = model(**path_inputs, return_dict=True)
            pred_len = length_out.length_logits.squeeze(-1)
            pred_len_rounded = pred_len.round().long().clamp(min=0, max=max_char_len - 1)

            batch_size_actual = int(pred_len_rounded.shape[0])
            pos = torch.arange(max_char_len, device=device).unsqueeze(0)

            input_ids = torch.full(
                (batch_size_actual, max_char_len), pad_id, dtype=torch.long, device=device
            )
            mask_positions = pos < pred_len_rounded.unsqueeze(1)
            eos_positions = pos == pred_len_rounded.unsqueeze(1)
            input_ids[mask_positions] = mask_id
            input_ids[eos_positions] = eos_id

            attention_mask = path_inputs["attention_mask"].clone()
            attention_mask[:, char_start:] = (pos < (pred_len_rounded + 1).unsqueeze(1)).long()

            outputs = model(
                path_coords=path_inputs["path_coords"],
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            logits = outputs.char_logits
            if logits is None:
                raise RuntimeError("Model did not return `char_logits`; is predict_char disabled?")
            preds = logits.argmax(dim=-1)

            for i in range(batch_size_actual):
                target_ids = labels_full[i][alnum_mask[i]].detach().cpu().tolist()
                pred_ids = preds[i, : int(pred_len_rounded[i].item())].detach().cpu().tolist()

                target_len = len(target_ids)
                pred_len_i = len(pred_ids)
                denom = max(target_len, pred_len_i)
                if denom == 0:
                    continue

                correct = 0
                for j in range(min(target_len, pred_len_i)):
                    if int(pred_ids[j]) == int(target_ids[j]):
                        correct += 1

                char_correct += correct
                char_total += denom

                word_total += 1
                if target_len == pred_len_i and correct == target_len:
                    word_correct += 1

    return BlindReconstructionMetrics(
        char_accuracy=(char_correct / char_total if char_total else 0.0),
        word_accuracy=(word_correct / word_total if word_total else 0.0),
        n_chars=char_total,
        n_words=word_total,
    )
