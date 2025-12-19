"""Shared helpers for evaluation metrics."""

from __future__ import annotations

from collections.abc import Iterable


def safe_token(tokenizer, token_id: int) -> str:
    try:
        return str(tokenizer.convert_ids_to_tokens(int(token_id)))
    except Exception:
        return str(token_id)


def vocab_start_id(tokenizer) -> int | None:
    # SwipeTokenizer wraps CharacterTokenizer as `._tokenizer`.
    inner = getattr(tokenizer, "_tokenizer", None)
    special_tokens = getattr(inner, "special_tokens", None)
    if isinstance(special_tokens, list):
        return len(special_tokens)
    return None


def iter_batches(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def iter_dataset_batches(dataset, batch_size: int) -> Iterable[dict[str, list]]:
    for start in range(0, len(dataset), batch_size):
        yield dataset[start : start + batch_size]
