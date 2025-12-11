"""Metrics for evaluating swipe keyboard model performance."""

import torch


class CharacterAccuracy:
    """Track character-level prediction accuracy."""

    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, labels: torch.Tensor):
        """
        Update metrics.

        Args:
            predictions: [batch, seq_len, vocab_size] logits
            labels: [batch, seq_len] with -100 for non-masked positions
        """
        pred_tokens = predictions.argmax(dim=-1)
        mask = labels != -100

        self.correct += (pred_tokens[mask] == labels[mask]).sum().item()
        self.total += mask.sum().item()

    def compute(self) -> float:
        """Compute accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        """Reset metrics."""
        self.correct = 0
        self.total = 0


class WordAccuracy:
    """Track word-level reconstruction accuracy."""

    def __init__(self, tokenizer):
        """
        Initialize metric.

        Args:
            tokenizer: Character tokenizer for decoding
        """
        self.tokenizer = tokenizer
        self.correct = 0
        self.total = 0

    def update(self, predictions: torch.Tensor, original_words: list):
        """
        Update metrics.

        Args:
            predictions: [batch, char_len, vocab_size] character predictions
            original_words: List of original words
        """
        pred_tokens = predictions.argmax(dim=-1)  # [batch, char_len]

        for pred, original in zip(pred_tokens, original_words):
            # Decode and extract word up to EOS token
            pred_ids = pred.cpu().tolist()

            # Find EOS token if present
            eos_id = self.tokenizer.eos_token_id
            if eos_id in pred_ids:
                eos_idx = pred_ids.index(eos_id)
                pred_ids = pred_ids[:eos_idx]  # Exclude EOS from comparison

            pred_word = self.tokenizer.decode(pred_ids)
            pred_word = pred_word.strip()

            # Case-insensitive comparison (tokenizer is case-insensitive)
            if pred_word == original.strip().lower():
                self.correct += 1
            self.total += 1

    def compute(self) -> float:
        """Compute accuracy."""
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        """Reset metrics."""
        self.correct = 0
        self.total = 0
