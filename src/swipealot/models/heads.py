"""Prediction heads for SwipeTransformer."""

import torch
import torch.nn as nn


class CharacterPredictionHead(nn.Module):
    """Prediction head for masked characters."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, vocab_size] logits
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        logits = self.decoder(x)
        return logits


class PathPredictionHead(nn.Module):
    """Prediction head for masked path coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, 3)  # Predict (x, y, t)
        self.activation = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, d_model]

        Returns:
            [batch, seq_len, 3] coordinates in [0, 1] range
        """
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.layer_norm(x)
        coords = self.decoder(x)
        coords = torch.sigmoid(coords)  # Ensure [0, 1] range
        return coords
