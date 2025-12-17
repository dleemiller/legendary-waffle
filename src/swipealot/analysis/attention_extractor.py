"""Attention extraction utilities for transformer model analysis.

This module contains:
- A legacy hook-based extractor for PyTorch TransformerEncoder models.
- Post-processing utilities for turning self-attention weights into
  per-character attention profiles over path positions (useful for
  visualization and distillation).
"""

import torch
import torch.nn as nn


class AttentionHookManager:
    """Manages forward hooks for extracting attention weights from transformer layers.

    The PyTorch TransformerEncoder doesn't return attention weights by default.
    This class uses forward hooks on MultiheadAttention modules to capture
    attention weights during the forward pass.

    Attributes:
        model: SwipeTransformerModel instance
        target_layers: List of layer indices to extract attention from
        attention_weights: Dict mapping layer_idx -> attention tensor
        hooks: List of registered hook handles for cleanup
    """

    def __init__(self, model: nn.Module, target_layers: list[int] = None):
        """Initialize the attention hook manager.

        Args:
            model: SwipeTransformerModel instance
            target_layers: List of layer indices to hook (default: [0, 6, 11])
        """
        self.model = model
        self.target_layers = target_layers if target_layers is not None else [0, 6, 11]
        self.attention_weights = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        """Create a hook closure for a specific layer.

        Args:
            layer_idx: Index of the transformer layer

        Returns:
            Hook function that captures attention weights
        """

        def hook(module: nn.Module, input: tuple, output: tuple):
            """Forward hook that captures attention weights.

            PyTorch's MultiheadAttention returns (attn_output, attn_weights) when
            need_weights=True and average_attn_weights=False.
            """
            # Output tuple: (attn_output, attn_weights) or just (attn_output,)
            if len(output) > 1 and output[1] is not None:
                # Store attention weights: [batch, n_heads, seq_len, seq_len]
                self.attention_weights[layer_idx] = output[1].detach()

        return hook

    def register_hooks(self):
        """Register forward hooks on target transformer layers.

        Hooks are registered on the self_attn (MultiheadAttention) module
        within each TransformerEncoderLayer.
        """
        for idx in self.target_layers:
            if idx >= len(self.model.encoder.layers):
                raise ValueError(
                    f"Layer index {idx} out of range. "
                    f"Model has {len(self.model.encoder.layers)} layers."
                )

            layer = self.model.encoder.layers[idx]

            # Hook into the MultiheadAttention module
            handle = layer.self_attn.register_forward_hook(self._make_hook(idx))
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks and clear stored attention weights."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.attention_weights.clear()

    def extract_attention(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[int, torch.Tensor]:
        """Run forward pass and extract attention weights from target layers.

        This method monkey-patches the MultiheadAttention forward methods to
        return attention weights, runs the forward pass, then restores originals.

        Args:
            path_coords: Path coordinates [batch, path_len, 3]
            char_tokens: Character tokens [batch, char_len]
            attention_mask: Attention mask [batch, seq_len]

        Returns:
            Dict mapping layer_idx -> attention_weights [batch, n_heads, seq_len, seq_len]
        """
        # Clear any previous attention weights
        self.attention_weights.clear()

        # Monkey-patch forward methods to return attention weights
        original_forwards = {}

        for idx in self.target_layers:
            layer = self.model.encoder.layers[idx]
            attn_module = layer.self_attn

            # Store original forward method
            original_forwards[idx] = attn_module.forward

            # Create patched forward that forces need_weights=True
            def make_patched_forward(original_forward):
                def patched_forward(
                    query,
                    key,
                    value,
                    key_padding_mask=None,
                    need_weights=True,
                    attn_mask=None,
                    average_attn_weights=False,
                    is_causal=False,
                ):
                    # Force attention weights to be returned (per-head)
                    return original_forward(
                        query,
                        key,
                        value,
                        key_padding_mask=key_padding_mask,
                        need_weights=True,  # Force True
                        attn_mask=attn_mask,
                        average_attn_weights=False,  # Per-head weights
                        is_causal=is_causal,
                    )

                return patched_forward

            # Apply patch
            attn_module.forward = make_patched_forward(original_forwards[idx])

        # Register hooks
        self.register_hooks()

        try:
            # Run forward pass
            with torch.no_grad():
                _ = self.model(
                    path_coords=path_coords,
                    char_tokens=char_tokens,
                    attention_mask=attention_mask,
                )

            # Return captured attention weights
            return self.attention_weights.copy()

        finally:
            # Cleanup: remove hooks and restore original forward methods
            self.remove_hooks()

            # Restore original forward methods
            for idx in self.target_layers:
                layer = self.model.encoder.layers[idx]
                layer.self_attn.forward = original_forwards[idx]

    def __enter__(self):
        """Context manager entry: register hooks."""
        self.register_hooks()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: cleanup hooks."""
        self.remove_hooks()
        return False


def extract_path_to_char_attention(
    attention_weights: torch.Tensor,
    path_len: int,
    char_len: int,
    aggregation: str = "max",
) -> torch.Tensor:
    """Extract character-to-path attention from a full attention matrix.

    Given the full attention matrix, extract the submatrix where character
    positions (queries) attend to path positions (keys), then aggregate
    across attention heads.

    Sequence structure: [CLS](0) + PATH + [SEP] + CHARS
    - CLS position: 0
    - Path positions: 1..path_len
    - SEP position: 1 + path_len
    - Character positions: (1 + path_len + 1)..(1 + path_len + 1 + char_len - 1)

    Args:
        attention_weights: Attention tensor [batch, n_heads, seq_len, seq_len]
        path_len: Number of path tokens in the sequence.
        char_len: Number of character tokens in the sequence.
        aggregation: How to aggregate across heads - "max", "mean", "sum", or "logsumexp".

    Returns:
        Attention from chars to paths: [batch, char_len, path_len]
    """
    if attention_weights.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor [batch, heads, seq, seq], "
            f"got shape {attention_weights.shape}"
        )

    if path_len <= 0 or char_len <= 0:
        raise ValueError(f"Invalid lengths: path_len={path_len}, char_len={char_len}")

    path_start = 1
    path_end = 1 + path_len
    char_start = 1 + path_len + 1
    char_end = char_start + char_len

    # Extract char→path submatrix:
    # Rows: character positions (queries), Cols: path positions (keys)
    char_to_path = attention_weights[:, :, char_start:char_end, path_start:path_end]

    # Aggregate across heads (dimension 1)
    if aggregation == "max":
        # Max pooling: take strongest attention signal from any head
        result = char_to_path.max(dim=1)[0]
    elif aggregation == "mean":
        # Average: consensus attention across heads
        result = char_to_path.mean(dim=1)
    elif aggregation == "sum":
        # Sum: total attention across heads
        result = char_to_path.sum(dim=1)
    elif aggregation == "logsumexp":
        # LogSumExp: smooth approximation to max
        result = torch.logsumexp(char_to_path, dim=1)
    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation}. Use 'max', 'mean', 'sum', or 'logsumexp'."
        )

    return result


def extract_special_token_to_path_attention(
    attention_weights: torch.Tensor,
    path_len: int,
    word_length: int,
    aggregation: str = "max",
) -> dict[str, torch.Tensor]:
    """Extract special token-to-path attention from full attention matrix.

    Extracts attention from CLS, SEP, and EOS tokens to path positions.

    Sequence structure: [CLS](0) + PATH + [SEP] + CHARS
    - CLS position: 0
    - SEP position: 1 + path_len
    - EOS position: (1 + path_len + 1) + word_length

    Args:
        attention_weights: Attention tensor [batch, n_heads, seq_len, seq_len]
        path_len: Number of path tokens in the sequence.
        word_length: Length of the word (to locate EOS token; excludes EOS itself).
        aggregation: How to aggregate across heads - "max", "mean", "sum", or "logsumexp"

    Returns:
        Dict with keys 'cls', 'sep', 'eos', each containing attention to path
        Shape of each: [batch, path_len]
    """
    if attention_weights.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor [batch, heads, seq, seq], "
            f"got shape {attention_weights.shape}"
        )

    if path_len <= 0:
        raise ValueError(f"Invalid path_len={path_len}")

    # Extract token→path for each special token
    # Rows: token positions, Cols: path positions (1..path_len)
    path_start = 1
    path_end = 1 + path_len
    sep_pos = 1 + path_len
    char_start = 1 + path_len + 1
    eos_pos = char_start + word_length

    cls_to_path = attention_weights[:, :, 0:1, path_start:path_end]  # [B, H, 1, P]
    sep_to_path = attention_weights[
        :, :, sep_pos : sep_pos + 1, path_start:path_end
    ]  # [B, H, 1, P]
    eos_to_path = attention_weights[
        :, :, eos_pos : eos_pos + 1, path_start:path_end
    ]  # [B, H, 1, P]

    # Aggregate across heads (dimension 1) and squeeze token dimension
    if aggregation == "max":
        cls_result = cls_to_path.max(dim=1)[0].squeeze(1)  # [batch, 128]
        sep_result = sep_to_path.max(dim=1)[0].squeeze(1)
        eos_result = eos_to_path.max(dim=1)[0].squeeze(1)
    elif aggregation == "mean":
        cls_result = cls_to_path.mean(dim=1).squeeze(1)
        sep_result = sep_to_path.mean(dim=1).squeeze(1)
        eos_result = eos_to_path.mean(dim=1).squeeze(1)
    elif aggregation == "sum":
        cls_result = cls_to_path.sum(dim=1).squeeze(1)
        sep_result = sep_to_path.sum(dim=1).squeeze(1)
        eos_result = eos_to_path.sum(dim=1).squeeze(1)
    elif aggregation == "logsumexp":
        cls_result = torch.logsumexp(cls_to_path, dim=1).squeeze(1)
        sep_result = torch.logsumexp(sep_to_path, dim=1).squeeze(1)
        eos_result = torch.logsumexp(eos_to_path, dim=1).squeeze(1)
    else:
        raise ValueError(
            f"Unknown aggregation method: {aggregation}. Use 'max', 'mean', 'sum', or 'logsumexp'."
        )

    return {
        "cls": cls_result,
        "sep": sep_result,
        "eos": eos_result,
    }


def identify_dominant_head(
    attention_weights: torch.Tensor, path_len: int, char_len: int
) -> torch.Tensor:
    """Identify which attention head has maximum attention for each character-path pair.

    Args:
        attention_weights: Attention tensor [batch, n_heads, seq_len, seq_len]
        path_len: Number of path tokens in the sequence.
        char_len: Number of character tokens in the sequence.

    Returns:
        Head indices [batch, char_len, path_len] indicating which head had the maximum
        attention for each char→path pair.
    """
    if attention_weights.dim() != 4:
        raise ValueError(
            f"Expected 4D attention tensor [batch, heads, seq, seq], got {attention_weights.shape}"
        )

    path_start = 1
    path_end = 1 + path_len
    char_start = 1 + path_len + 1
    char_end = char_start + char_len
    char_to_path = attention_weights[:, :, char_start:char_end, path_start:path_end]

    # Get indices of maximum head for each position
    max_head_indices = char_to_path.argmax(dim=1)

    return max_head_indices


def compute_char_to_path_attention_profile(
    attentions: tuple[torch.Tensor, ...],
    *,
    path_len: int,
    char_len: int,
    head_aggregation: str = "mean",
    layer_aggregation: str = "mean_last_k",
    last_k_layers: int = 4,
    renormalize_over_path: bool = True,
    attention_mask: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    exclude_token_ids: set[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Turn per-layer self-attention into a per-character attention profile over path positions.

    This is intended for distillation/visualization: for each character token (query),
    produce a 1D profile over path positions (keys).

    Returns a dict containing:
    - profile: [batch, char_len, path_len]
    - path_mass: [batch, char_len] (fraction of attention landing on path keys, pre-renorm)
    - entropy: [batch, char_len] (entropy of profile after renorm; 0 where invalid)
    - query_keep_mask: [batch, char_len] (which queries were kept)
    - path_key_mask: [batch, path_len] (valid path positions from attention_mask if provided)
    """
    if len(attentions) == 0:
        raise ValueError("attentions must be a non-empty tuple of [B, H, S, S] tensors")

    if layer_aggregation == "last":
        selected = [attentions[-1]]
    elif layer_aggregation == "mean":
        selected = list(attentions)
    elif layer_aggregation == "mean_last_k":
        k = min(last_k_layers, len(attentions))
        selected = list(attentions[-k:])
    else:
        raise ValueError(
            f"Unknown layer_aggregation={layer_aggregation!r} "
            "(use 'last', 'mean', or 'mean_last_k')"
        )

    def slice_layer(a: torch.Tensor) -> torch.Tensor:
        path_start = 1
        path_end = 1 + path_len
        char_start = 1 + path_len + 1
        char_end = char_start + char_len
        return a[:, :, char_start:char_end, path_start:path_end]

    sliced = [slice_layer(a) for a in selected]  # [B, H, C, P]
    stacked = torch.stack(sliced, dim=0)  # [L, B, H, C, P]

    if head_aggregation == "mean":
        head_pooled = stacked.mean(dim=2)  # [L, B, C, P]
    elif head_aggregation == "max":
        head_pooled = stacked.max(dim=2)[0]
    elif head_aggregation == "sum":
        head_pooled = stacked.sum(dim=2)
    elif head_aggregation == "logsumexp":
        head_pooled = torch.logsumexp(stacked, dim=2)
    else:
        raise ValueError(
            f"Unknown head_aggregation={head_aggregation!r} "
            "(use 'mean', 'max', 'sum', or 'logsumexp')"
        )

    # Layer pooling
    profile = head_pooled.mean(dim=0)  # [B, C, P]

    # Optional key masking (defensive; should already be zero from attention masking)
    if attention_mask is not None:
        path_start = 1
        path_end = 1 + path_len
        path_key_mask = attention_mask[:, path_start:path_end].to(profile.dtype)  # [B, P]
    else:
        path_key_mask = torch.ones(
            profile.shape[0], path_len, device=profile.device, dtype=profile.dtype
        )

    profile = profile * path_key_mask.unsqueeze(1)
    path_mass = profile.sum(dim=-1)  # [B, C]

    # Query filtering
    if input_ids is not None:
        exclude = exclude_token_ids or set()
        if len(exclude) > 0:
            query_keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
            for token_id in exclude:
                query_keep_mask = query_keep_mask & (input_ids != token_id)
        else:
            query_keep_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        query_keep_mask = torch.ones(
            profile.shape[0], char_len, device=profile.device, dtype=torch.bool
        )

    if attention_mask is not None:
        char_start = 1 + path_len + 1
        char_end = char_start + char_len
        char_attn_mask = attention_mask[:, char_start:char_end].bool()
        query_keep_mask = query_keep_mask & char_attn_mask

    profile = profile * query_keep_mask.unsqueeze(-1).to(profile.dtype)
    path_mass = path_mass * query_keep_mask.to(profile.dtype)

    # Renormalize within path slice (gives a clean 1D distribution over time)
    if renormalize_over_path:
        denom = profile.sum(dim=-1, keepdim=True)
        profile = torch.where(denom > 0, profile / denom, torch.zeros_like(profile))

    # Entropy over path positions (0 where invalid)
    eps = 1e-12
    p = torch.clamp(profile, min=0.0)
    entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    entropy = entropy * query_keep_mask.to(entropy.dtype)

    return {
        "profile": profile,
        "path_mass": path_mass,
        "entropy": entropy,
        "query_keep_mask": query_keep_mask,
        "path_key_mask": path_key_mask.to(torch.bool),
    }


def get_attention_statistics(
    attention_weights: torch.Tensor, dim: str = "heads"
) -> dict[str, torch.Tensor]:
    """Compute statistics of attention distribution.

    Args:
        attention_weights: Attention tensor (various shapes depending on context)
        dim: Which dimension to compute statistics over

    Returns:
        Dict with 'mean', 'std', 'min', 'max', 'entropy'
    """
    stats = {
        "mean": attention_weights.mean(),
        "std": attention_weights.std(),
        "min": attention_weights.min(),
        "max": attention_weights.max(),
    }

    # Compute entropy: H(p) = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-10
    p = attention_weights + eps
    entropy = -(p * torch.log(p)).sum()
    stats["entropy"] = entropy

    return stats
