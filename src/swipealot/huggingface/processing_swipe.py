"""Processor for handling multimodal swipe inputs (path + text)."""

import numpy as np
import torch
from transformers import ProcessorMixin


class SwipeProcessor(ProcessorMixin):
    """
    Processor for handling multimodal swipe inputs (path coordinates + text).

    This processor combines path coordinate preprocessing with text tokenization,
    creating the inputs needed for SwipeTransformer models.

    Args:
        tokenizer: SwipeTokenizer instance
        max_path_len (int): Maximum path length. Defaults to 64.
        max_char_len (int): Maximum character length. Defaults to 38.
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"  # Will use auto_map from tokenizer_config.json

    def __init__(self, tokenizer=None, max_path_len: int = 64, max_char_len: int = 38):
        self.tokenizer = tokenizer
        self.max_path_len = max_path_len
        self.max_char_len = max_char_len
        # Attributes expected by newer transformers (not used for swipe models)
        self.chat_template = None
        self.audio_tokenizer = None
        self.feature_extractor = None
        self.image_processor = None

    def __call__(
        self,
        path_coords: list[list[list[float]]] | torch.Tensor | np.ndarray | None = None,
        text: str | list[str] | None = None,
        padding: bool | str = True,
        truncation: bool = True,
        max_length: int | None = None,
        return_tensors: str | None = "pt",
        **kwargs,
    ):
        """
        Process path coordinates and text into model inputs.

        Args:
            path_coords: List of paths or tensor [batch, path_len, 3]
                        Each point is (x, y, time). Can be None if only processing text.
            text: String or list of strings to encode. Can be None if only processing paths.
            padding: Whether to pad sequences. Can be True/False or "max_length"
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length for text (overrides max_char_len)
            return_tensors: "pt" for PyTorch, "np" for NumPy, None for lists
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary with:
                - path_coords: [batch, max_path_len, 3] (if path_coords provided)
                - char_tokens: [batch, max_char_len] (if text provided)
                - attention_mask: [batch, total_seq_len]
        """
        if path_coords is None and text is None:
            raise ValueError("Must provide either path_coords or text (or both)")

        # Determine batch size
        if path_coords is not None:
            # Handle path coordinates
            if isinstance(path_coords, (list, tuple)):
                # Check if it's a batch or single path
                if len(path_coords) > 0 and isinstance(path_coords[0][0], (list, tuple)):
                    # Batch of paths [[path1], [path2], ...]
                    path_coords = torch.tensor(path_coords, dtype=torch.float32)
                else:
                    # Single path [[x,y,t], [x,y,t], ...]
                    path_coords = torch.tensor([path_coords], dtype=torch.float32)
            elif isinstance(path_coords, np.ndarray):
                path_coords = torch.from_numpy(path_coords).float()
                if path_coords.dim() == 2:
                    # Single path, add batch dimension
                    path_coords = path_coords.unsqueeze(0)
            elif isinstance(path_coords, torch.Tensor):
                if path_coords.dim() == 2:
                    # Single path, add batch dimension
                    path_coords = path_coords.unsqueeze(0)

            batch_size = path_coords.shape[0]
        elif text is not None:
            if isinstance(text, str):
                batch_size = 1
                text = [text]
            else:
                batch_size = len(text)
        else:
            batch_size = 1

        result = {}

        # Process path coordinates
        if path_coords is not None:
            current_path_len = path_coords.shape[1]

            # Truncate if needed
            if truncation and current_path_len > self.max_path_len:
                path_coords = path_coords[:, : self.max_path_len, :]
                current_path_len = self.max_path_len

            # Pad if needed
            if padding and current_path_len < self.max_path_len:
                pad_len = self.max_path_len - current_path_len
                path_coords = torch.cat([path_coords, torch.zeros(batch_size, pad_len, 3)], dim=1)

            # Create path mask (1 = real data, 0 = padding)
            path_mask = torch.ones(batch_size, self.max_path_len, dtype=torch.long)
            if padding and current_path_len < self.max_path_len:
                path_mask[:, current_path_len:] = 0

            result["path_coords"] = path_coords
            # Store path_mask internally for attention_mask construction
            _path_mask = path_mask
        else:
            # No path coords provided, create empty/zero tensors
            path_coords = torch.zeros(batch_size, self.max_path_len, 3)
            _path_mask = torch.zeros(batch_size, self.max_path_len, dtype=torch.long)
            result["path_coords"] = path_coords

        # Process text
        if text is not None:
            # Ensure text is a list
            if isinstance(text, str):
                text = [text]

            # Tokenize text
            text_max_length = max_length if max_length is not None else self.max_char_len

            encoded = self.tokenizer(
                text,
                padding="max_length" if padding else False,
                truncation=truncation,
                max_length=text_max_length,
                return_tensors=return_tensors,
                **kwargs,
            )

            # Rename input_ids to char_tokens for our model
            result["char_tokens"] = encoded["input_ids"]
            # Store char_mask internally for attention_mask construction
            _char_mask = encoded["attention_mask"]
        else:
            # No text provided, create padding tokens
            if return_tensors == "pt":
                char_tokens = torch.full(
                    (batch_size, self.max_char_len), self.tokenizer.pad_token_id, dtype=torch.long
                )
                _char_mask = torch.zeros(batch_size, self.max_char_len, dtype=torch.long)
            elif return_tensors == "np":
                char_tokens = np.full(
                    (batch_size, self.max_char_len), self.tokenizer.pad_token_id, dtype=np.int64
                )
                _char_mask = np.zeros((batch_size, self.max_char_len), dtype=np.int64)
            else:
                char_tokens = [
                    [self.tokenizer.pad_token_id] * self.max_char_len for _ in range(batch_size)
                ]
                _char_mask = [[0] * self.max_char_len for _ in range(batch_size)]

            result["char_tokens"] = char_tokens

        # Create combined attention mask: [CLS] + path + [SEP] + chars
        # Sequence structure: [CLS:1] + _path_mask + [SEP:1] + _char_mask
        if return_tensors == "pt":
            cls_mask = torch.ones(batch_size, 1, dtype=torch.long)
            sep_mask = torch.ones(batch_size, 1, dtype=torch.long)
            attention_mask = torch.cat([cls_mask, _path_mask, sep_mask, _char_mask], dim=1)
        elif return_tensors == "np":
            cls_mask = np.ones((batch_size, 1), dtype=np.int64)
            sep_mask = np.ones((batch_size, 1), dtype=np.int64)
            attention_mask = np.concatenate([cls_mask, _path_mask, sep_mask, _char_mask], axis=1)
        else:
            cls_mask = [[1] for _ in range(batch_size)]
            sep_mask = [[1] for _ in range(batch_size)]
            attention_mask = [
                cls + path.tolist() + sep + char
                for cls, path, sep, char in zip(
                    cls_mask, _path_mask, sep_mask, _char_mask, strict=False
                )
            ]

        result["attention_mask"] = attention_mask

        # Convert to requested format
        if return_tensors == "np":
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].numpy()
        elif return_tensors is None:
            for key in result:
                if isinstance(result[key], torch.Tensor):
                    result[key] = result[key].tolist()

        return result

    def batch_decode(self, token_ids, **kwargs):
        """
        Decode token IDs to strings.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(token_ids, **kwargs)

    def decode(self, token_ids, **kwargs):
        """
        Decode single sequence of token IDs to string.

        Args:
            token_ids: Token IDs to decode
            **kwargs: Additional arguments passed to tokenizer

        Returns:
            Decoded string
        """
        return self.tokenizer.decode(token_ids, **kwargs)
