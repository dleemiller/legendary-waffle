"""HuggingFace-compatible model classes for SwipeTransformer."""

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    SequenceClassifierOutput,
)

from .configuration_swipe import SwipeCrossEncoderConfig, SwipeTransformerConfig


class SwipeTransformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface
    for downloading and loading pretrained models.
    """

    config_class = SwipeTransformerConfig
    base_model_prefix = "swipe_transformer"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)


class SwipeTransformerModel(SwipeTransformerPreTrainedModel):
    """
    HuggingFace-compatible SwipeTransformerModel.

    This model reuses the existing components from src/swipealot/models/
    and wraps them in a HuggingFace-compatible interface.

    Args:
        config (SwipeTransformerConfig): Model configuration
    """

    def __init__(self, config: SwipeTransformerConfig):
        super().__init__(config)
        self.config = config

        # Import existing components
        from ..models.embeddings import MixedEmbedding
        from ..models.heads import CharacterPredictionHead, PathPredictionHead

        # Embeddings
        self.embeddings = MixedEmbedding(
            vocab_size=config.vocab_size,
            max_path_len=config.max_path_len,
            max_char_len=config.max_char_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Prediction heads
        self.char_head = CharacterPredictionHead(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
        )

        if config.predict_path:
            self.path_head = PathPredictionHead(d_model=config.d_model)
        else:
            self.path_head = None

        # Initialize weights
        self.post_init()

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
    ):
        """
        Forward pass of the model.

        Args:
            path_coords (torch.Tensor): Path coordinates [batch, path_len, 3]
            char_tokens (torch.Tensor): Character tokens [batch, char_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
            labels (torch.Tensor, optional): Labels for loss calculation [batch, char_len]
            return_dict (bool, optional): Whether to return ModelOutput object
            output_hidden_states (bool, optional): Whether to output hidden states

        Returns:
            BaseModelOutput or tuple: Model outputs
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full(
            (batch_size, 1), fill_value=self.config.cls_token_id, dtype=torch.long, device=device
        )
        sep_token = torch.full(
            (batch_size, 1), fill_value=self.config.sep_token_id, dtype=torch.long, device=device
        )

        # Get embeddings
        embeddings = self.embeddings(path_coords, char_tokens, cls_token, sep_token)

        # Prepare attention mask for encoder
        if attention_mask is not None:
            # Convert attention mask: 1 = attend, 0 = ignore
            # PyTorch expects: False = attend, True = ignore
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode (batch_first=True is set in TransformerEncoderLayer)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Character prediction
        char_logits = self.char_head(hidden_states)

        # Path prediction (if enabled)
        path_logits = None
        if self.path_head is not None:
            path_logits = self.path_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.pad_token_id)
            # Extract character positions from hidden states
            # Sequence is: [CLS] + path + [SEP] + chars
            path_len = path_coords.shape[1]
            char_start = 1 + path_len + 1  # After [CLS], path, and [SEP]
            char_hidden = hidden_states[:, char_start : char_start + labels.shape[1], :]
            char_pred = self.char_head(char_hidden)
            loss = loss_fct(char_pred.reshape(-1, self.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            output = (hidden_states, char_logits)
            if path_logits is not None:
                output = output + (path_logits,)
            return ((loss,) + output) if loss is not None else output

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )


class SwipeCrossEncoderForSequenceClassification(SwipeTransformerPreTrainedModel):
    """
    HuggingFace-compatible cross-encoder for sequence classification.

    This model is designed for similarity scoring between swipe paths and words.
    It extracts the SEP token embedding and passes it through a classification head.

    Args:
        config (SwipeCrossEncoderConfig): Model configuration
    """

    config_class = SwipeCrossEncoderConfig
    base_model_prefix = "swipe_cross_encoder"

    def __init__(self, config: SwipeCrossEncoderConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels

        # Import existing components
        from ..models.embeddings import MixedEmbedding
        from ..models.heads import ClassificationHead

        # Embeddings
        self.embeddings = MixedEmbedding(
            vocab_size=config.vocab_size,
            max_path_len=config.max_path_len,
            max_char_len=config.max_char_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Classification head
        self.classifier = ClassificationHead(
            d_model=config.d_model,
            num_labels=config.num_labels,
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ):
        """
        Forward pass for cross-encoder.

        Args:
            path_coords (torch.Tensor): Path coordinates [batch, path_len, 3]
            char_tokens (torch.Tensor): Character tokens [batch, char_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
            labels (torch.Tensor, optional): Labels for loss calculation [batch, num_labels]
            return_dict (bool, optional): Whether to return ModelOutput object

        Returns:
            SequenceClassifierOutput or tuple: Model outputs with logits and optional loss
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full(
            (batch_size, 1), fill_value=self.config.cls_token_id, dtype=torch.long, device=device
        )
        sep_token = torch.full(
            (batch_size, 1), fill_value=self.config.sep_token_id, dtype=torch.long, device=device
        )

        # Get embeddings
        embeddings = self.embeddings(path_coords, char_tokens, cls_token, sep_token)

        # Prepare attention mask
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode (batch_first=True is set in TransformerEncoderLayer)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Extract SEP token embedding
        # SEP is at position 1 + path_len
        path_len = path_coords.shape[1]
        sep_position = 1 + path_len
        sep_embedding = hidden_states[:, sep_position, :]  # [batch, d_model]

        # Classification
        logits = self.classifier(sep_embedding)  # [batch, num_labels]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                else:
                    self.config.problem_type = "single_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + (hidden_states,)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=(hidden_states,),
        )


class SwipeModel(SwipeTransformerPreTrainedModel):
    """
    Base Swipe model for extracting embeddings.

    This model returns the SEP token embedding, which can be used for:
    - Vector databases
    - Semantic search
    - Similarity computation

    The SEP token embedding represents the joint encoding of the path and text.

    Usage:
        ```python
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            "your-username/swipe-model",
            trust_remote_code=True
        )

        # Get embeddings
        outputs = model(path_coords=paths, char_tokens=tokens)
        embeddings = outputs.pooler_output  # SEP token embeddings
        ```

    Args:
        config (SwipeTransformerConfig or SwipeCrossEncoderConfig): Model configuration
    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Import existing components
        from ..models.embeddings import MixedEmbedding

        # Embeddings
        self.embeddings = MixedEmbedding(
            vocab_size=config.vocab_size,
            max_path_len=config.max_path_len,
            max_char_len=config.max_char_len,
            d_model=config.d_model,
            dropout=config.dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LayerNorm
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            enable_nested_tensor=False,
        )

        # Initialize weights
        self.post_init()

    def forward(
        self,
        path_coords: torch.Tensor,
        char_tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
        output_hidden_states: bool | None = None,
    ):
        """
        Forward pass that returns embeddings.

        Args:
            path_coords (torch.Tensor): Path coordinates [batch, path_len, 3]
            char_tokens (torch.Tensor): Character tokens [batch, char_len]
            attention_mask (torch.Tensor, optional): Attention mask [batch, seq_len]
            return_dict (bool, optional): Whether to return ModelOutput object
            output_hidden_states (bool, optional): Whether to output all hidden states

        Returns:
            BaseModelOutputWithPooling with:
                - last_hidden_state: Full sequence hidden states [batch, seq_len, d_model]
                - pooler_output: SEP token embeddings [batch, d_model]
                - hidden_states: Tuple of hidden states (if output_hidden_states=True)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size = path_coords.shape[0]
        device = path_coords.device

        # Create [CLS] and [SEP] tokens
        cls_token = torch.full(
            (batch_size, 1), fill_value=self.config.cls_token_id, dtype=torch.long, device=device
        )
        sep_token = torch.full(
            (batch_size, 1), fill_value=self.config.sep_token_id, dtype=torch.long, device=device
        )

        # Get embeddings
        embeddings = self.embeddings(path_coords, char_tokens, cls_token, sep_token)

        # Prepare attention mask
        if attention_mask is not None:
            src_key_padding_mask = attention_mask == 0
        else:
            src_key_padding_mask = None

        # Encode (batch_first=True is set in TransformerEncoderLayer)
        hidden_states = self.encoder(embeddings, src_key_padding_mask=src_key_padding_mask)

        # Extract SEP token embedding (pooler output)
        # SEP is at position 1 + path_len
        path_len = path_coords.shape[1]
        sep_position = 1 + path_len
        pooler_output = hidden_states[:, sep_position, :]  # [batch, d_model]

        if not return_dict:
            return (hidden_states, pooler_output)

        return BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooler_output,
            hidden_states=(hidden_states,) if output_hidden_states else None,
        )
