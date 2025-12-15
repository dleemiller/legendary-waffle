# Swipe Keyboard Cross-Encoder

A multimodal transformer model for swipe keyboard path-to-word matching. This model takes swipe path coordinates (x, y, t) and candidate words as input and outputs a similarity score.

## Model Description

**Architecture:** Custom multimodal transformer with:
- **Path embedding**: Linear projection of (x, y, t) coordinates to embedding space
- **Character embedding**: Learned character embeddings for text
- **Positional embedding**: Learned position embeddings
- **Type embedding**: Token type embeddings to distinguish path vs text tokens
- **Transformer encoder**: Multi-layer transformer with pre-LayerNorm architecture
- **Classification head**: SEP token pooling + MLP for similarity scoring

**Sequence Structure:** `[CLS] + path_tokens + [SEP] + char_tokens`

**Training:** Trained on [futo-org/swipe.futo.org](https://huggingface.co/datasets/futo-org/swipe.futo.org) dataset using Multiple Negatives Ranking (MNR) loss.

## Model Parameters

Replace these values when uploading a specific model:

- **Hidden size (d_model)**: {d_model}
- **Number of layers (n_layers)**: {n_layers}
- **Number of heads (n_heads)**: {n_heads}
- **Feedforward dimension (d_ff)**: {d_ff}
- **Dropout**: {dropout}
- **Vocabulary size**: {vocab_size}
- **Max path length**: {max_path_len}
- **Max character length**: {max_char_len}

## Usage

This model requires `trust_remote_code=True` to load due to its custom architecture.

### Installation

```bash
pip install transformers torch
```

### Quick Start with Wrapper (Recommended)

The easiest way to use this model is with the `SwipeCrossEncoder` wrapper:

```python
from swipealot.huggingface import SwipeCrossEncoder

# Load model
model = SwipeCrossEncoder("your-username/swipe-cross-encoder")

# Swipe path coordinates [path_len, 3] where each point is (x, y, time)
path_coords = [
    [0.1, 0.2, 0.0],
    [0.15, 0.25, 0.1],
    [0.2, 0.3, 0.2],
    [0.25, 0.35, 0.3],
    # ... more points
]

# Candidate words
candidates = ["hello", "world", "help", "hold"]

# Get similarity scores
scores = model.predict([path_coords], candidates)
print(scores)  # [0.95, 0.12, 0.78, 0.45]

# Rank candidates
ranked = model.rank(path_coords, candidates, top_k=3)
print(ranked)  # [("hello", 0.95), ("help", 0.78), ("hold", 0.45)]
```

### Direct Usage with Transformers

```python
from transformers import AutoModelForSequenceClassification
import torch

# Load model and processor
model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/swipe-cross-encoder",
    trust_remote_code=True
)

# You'll need to manually create the processor
from swipealot.huggingface import SwipeTokenizer, SwipeProcessor

tokenizer = SwipeTokenizer.from_pretrained("your-username/swipe-cross-encoder")
processor = SwipeProcessor.from_pretrained("your-username/swipe-cross-encoder")

# Prepare inputs
path_coords = [[0.1, 0.2, 0.0], [0.15, 0.25, 0.1], [0.2, 0.3, 0.2]]
text = "hello"

inputs = processor(
    path_coords=path_coords,
    text=text,
    return_tensors="pt",
    padding=True,
)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    score = outputs.logits.squeeze(-1)
    print(f"Similarity score: {score.item()}")
```

### Batch Inference

```python
# Multiple paths and words
paths = [path1, path2, path3]
words = ["hello", "world", "test"]

# Process batch
inputs = processor(paths, words, return_tensors="pt", padding=True)

# Inference
with torch.no_grad():
    outputs = model(**inputs)
    scores = outputs.logits.squeeze(-1)
    print(scores)  # [score1, score2, score3]
```

## Training Details

- **Dataset**: futo-org/swipe.futo.org
- **Training split**: {train_split}
- **Validation split**: {val_split}
- **Batch size**: {batch_size}
- **Learning rate**: {learning_rate} (head), {encoder_learning_rate} (encoder)
- **Epochs**: {num_epochs}
- **Loss function**: Multiple Negatives Ranking (MNR) with temperature scaling
- **Negatives per sample**: {num_negatives}
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup + cosine decay

### Training Metrics

- **Final validation accuracy**: {val_accuracy}%
- **Best epoch**: {best_epoch}
- **Training time**: {training_time}

## Input Format

### Path Coordinates

Path coordinates should be normalized swipe traces where:
- **x, y**: Normalized screen coordinates in [0, 1] range
- **t**: Normalized timestamp in [0, 1] range

Each path is a sequence of (x, y, t) tuples:
```python
path = [
    [x1, y1, t1],
    [x2, y2, t2],
    ...
]
```

### Text

Text input should be lowercase strings. The model uses character-level tokenization.

## Limitations

- **Coordinate range**: Expects normalized coordinates in [0, 1]. Raw pixel coordinates must be normalized first.
- **Maximum path length**: {max_path_len} points. Longer paths will be truncated.
- **Maximum word length**: {max_char_len} characters. Longer words will be truncated.
- **Case sensitivity**: All text is lowercased. The model is case-insensitive.
- **Vocabulary**: Character-level tokenizer limited to dataset vocabulary.
- **Language**: Trained on English text only.

## Performance

Replace with actual performance metrics:

- **Accuracy on test set**: {test_accuracy}%
- **Top-3 accuracy**: {top3_accuracy}%
- **Top-5 accuracy**: {top5_accuracy}%
- **Inference speed**: {inference_speed} samples/sec (GPU: {gpu_type})

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{swipe-cross-encoder,
  author = {Your Name},
  title = {Swipe Keyboard Cross-Encoder},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/your-username/swipe-cross-encoder}
}
```

## License

Apache 2.0

## Acknowledgments

- Dataset: [FUTO Keyboard](https://keyboard.futo.org/)
- Built with [🤗 Transformers](https://huggingface.co/transformers/)
- Model architecture inspired by multimodal transformers (CLIP, LayoutLM)

## Contact

For questions or issues, please open an issue on the [GitHub repository](https://github.com/your-username/your-repo) or contact [your-email].
