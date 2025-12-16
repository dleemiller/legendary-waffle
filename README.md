# SwipeALot

Multimodal transformer for swipe keyboard prediction.

## Setup

```bash
uv sync --extra cu128
```

## Commands

### Train

```bash
uv run train --config configs/base.yaml
```

Train model with specified config.

### Evaluate

```bash
uv run evaluate --checkpoint checkpoints/base/best.pt --split test
```

Evaluate checkpoint on dataset split.

### Convert to HuggingFace

```bash
uv run convert-hf --checkpoint checkpoints/base/best.pt --output hf_model
```

Convert training checkpoint to HuggingFace format.

### Test HuggingFace Model

```bash
uv run test-hf --model-path hf_model --num-samples 100
```

Test converted HuggingFace model.

### Generate Attention Map

```bash
uv run attention-map --checkpoint checkpoints/base/best.pt --word "hello" --output attention.png
```

Visualize attention patterns.

## Test

```bash
uv run pytest
```

Run all tests.

```bash
uv run pytest tests/huggingface/
```

Run specific test directory.

## Lint

```bash
uv run ruff check .
```

Check code style.

```bash
uv run ruff check --fix .
```

Auto-fix linting errors.

## Model

HuggingFace model: [dleemiller/SwipeALot-base](https://huggingface.co/dleemiller/SwipeALot-base)

Dataset: [futo-org/swipe.futo.org](https://huggingface.co/datasets/futo-org/swipe.futo.org)
