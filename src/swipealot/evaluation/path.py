"""Path reconstruction evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tqdm import tqdm

from swipealot.evaluation.common import iter_dataset_batches


@dataclass
class PathReconstructionMetrics:
    masked_mse: float
    n_samples: int


def evaluate_path_reconstruction_masked_mse(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
    mask_prob: float,
    mse_dims: list[int] | None,
    seed: int,
) -> PathReconstructionMetrics:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    masked_mse_sum = 0.0
    masked_mse_count = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Path reconstruction (masked MSE)",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=words, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }

            path_coords = inputs["path_coords"]  # [B, P, D]
            path_mask = inputs["attention_mask"][:, 1 : 1 + int(processor.max_path_len)]  # [B, P]

            valid = path_mask.to(torch.bool)
            if not valid.any():
                continue

            r = torch.rand(valid.shape, generator=rng, device="cpu")
            masked = (r < float(mask_prob)) & valid.cpu()
            masked = masked.to(device)

            masked_path = path_coords.clone()
            masked_path[masked] = 0.0

            outputs = model(
                path_coords=masked_path,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                return_dict=True,
            )
            recon = getattr(outputs, "path_logits", None)
            if recon is None:
                recon = getattr(outputs, "path_coords_pred", None)
            if recon is None:
                raise RuntimeError(
                    "Model did not return `path_logits`/`path_coords_pred`; is predict_path disabled?"
                )

            if mse_dims is None:
                pred = recon[masked]
                tgt = path_coords[masked]
            else:
                pred = recon[:, :, mse_dims][masked]
                tgt = path_coords[:, :, mse_dims][masked]

            if pred.numel() == 0:
                continue

            mse = torch.mean((pred - tgt) ** 2).item()
            masked_mse_sum += float(mse)
            masked_mse_count += 1

    return PathReconstructionMetrics(
        masked_mse=(masked_mse_sum / masked_mse_count if masked_mse_count else 0.0),
        n_samples=masked_mse_count,
    )
