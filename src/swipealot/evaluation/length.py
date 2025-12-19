"""Length prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from swipealot.evaluation.common import iter_dataset_batches
from swipealot.text_utils import swipable_length


@dataclass
class LengthEvalResult:
    mae: float
    rmse: float
    acc_exact_rounded: float
    acc_within_1: float
    acc_within_2: float
    count: int


def evaluate_length_dataset(
    *,
    model,
    processor,
    dataset,
    device: torch.device,
    batch_size: int,
) -> LengthEvalResult:
    preds: list[float] = []
    targets: list[int] = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(
            iter_dataset_batches(dataset, batch_size),
            total=(len(dataset) + batch_size - 1) // batch_size,
            desc="Length eval",
        ):
            words = batch["word"]
            paths = batch["data"]

            inputs = processor(path_coords=paths, text=None, return_tensors="pt")
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()
            }
            outputs = model(**inputs, return_dict=True)

            length_logits = getattr(outputs, "length_logits", None)
            if length_logits is None:
                raise RuntimeError(
                    "Model did not return `length_logits`; is predict_length disabled?"
                )

            length_pred = length_logits.reshape(-1).detach().cpu().numpy().astype(np.float64)
            preds.extend(length_pred.tolist())
            targets.extend([swipable_length(w) for w in words])

    pred_arr = np.array(preds, dtype=np.float64)
    tgt_arr = np.array(targets, dtype=np.float64)
    err = pred_arr - tgt_arr

    mae = float(np.mean(np.abs(err))) if len(err) else 0.0
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else 0.0

    pred_round = np.round(pred_arr).clip(min=0.0)
    acc_exact = float(np.mean(pred_round == tgt_arr)) if len(err) else 0.0
    acc_w1 = float(np.mean(np.abs(pred_round - tgt_arr) <= 1.0)) if len(err) else 0.0
    acc_w2 = float(np.mean(np.abs(pred_round - tgt_arr) <= 2.0)) if len(err) else 0.0

    return LengthEvalResult(
        mae=mae,
        rmse=rmse,
        acc_exact_rounded=acc_exact,
        acc_within_1=acc_w1,
        acc_within_2=acc_w2,
        count=len(targets),
    )
