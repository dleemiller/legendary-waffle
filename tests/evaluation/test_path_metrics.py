import torch

from swipealot.evaluation.path import evaluate_path_reconstruction_masked_mse


class _SimpleDataset:
    def __init__(self, items: list[dict]):
        self._items = items

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            subset = self._items[idx]
            return {k: [item[k] for item in subset] for k in subset[0].keys()}
        return self._items[idx]


class _IdentityPathModel:
    def eval(self):
        return self

    def __call__(self, *, path_coords, input_ids, attention_mask, return_dict: bool = True):
        class _Out:
            def __init__(self, path_logits):
                self.path_logits = path_logits

        return _Out(path_logits=path_coords)


class _Processor:
    def __init__(self, *, max_path_len: int, path_input_dim: int):
        self.max_path_len = max_path_len
        self.max_char_len = 1
        self._path_input_dim = path_input_dim

    def __call__(self, *, path_coords, text, return_tensors: str):
        assert return_tensors == "pt"
        path = torch.tensor(path_coords, dtype=torch.float32)
        if path.ndim == 2:
            path = path.unsqueeze(0)
        batch_size = path.shape[0]

        input_ids = torch.zeros((batch_size, 1), dtype=torch.long)
        attention_mask = torch.ones(
            (batch_size, 1 + int(self.max_path_len) + 1 + 1), dtype=torch.long
        )
        return {"path_coords": path, "input_ids": input_ids, "attention_mask": attention_mask}


def test_path_reconstruction_uses_path_logits_and_masks_all_points():
    dataset = _SimpleDataset(
        [
            {"word": "a", "data": [[1.0, 1.0], [1.0, 1.0]]},
            {"word": "b", "data": [[1.0, 1.0], [1.0, 1.0]]},
        ]
    )
    model = _IdentityPathModel()
    processor = _Processor(max_path_len=2, path_input_dim=2)

    metrics = evaluate_path_reconstruction_masked_mse(
        model=model,
        processor=processor,
        dataset=dataset,
        device=torch.device("cpu"),
        batch_size=2,
        mask_prob=1.0,
        mse_dims=None,
        seed=0,
    )

    assert metrics.n_samples == 1
    assert metrics.masked_mse == 1.0
