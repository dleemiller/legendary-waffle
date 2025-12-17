import torch

from swipealot.analysis import compute_char_to_path_attention_profile
from swipealot.huggingface import SwipeTransformerConfig, SwipeTransformerModel


def test_output_attentions_shapes():
    config = SwipeTransformerConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        vocab_size=25,
        max_path_len=8,
        max_char_len=6,
        path_input_dim=6,
        predict_char=True,
        predict_path=False,
        predict_length=False,
    )
    model = SwipeTransformerModel(config)

    batch_size = 2
    path_coords = torch.randn(batch_size, config.max_path_len, config.path_input_dim)
    input_ids = torch.randint(6, config.vocab_size, (batch_size, config.max_char_len))
    seq_len = 1 + config.max_path_len + 1 + config.max_char_len
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    outputs = model(
        input_ids=input_ids,
        path_coords=path_coords,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )

    assert outputs.attentions is not None
    assert len(outputs.attentions) == config.n_layers
    for attn in outputs.attentions:
        assert attn.shape == (batch_size, config.n_heads, seq_len, seq_len)


def test_char_to_path_profile_renormalizes_and_respects_masks():
    config = SwipeTransformerConfig(
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_ff=64,
        dropout=0.0,
        vocab_size=25,
        max_path_len=8,
        max_char_len=6,
        path_input_dim=6,
        predict_char=True,
        predict_path=False,
        predict_length=False,
    )
    model = SwipeTransformerModel(config)

    batch_size = 2
    path_coords = torch.randn(batch_size, config.max_path_len, config.path_input_dim)
    input_ids = torch.randint(6, config.vocab_size, (batch_size, config.max_char_len))
    input_ids[:, -1] = config.eos_token_id

    seq_len = 1 + config.max_path_len + 1 + config.max_char_len
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    # Mask out the last 3 path positions.
    attention_mask[:, 1 + (config.max_path_len - 3) : 1 + config.max_path_len] = 0

    outputs = model(
        input_ids=input_ids,
        path_coords=path_coords,
        attention_mask=attention_mask,
        output_attentions=True,
        return_dict=True,
    )

    result = compute_char_to_path_attention_profile(
        outputs.attentions,
        path_len=config.max_path_len,
        char_len=config.max_char_len,
        head_aggregation="mean",
        layer_aggregation="mean_last_k",
        last_k_layers=2,
        renormalize_over_path=True,
        attention_mask=attention_mask,
        input_ids=input_ids,
        exclude_token_ids={config.pad_token_id, config.eos_token_id, config.mask_token_id},
    )

    profile = result["profile"]
    query_keep = result["query_keep_mask"]
    path_key_mask = result["path_key_mask"]

    assert profile.shape == (batch_size, config.max_char_len, config.max_path_len)
    assert query_keep.shape == (batch_size, config.max_char_len)
    assert path_key_mask.shape == (batch_size, config.max_path_len)

    # Masked path positions must have zero mass.
    assert torch.all(profile[:, :, -3:] == 0)

    # For kept queries with at least one valid path key, the profile should be a distribution over path.
    sums = profile.sum(dim=-1)
    for b in range(batch_size):
        for c in range(config.max_char_len):
            if query_keep[b, c] and path_key_mask[b].any():
                assert torch.isclose(sums[b, c], torch.tensor(1.0), atol=1e-4)
            else:
                assert torch.isclose(sums[b, c], torch.tensor(0.0), atol=1e-6)
