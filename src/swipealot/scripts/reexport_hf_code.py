"""Re-export SwipeALot HuggingFace remote-code files into existing checkpoints.

This is useful when you fix logic in `src/swipealot/huggingface/*` (e.g. tokenizer
punctuation handling) and want all already-exported checkpoint folders to load the
updated code via `trust_remote_code=True`.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from swipealot.utils import configure_hf_env


@dataclass(frozen=True)
class CheckpointDir:
    path: Path
    has_weights: bool
    has_tokenizer: bool
    has_processor: bool


def _is_checkpoint_dir(path: Path) -> CheckpointDir | None:
    if not path.is_dir():
        return None
    config_path = path / "config.json"
    if not config_path.exists():
        return None

    has_weights = any((path / name).exists() for name in ["model.safetensors", "pytorch_model.bin"])
    if not has_weights:
        return None

    has_tokenizer = (path / "vocab.json").exists() or (path / "tokenizer_config.json").exists()
    has_processor = any(
        (path / name).exists() for name in ["processor_config.json", "preprocessor_config.json"]
    )
    return CheckpointDir(
        path=path,
        has_weights=has_weights,
        has_tokenizer=has_tokenizer,
        has_processor=has_processor,
    )


def _find_checkpoint_dirs(root: Path) -> list[CheckpointDir]:
    out: list[CheckpointDir] = []
    for config_path in root.rglob("config.json"):
        ckpt = _is_checkpoint_dir(config_path.parent)
        if ckpt is not None:
            out.append(ckpt)
    out.sort(key=lambda c: str(c.path))
    return out


def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _set_env(hf_home: Path, offline: bool) -> None:
    configure_hf_env(hf_home, offline=offline, overwrite=False, set_hub_cache=False)


def _reexport_one(checkpoint_dir: Path, *, dry_run: bool) -> None:
    from swipealot.huggingface import SwipeProcessor, SwipeTokenizer
    from swipealot.training.checkpoint_utils import prepare_checkpoint_for_hub

    if dry_run:
        return

    # 1) Copy updated python files (modeling/config/tokenizer/processor + deps) into the folder.
    prepare_checkpoint_for_hub(checkpoint_dir)

    # 2) Re-save tokenizer + processor configs (auto_map is handled by save_pretrained).
    tokenizer = None
    if (checkpoint_dir / "vocab.json").exists() or (
        checkpoint_dir / "tokenizer_config.json"
    ).exists():
        tokenizer = SwipeTokenizer.from_pretrained(str(checkpoint_dir))
        tokenizer.save_pretrained(str(checkpoint_dir))

    cfg = _read_json(checkpoint_dir / "config.json")
    max_path_len = int(cfg.get("max_path_len", 64))
    max_char_len = int(cfg.get("max_char_len", 38))
    path_input_dim = int(cfg.get("path_input_dim", 6))

    proc_cfg_path = (
        checkpoint_dir / "processor_config.json"
        if (checkpoint_dir / "processor_config.json").exists()
        else checkpoint_dir / "preprocessor_config.json"
    )
    if proc_cfg_path.exists():
        proc_cfg = _read_json(proc_cfg_path)
        path_resample_mode = str(proc_cfg.get("path_resample_mode", "time"))
    else:
        path_resample_mode = "time"

    if tokenizer is None:
        # Fallback: create a default tokenizer; vocab will be deterministic a-z + 0-9.
        tokenizer = SwipeTokenizer()
        tokenizer.save_pretrained(str(checkpoint_dir))

    processor = SwipeProcessor(
        tokenizer=tokenizer,
        max_path_len=max_path_len,
        max_char_len=max_char_len,
        path_input_dim=path_input_dim,
        path_resample_mode=path_resample_mode,
    )
    processor.save_pretrained(str(checkpoint_dir))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-export SwipeALot HuggingFace remote-code files into existing checkpoints"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="checkpoints",
        help="Root directory to scan (default: checkpoints)",
    )
    parser.add_argument(
        "--hf-home",
        type=str,
        default=None,
        help="Optional HF cache root to use (respects default HF env when omitted)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Enable offline mode (recommended; avoids network retries)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list discovered checkpoint directories; do not modify anything",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only process the first N checkpoint dirs (0 = no limit)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"--root not found: {root}")

    if args.hf_home is not None:
        hf_home = Path(args.hf_home)
        hf_home.mkdir(parents=True, exist_ok=True)
        _set_env(hf_home, bool(args.offline))
    elif args.offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    ckpts = _find_checkpoint_dirs(root)
    if args.limit and args.limit > 0:
        ckpts = ckpts[: int(args.limit)]

    if not ckpts:
        print(f"No checkpoint dirs found under {root}")
        return

    print(f"Found {len(ckpts)} checkpoint dirs under {root}")
    for ckpt in ckpts:
        flags = []
        if ckpt.has_tokenizer:
            flags.append("tokenizer")
        if ckpt.has_processor:
            flags.append("processor")
        flags_str = f" ({', '.join(flags)})" if flags else ""
        print(f"- {ckpt.path}{flags_str}")

    if args.dry_run:
        return

    for ckpt in ckpts:
        _reexport_one(ckpt.path, dry_run=False)

    print(f"Re-export complete for {len(ckpts)} checkpoint dirs.")


if __name__ == "__main__":
    main()
