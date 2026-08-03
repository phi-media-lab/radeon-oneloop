#!/usr/bin/env python3
"""Run pinned VkSplat on a validated COLMAP workspace dataset."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(root: Path, image_dir: str, sparse_dir: str) -> dict[str, Any]:
    images = root / image_dir
    sparse = root / sparse_dir
    image_paths = (
        sorted(
            path
            for path in images.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if images.is_dir()
        else []
    )
    if len(image_paths) < 8:
        raise ValueError(
            f"workspace capture requires at least 8 images, found {len(image_paths)}"
        )
    model_format = None
    model_paths: list[Path] = []
    for suffix in ("bin", "txt"):
        candidates = [
            sparse / f"{name}.{suffix}" for name in ("cameras", "images", "points3D")
        ]
        if all(path.is_file() for path in candidates):
            model_format, model_paths = suffix, candidates
            break
    if model_format is None:
        raise FileNotFoundError(f"missing complete COLMAP model under {sparse}")
    ledger = [
        (str(path.relative_to(root)), sha256(path)) for path in image_paths + model_paths
    ]
    dataset_hash = hashlib.sha256(
        "".join(f"{value}  {name}\n" for name, value in ledger).encode()
    ).hexdigest()
    return {
        "dataset_hash": dataset_hash,
        "images": len(image_paths),
        "model_format": model_format,
        "files": [{"path": name, "sha256": value} for name, value in ledger],
    }


def load_trainer(source: Path) -> Any:
    trainer_path = source / "vksplat" / "simple_trainer.py"
    if not trainer_path.is_file():
        raise FileNotFoundError(trainer_path)
    spec = importlib.util.spec_from_file_location(
        "oneloop_pinned_vksplat_trainer", trainer_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument(
        "--strategy", choices=("default", "mcmc"), default="default"
    )
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_interval <= 1:
        raise ValueError("steps must be positive and eval-interval must exceed one")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_report = validate_dataset(dataset, args.image_dir, args.sparse_dir)
    trainer = load_trainer(args.source.resolve())
    config_type = (
        trainer.MCMCTrainerConfig
        if args.strategy == "mcmc"
        else trainer.TrainerConfig
    )
    config = config_type()
    config.enable_viewer = False
    config.output_dir = str(output)
    config.output_ply = str(output / "splat.ply")
    config.dataset_dir = str(dataset)
    config.image_dir = str((dataset / args.image_dir).resolve()) + os.sep
    config.sparse_dir = str((dataset / args.sparse_dir).resolve()) + os.sep
    config.mask_dir = ""
    config.train_steps = args.steps
    config.max_steps = args.steps
    config.eval_interval = args.eval_interval
    config.save_train_renders = False
    trainer.train(config)
    if args.evaluate:
        trainer.eval(config)
    required = [
        output / "config.json",
        output / "train.json",
        output / "splat.ply",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"VkSplat did not produce required artifacts: {missing}")
    train = json.loads((output / "train.json").read_text())
    report = {
        "schema_version": "radeon_oneloop.gaussian_train.v1",
        "backend": "vksplat_vulkan",
        "strategy": args.strategy,
        "steps": args.steps,
        "dataset": dataset_report,
        "num_splats": train.get("num_splats"),
        "elapsed_seconds": train.get("time_elapsed"),
        "vram_bytes": train.get("vram"),
        "peak_vram_bytes": train.get("peak_vram"),
        "artifacts": {path.name: sha256(path) for path in required},
        "evaluation": (
            json.loads((output / "eval.json").read_text())
            if (output / "eval.json").is_file()
            else None
        ),
    }
    payload = json.dumps(report, indent=2) + "\n"
    (output / "oneloop_metrics.json").write_text(payload, encoding="utf-8")
    if os.environ.get("ONELOOP_RUN_DIR"):
        Path(os.environ["ONELOOP_RUN_DIR"], "metrics.json").write_text(
            payload, encoding="utf-8"
        )
    print(payload, end="")


if __name__ == "__main__":
    main()
