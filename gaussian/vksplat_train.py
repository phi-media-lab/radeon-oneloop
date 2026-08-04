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


FROZEN_MEANS_LR = 1.0e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(
    root: Path,
    image_dir: str,
    sparse_dir: str,
    *,
    min_images: int = 8,
    mask_dir: str | None = None,
) -> dict[str, Any]:
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
    if len(image_paths) < min_images:
        raise ValueError(
            f"dataset requires at least {min_images} images, found {len(image_paths)}"
        )
    mask_paths: list[Path] = []
    if mask_dir is not None:
        masks = root / mask_dir
        if not masks.is_dir():
            raise FileNotFoundError(f"missing mask directory: {masks}")
        for image_path in image_paths:
            mask_path = masks / f"{image_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(f"missing mask for {image_path.name}: {mask_path}")
            mask_paths.append(mask_path)
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
        (str(path.relative_to(root)), sha256(path))
        for path in image_paths + mask_paths + model_paths
    ]
    dataset_hash = hashlib.sha256(
        "".join(f"{value}  {name}\n" for name, value in ledger).encode()
    ).hexdigest()
    return {
        "dataset_hash": dataset_hash,
        "images": len(image_paths),
        "masks": len(mask_paths),
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
    # The pinned binding concatenates ``generated/...`` directly onto the
    # shader-directory argument. Its upstream trainer supplies one trailing
    # separator, which is consumed by that concatenation on this build.
    # Preserve the clean upstream checkout and adapt only the shader lookup at
    # runtime by supplying ``shader//``.
    upstream_join_dir = module.join_dir

    def shader_safe_join_dir(parent: str, child: str) -> str:
        value = upstream_join_dir(parent, child)
        if child == "shader":
            return value.rstrip(os.sep) + os.sep + os.sep
        return value

    module.join_dir = shader_safe_join_dir
    return module


def freeze_geometry_learning_rates(config: Any) -> dict[str, float]:
    """Freeze visual-hull centers and shapes while retaining appearance fitting."""

    required = ("means_lr", "means_lr_final", "scales_lr", "quats_lr")
    missing = [name for name in required if not hasattr(config, name)]
    if missing:
        raise AttributeError(f"trainer config is missing geometry rates: {missing}")
    original = {name: float(getattr(config, name)) for name in required}
    # VkSplat's exponential center schedule evaluates log(lr). Exact zero
    # therefore creates NaNs even though the optimizer should be frozen. Use a
    # constant, negligible positive center rate and exact zero for shape rates.
    config.means_lr = FROZEN_MEANS_LR
    config.means_lr_final = FROZEN_MEANS_LR
    config.scales_lr = 0.0
    config.quats_lr = 0.0
    return original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-dir", default="images")
    parser.add_argument("--sparse-dir", default="sparse/0")
    parser.add_argument("--mask-dir")
    parser.add_argument("--min-images", type=int, default=8)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument(
        "--strategy", choices=("default", "mcmc"), default="default"
    )
    parser.add_argument("--freeze-higher-sh", action="store_true")
    parser.add_argument(
        "--freeze-geometry",
        action="store_true",
        help="Keep visual-hull means/scales/quaternions fixed and fit appearance only.",
    )
    parser.add_argument("--disable-refinement", action="store_true")
    parser.add_argument("--init-scale", type=float)
    parser.add_argument("--scale-reg", type=float)
    parser.add_argument("--opacity-reg", type=float)
    parser.add_argument("--eval-interval", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--host-role", default="unspecified_nonformal")
    args = parser.parse_args()
    if args.steps <= 0 or args.eval_interval <= 1 or args.min_images <= 0:
        raise ValueError("steps must be positive and eval-interval must exceed one")
    if args.formal and args.host_role != "radeon_c_gpu0_gfx1100_formal":
        raise ValueError("formal VkSplat evidence is restricted to radeon-c GPU0 gfx1100")
    if not args.formal and args.host_role.rsplit("_", 1)[-1] == "formal":
        raise ValueError("a nonformal run cannot use a formal host-role label")
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_report = validate_dataset(
        dataset,
        args.image_dir,
        args.sparse_dir,
        min_images=args.min_images,
        mask_dir=args.mask_dir,
    )
    trainer = load_trainer(args.source.resolve())
    # VkSplat's upstream training loop shuffles image indices with Python's
    # module-level random generator. Bind both Python and NumPy generators to
    # the recorded seed before constructing or running the trainer.
    trainer.random.seed(args.seed)
    trainer.np.random.seed(args.seed)
    config_type = (
        trainer.MCMCTrainerConfig
        if args.strategy == "mcmc"
        else trainer.TrainerConfig
    )
    config = config_type()
    original_geometry_learning_rates = None
    if args.freeze_higher_sh:
        config.features_rest_lr = 0.0
    if args.freeze_geometry:
        original_geometry_learning_rates = freeze_geometry_learning_rates(config)
    if args.disable_refinement:
        config.refine_start_iter = args.steps + 1
        config.refine_stop_iter = 0
    if args.init_scale is not None:
        if args.init_scale <= 0:
            raise ValueError("init-scale must be positive")
        config.init_scale = args.init_scale
    if args.scale_reg is not None:
        if args.scale_reg < 0:
            raise ValueError("scale-reg cannot be negative")
        config.scale_reg = args.scale_reg
    if args.opacity_reg is not None:
        if args.opacity_reg < 0:
            raise ValueError("opacity-reg cannot be negative")
        config.opacity_reg = args.opacity_reg
    config.enable_viewer = False
    config.output_dir = str(output)
    config.output_ply = str(output / "splat.ply")
    config.dataset_dir = str(dataset)
    config.image_dir = str((dataset / args.image_dir).resolve()) + os.sep
    config.sparse_dir = str((dataset / args.sparse_dir).resolve()) + os.sep
    config.mask_dir = (
        str((dataset / args.mask_dir).resolve()) + os.sep
        if args.mask_dir is not None
        else ""
    )
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
        "formal": bool(args.formal),
        "host_role": args.host_role,
        "backend": "vksplat_vulkan",
        "strategy": args.strategy,
        "optimization_profile": {
            "freeze_higher_sh": bool(args.freeze_higher_sh),
            "features_rest_lr": config.features_rest_lr,
            "freeze_geometry": bool(args.freeze_geometry),
            "original_geometry_learning_rates": original_geometry_learning_rates,
            "means_lr": config.means_lr,
            "means_lr_final": config.means_lr_final,
            "scales_lr": config.scales_lr,
            "quats_lr": config.quats_lr,
            "disable_refinement": bool(args.disable_refinement),
            "refine_start_iter": config.refine_start_iter,
            "refine_stop_iter": config.refine_stop_iter,
            "init_scale": config.init_scale,
            "scale_reg": config.scale_reg,
            "opacity_reg": config.opacity_reg,
        },
        "steps": args.steps,
        "seed": args.seed,
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
