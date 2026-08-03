"""Stratified ACT action-reconstruction diagnostic for paired checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from evaluation.policy_latency import latency_summary


def evenly_spaced_indices(values: list[int], limit: int) -> list[int]:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    ordered = sorted(values)
    if len(ordered) <= limit:
        return ordered
    if limit == 1:
        return [ordered[len(ordered) // 2]]
    return [ordered[position * (len(ordered) - 1) // (limit - 1)] for position in range(limit)]


def scalar_summary(values: list[float]) -> dict[str, float | int]:
    summary = latency_summary(values)
    return {
        "samples": summary["samples"],
        "mean": summary["mean_ms"],
        "p50": summary["p50_ms"],
        "p95": summary["p95_ms"],
        "p99": summary["p99_ms"],
        "min": summary["min_ms"],
        "max": summary["max_ms"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--samples-per-role", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader, Subset

    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    from radeon_oneloop.train_command import assert_single_radeon

    hardware = assert_single_radeon()
    checkpoint = args.checkpoint.resolve()
    dataset_root = args.dataset_root.resolve()
    targets_path = dataset_root / "oneloop/phase_targets.parquet"
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    if not targets_path.is_file():
        raise FileNotFoundError(targets_path)

    cfg = TrainPipelineConfig.from_pretrained(checkpoint)
    cfg.dataset.root = dataset_root
    cfg.dataset.video_backend = "pyav"
    cfg.policy.device = "cuda"
    cfg.policy.pretrained_path = checkpoint
    dataset = make_dataset(cfg)
    policy = make_policy(cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    targets = pd.read_parquet(targets_path, columns=["index", "segment_role"])
    roles = sorted(str(value) for value in targets["segment_role"].unique())
    result_roles: dict[str, Any] = {}
    with torch.inference_mode():
        for role in roles:
            population = [int(value) for value in targets.loc[targets["segment_role"] == role, "index"]]
            indices = evenly_spaced_indices(population, args.samples_per_role)
            loader = DataLoader(
                Subset(dataset, indices),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            chunk_l1: list[float] = []
            first_action_l1: list[float] = []
            for batch in loader:
                batch = preprocessor(batch)
                predicted = policy.predict_action_chunk(batch)
                target = batch["action"]
                valid = ~batch["action_is_pad"]
                absolute_error = (predicted - target).abs()
                valid_expanded = valid.unsqueeze(-1).expand_as(absolute_error)
                per_sample = (absolute_error * valid_expanded).sum(dim=(1, 2)) \
                    / valid_expanded.sum(dim=(1, 2)).clamp_min(1)
                first = absolute_error[:, 0, :].mean(dim=1)
                chunk_l1.extend(float(value) for value in per_sample.cpu())
                first_action_l1.extend(float(value) for value in first.cpu())
            result_roles[role] = {
                "population_frames": len(population),
                "sampled_frames": len(indices),
                "index_first": indices[0],
                "index_last": indices[-1],
                "normalized_chunk_l1": scalar_summary(chunk_l1),
                "normalized_first_action_l1": scalar_summary(first_action_l1),
            }

    result = {
        "schema_version": "radeon_oneloop.action_reconstruction.v1",
        "diagnostic_scope": "deterministic_stratified_training_set_frames",
        "checkpoint": str(checkpoint),
        "dataset_root": str(dataset_root),
        "targets_path": str(targets_path),
        "samples_per_role_limit": args.samples_per_role,
        "roles": result_roles,
        "hardware": hardware,
        "task_success": None,
        "limitations": [
            "The formal dataset has no independent validation split.",
            "These reconstruction errors are training-set diagnostics, not task-success estimates.",
            "Closed-loop task performance must be reported separately.",
        ],
    }
    output = args.output
    run_dir = os.environ.get("ONELOOP_RUN_DIR")
    if output is None and run_dir:
        output = Path(run_dir, "metrics.json")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
