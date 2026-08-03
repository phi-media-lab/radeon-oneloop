"""Benchmark ACT inference on one real dataset observation and one Radeon."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("latency sample list is empty")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def latency_summary(values: list[float]) -> dict[str, float | int]:
    return {
        "samples": len(values),
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def to_robot_observation(item: dict[str, Any], input_keys: list[str]) -> dict[str, Any]:
    import numpy as np
    import torch

    observation: dict[str, Any] = {}
    for key in input_keys:
        value = item[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"dataset item {key!r} is not a tensor")
        value = value.detach().cpu()
        if "image" in key:
            if value.ndim != 3 or value.shape[0] not in (1, 3, 4):
                raise ValueError(f"unexpected image shape for {key}: {tuple(value.shape)}")
            observation[key] = (
                value.clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).numpy()
            )
        else:
            observation[key] = value.to(torch.float32).numpy().astype(np.float32, copy=False)
    return observation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--dataset-repo-id",
        default="phi-media-lab/radeon-oneloop-formal-handover-v1",
    )
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.warmup < 1 or args.iterations < 2:
        raise ValueError("warmup must be positive and iterations must be at least two")

    import numpy as np
    import torch
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.utils.control_utils import predict_action

    from radeon_oneloop.train_command import assert_single_radeon

    hardware = assert_single_radeon()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)

    dataset = LeRobotDataset(
        args.dataset_repo_id,
        root=args.dataset_root.resolve(),
        video_backend="pyav",
    )
    if not 0 <= args.frame_index < len(dataset):
        raise IndexError(args.frame_index)

    policy_cfg = PreTrainedConfig.from_pretrained(
        str(checkpoint),
        cli_overrides=["--device=cuda"],
    )
    policy_cfg.pretrained_path = str(checkpoint)
    policy = make_policy(policy_cfg, ds_meta=dataset.meta)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(checkpoint),
        dataset_stats=dataset.meta.stats,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    input_keys = [
        key
        for key in policy.config.input_features
        if key.startswith("observation.") and key in dataset.meta.features
    ]
    observation = to_robot_observation(dataset[args.frame_index], input_keys)
    device = torch.device("cuda")

    def invoke(*, reset: bool) -> tuple[float, Any]:
        if reset:
            policy.reset()
        torch.cuda.synchronize()
        started = time.perf_counter_ns()
        action = predict_action(
            observation,
            policy,
            device,
            preprocessor,
            postprocessor,
            bool(policy.config.use_amp),
        )
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        return elapsed_ms, action

    for _ in range(args.warmup):
        invoke(reset=True)

    torch.cuda.reset_peak_memory_stats()
    chunk_latencies: list[float] = []
    last_action = None
    for _ in range(args.iterations):
        elapsed, last_action = invoke(reset=True)
        chunk_latencies.append(elapsed)

    policy.reset()
    invoke(reset=False)
    dispatch_latencies: list[float] = []
    dispatch_count = min(args.iterations, int(policy.config.n_action_steps) - 1)
    for _ in range(dispatch_count):
        elapsed, last_action = invoke(reset=False)
        dispatch_latencies.append(elapsed)

    if last_action is None:
        raise RuntimeError("inference produced no action")
    action_np = last_action.detach().cpu().numpy()
    if not np.isfinite(action_np).all():
        raise RuntimeError("inference produced a non-finite action")

    result = {
        "schema_version": "radeon_oneloop.policy_latency.v1",
        "checkpoint": str(checkpoint),
        "dataset_root": str(args.dataset_root.resolve()),
        "dataset_repo_id": args.dataset_repo_id,
        "frame_index": args.frame_index,
        "input_keys": input_keys,
        "input_shapes": {key: list(value.shape) for key, value in observation.items()},
        "action_shape": list(action_np.shape),
        "action_finite": True,
        "action_chunk_horizon": int(policy.config.n_action_steps),
        "chunk_generation": latency_summary(chunk_latencies),
        "queued_action_dispatch": latency_summary(dispatch_latencies),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        "hardware": hardware,
        "task_success": None,
        "task_success_note": "Latency benchmark only; task success requires closed-loop rollout evidence.",
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
