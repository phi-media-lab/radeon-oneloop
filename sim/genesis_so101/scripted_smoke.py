#!/usr/bin/env python3
"""Run deterministic dual-arm sweeps and headless camera checks on AMD GPU."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch

from radeon_oneloop.contracts import CAMERA_KEYS, IMAGE_SHAPE_HWC

from .scene import HOME_ACTION, build


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    task, handles = build(args.asset_root.resolve(), seed=args.seed, show_viewer=False)
    build_seconds = time.perf_counter() - started
    step_times = []
    rendered = None
    for step in range(args.steps):
        action = list(HOME_ACTION)
        joint = (step // 140) % 5
        offset = 8.0 * math.sin(2.0 * math.pi * (step % 140) / 140.0)
        action[joint] += offset
        action[6 + joint] -= offset
        begin = time.perf_counter()
        observation = task.step(action, render=(step in (0, args.steps - 1)))
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - begin)
        if CAMERA_KEYS[0] in observation:
            rendered = observation
    if rendered is None:
        rendered = task.observe(render=True)
    for key in CAMERA_KEYS:
        image = np.asarray(rendered[key])
        if image.shape != IMAGE_SHAPE_HWC:
            raise RuntimeError(f"{key} shape mismatch: {image.shape}")
        iio.imwrite(args.output / (key.rsplit(".", 1)[-1] + ".png"), image.astype(np.uint8))
    state = np.asarray(rendered["observation.state"])
    if state.shape != (12,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid state: shape={state.shape}")
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema_version": "radeon_oneloop.genesis_smoke.v1",
        "backend": str(handles.gs.backend),
        "device": str(handles.gs.device),
        "torch_device": torch.cuda.get_device_name(0),
        "gcn_arch": str(getattr(props, "gcnArchName", "")),
        "steps": args.steps,
        "build_seconds": build_seconds,
        "step_ms": {
            "mean": 1000.0 * float(np.mean(step_times)),
            "p50": 1000.0 * float(np.percentile(step_times, 50)),
            "p95": 1000.0 * float(np.percentile(step_times, 95)),
            "p99": 1000.0 * float(np.percentile(step_times, 99)),
        },
        "observation": {
            "state_shape": list(state.shape),
            "camera_shapes": {
                key: list(np.asarray(rendered[key]).shape) for key in CAMERA_KEYS
            },
        },
        "task_success": task.success(),
        "task_success_note": (
            "Joint-sweep smoke validates the scene and contracts; it is not a handover evaluation."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
