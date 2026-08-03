#!/usr/bin/env python3
"""Derive paired training summaries from public formal evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


PROGRESS = re.compile(
    r"ot_train\.py:\d+ step:(?P<label>\S+) .*?"
    r"loss:(?P<loss>[-+0-9.eE]+) grdn:(?P<gradient>[-+0-9.eE]+) .*?"
    r"updt_s:(?P<update>[-+0-9.eE]+) data_s:(?P<data>[-+0-9.eE]+)"
)


def parse_progress(log: str, *, log_frequency: int = 50) -> list[dict[str, float | int | str]]:
    points: list[dict[str, float | int | str]] = []
    for line in log.splitlines():
        match = PROGRESS.search(line)
        if match:
            points.append(
                {
                    "step": (len(points) + 1) * log_frequency,
                    "displayed_step": match.group("label"),
                    "loss": float(match.group("loss")),
                    "gradient_norm": float(match.group("gradient")),
                    "update_seconds": float(match.group("update")),
                    "data_seconds": float(match.group("data")),
                }
            )
    if not points:
        raise ValueError("training log contains no progress records")
    return points


def parse_gpu_samples(value: str) -> dict[str, float | int]:
    utilization: list[int] = []
    memory: list[int] = []
    for line in value.splitlines()[1:]:
        if "\t" not in line:
            continue
        sample = line.split("\t", 1)[1].strip().rstrip(";")
        for record in sample.split(";"):
            fields = [field.strip() for field in record.split(",")]
            if len(fields) >= 3 and fields[0].startswith("card"):
                utilization.append(int(fields[1]))
                memory.append(int(fields[2]))
    if not utilization:
        raise ValueError("GPU sample file contains no device records")
    return {
        "samples": len(utilization),
        "gpu_utilization_mean_percent": sum(utilization) / len(utilization),
        "gpu_utilization_peak_percent": max(utilization),
        "vram_allocated_mean_percent": sum(memory) / len(memory),
        "vram_allocated_peak_percent": max(memory),
    }


def summarize_run(path: Path) -> dict[str, Any]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    hardware = json.loads((path / "hardware.json").read_text(encoding="utf-8"))
    # LeRobot's structured progress logger writes to stderr, while our command
    # wrapper and result JSON write to stdout.  Read both so the public
    # summarizer follows the evidence rather than relying on a stream choice.
    training_log = "\n".join(
        (path / name).read_text(encoding="utf-8")
        for name in ("stdout.log", "stderr.log")
    )
    progress = parse_progress(training_log)
    gpu = parse_gpu_samples((path / "gpu_samples.tsv").read_text(encoding="utf-8"))
    configured_steps = int(metrics["configured_steps"])
    if progress[-1]["step"] != configured_steps:
        raise ValueError(
            f"terminal progress step {progress[-1]['step']} does not match {configured_steps}"
        )
    if metrics.get("exit_code") != 0 or manifest.get("status") != "done":
        raise ValueError(f"run is not successful: {path}")
    ledger_path = path / "checkpoint_tree.json"
    checkpoint = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.is_file() else None
    result = {
        "job_id": manifest["job_id"],
        "git_commit": manifest["git_commit"],
        "config_sha256": manifest["config_hash"],
        "dataset_sha256": manifest["dataset_hash"],
        "seed": manifest["seed"],
        "gpu_uid": manifest["gpu_uid"],
        "configured_steps": configured_steps,
        "elapsed_seconds": float(metrics["elapsed_seconds"]),
        "terminal_loss": progress[-1]["loss"],
        "terminal_gradient_norm": progress[-1]["gradient_norm"],
        "progress": progress,
        "gpu_samples": gpu,
        "total_vram_bytes": int(hardware["total_memory"]),
        "peak_sampled_vram_bytes_upper_bound": int(
            int(hardware["total_memory"]) * gpu["vram_allocated_peak_percent"] / 100
        ),
        "checkpoint": checkpoint,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--phase-aware", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "schema_version": "radeon_oneloop.paired_training_summary.v1",
        "baseline": summarize_run(args.baseline.resolve()),
        "phase_aware": summarize_run(args.phase_aware.resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
