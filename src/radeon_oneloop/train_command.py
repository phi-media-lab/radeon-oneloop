"""Generate and optionally execute the two frozen LeRobot ACT commands."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


class TrainingConfigError(ValueError):
    pass


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyYAML is required to load experiment configs") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TrainingConfigError(f"config is not an object: {path}")
    return value


def assert_fair_pair(baseline: dict[str, Any], phase: dict[str, Any]) -> None:
    """Fail if anything except the declared phase-aware settings differs."""
    keys = ("dataset", "policy", "optimizer", "training", "reproducibility")
    left = copy.deepcopy({key: baseline.get(key) for key in keys})
    right = copy.deepcopy({key: phase.get(key) for key in keys})
    for config in (left, right):
        config.get("training", {}).pop("output_dir", None)
    if left != right:
        differences = [key for key in keys if left.get(key) != right.get(key)]
        raise TrainingConfigError(f"formal pair differs outside method: {differences}")
    if baseline.get("method", {}).get("phase_aware") is not False:
        raise TrainingConfigError("baseline must set method.phase_aware=false")
    if phase.get("method", {}).get("phase_aware") is not True:
        raise TrainingConfigError("phase-aware config must set method.phase_aware=true")
    if phase.get("method", {}).get("intervention") != "per_frame_loss_weighting":
        raise TrainingConfigError("the frozen phase-aware method is per_frame_loss_weighting")


def assert_single_radeon() -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    if not torch.cuda.is_available() or count != 1:
        raise RuntimeError(f"formal training requires exactly one ROCm-visible GPU; available={count}")
    props = torch.cuda.get_device_properties(0)
    architecture = getattr(props, "gcnArchName", "")
    if not str(architecture).startswith("gfx1100"):
        raise RuntimeError(f"formal training requires gfx1100, got {architecture!r}")
    return {
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_count": count,
        "device_name": torch.cuda.get_device_name(0),
        "gcn_arch": architecture,
        "total_memory": int(props.total_memory),
    }


def build_command(config: dict[str, Any], *, python: Path, dataset_root: Path, output_dir: Path) -> list[str]:
    dataset = config["dataset"]
    policy = config["policy"]
    training = config["training"]
    reproducibility = config["reproducibility"]
    command = [
        str(python),
        "-m",
        "lerobot.scripts.lerobot_train",
        "--policy.type=act",
        f"--policy.device={policy['device']}",
        f"--dataset.repo_id={dataset['repo_id']}",
        f"--dataset.root={dataset_root}",
        f"--dataset.video_backend={dataset['video_backend']}",
        f"--output_dir={output_dir}",
        f"--job_name={config['experiment']}",
        f"--batch_size={training['batch_size']}",
        f"--steps={training['steps']}",
        f"--num_workers={training['num_workers']}",
        f"--log_freq={training['log_freq']}",
        "--save_checkpoint=true",
        f"--save_freq={training['save_freq']}",
        f"--seed={reproducibility['seed']}",
        "--wandb.enable=false",
    ]
    if config["method"]["phase_aware"]:
        target_path = dataset_root / config["method"]["targets_relative_path"]
        if not target_path.is_file():
            raise FileNotFoundError(target_path)
        command.extend(
            [
                "--use_act_awr=true",
                f"--act_awr_targets_path={target_path}",
                "--act_awr_run_id=radeon_oneloop_phase_v1",
                "--act_awr_weight_column=act_awr_weight",
                "--act_awr_missing_weight=0.0",
                "--act_awr_normalize_batch=true",
                "--act_awr_epsilon=1e-6",
            ]
        )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--paired-config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--command-json", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--skip-hardware-check", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paired = load_config(args.paired_config)
    if config["method"]["phase_aware"]:
        assert_fair_pair(paired, config)
    else:
        assert_fair_pair(config, paired)
    command = build_command(
        config,
        python=args.python.resolve(),
        dataset_root=args.dataset_root.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    hardware = None if args.skip_hardware_check else assert_single_radeon()
    record = {
        "schema_version": "radeon_oneloop.train_command.v1",
        "experiment": config["experiment"],
        "parent_checkpoint": None,
        "hardware": hardware,
        "cwd": os.getcwd(),
        "argv": command,
        "shell": shlex.join(command),
    }
    if args.command_json:
        args.command_json.parent.mkdir(parents=True, exist_ok=True)
        args.command_json.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(shlex.join(command), flush=True)
    if args.execute:
        raise SystemExit(subprocess.run(command, check=False).returncode)


if __name__ == "__main__":
    main()
