"""Build deterministic frame weights for the formal phase-aware ACT run."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WEIGHTS = {
    "bc_demonstration": 1.0,
    "success_policy": 1.0,
    "failed_policy_prefix": 0.05,
    "recovery": 4.0,
    "correction": 4.0,
    "failed_policy_suffix": 0.05,
    "failed_policy_no_correction": 0.0,
}


@dataclass(frozen=True)
class Interval:
    start: int
    end: int
    role: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid interval: {self}")
        if self.role not in DEFAULT_WEIGHTS:
            raise ValueError(f"unknown segment role: {self.role}")


def frame_role(frame_index: int, intervals: Iterable[Interval], *, fallback: str) -> str:
    """Return the first explicit segment covering a frame, otherwise fallback."""
    for interval in intervals:
        if interval.start <= frame_index <= interval.end:
            return interval.role
    return fallback


def normalize_positive_mean(weights: list[float]) -> list[float]:
    positive = [value for value in weights if value > 0]
    if not positive:
        raise ValueError("at least one frame must have positive weight")
    scale = sum(positive) / len(positive)
    return [value / scale if value > 0 else 0.0 for value in weights]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def _optional_interval(segment: dict[str, Any] | None, role: str) -> Interval | None:
    if not segment:
        return None
    start = segment.get("start_frame_index")
    end = segment.get("end_frame_index")
    if start is None or end is None:
        return None
    return Interval(int(start), int(end), role)


def _episode_plan(row: dict[str, Any]) -> tuple[str, list[Interval]]:
    source_kind = str(row.get("source_kind", "hil"))
    if source_kind == "bc_seed":
        return "bc_demonstration", []
    success = bool(row.get("handover_success"))
    if success:
        return "success_policy", []
    segments = row.get("segments") or {}
    intervals = []
    for name, role in (
        ("policy_prefix", "failed_policy_prefix"),
        ("recovery", "recovery"),
        ("correction", "correction"),
        ("human_correction", "correction"),
        ("policy_suffix", "failed_policy_suffix"),
    ):
        interval = _optional_interval(segments.get(name), role)
        if interval is not None and interval not in intervals:
            intervals.append(interval)
    fallback = "failed_policy_prefix" if intervals else "failed_policy_no_correction"
    return fallback, intervals


def build_targets(
    *,
    dataset_parquet: Path,
    episode_manifest: Path,
    output_parquet: Path,
    report_path: Path,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised on remote data environment
        raise RuntimeError("pandas and pyarrow are required; install radeon-oneloop[data]") from exc

    effective_weights = dict(DEFAULT_WEIGHTS)
    if weights:
        unknown = set(weights) - set(effective_weights)
        if unknown:
            raise ValueError(f"unknown weight roles: {sorted(unknown)}")
        effective_weights.update({key: float(value) for key, value in weights.items()})
    if any(value < 0 for value in effective_weights.values()):
        raise ValueError("weights must be non-negative")

    manifest_rows = _read_jsonl(episode_manifest)
    manifest = {int(row["episode_index"]): row for row in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("episode manifest contains duplicate episode_index values")
    data = pd.read_parquet(dataset_parquet, columns=["index", "episode_index", "frame_index"])
    observed = {int(value) for value in data["episode_index"].unique()}
    missing = sorted(observed - set(manifest))
    extra = sorted(set(manifest) - observed)
    if missing or extra:
        raise ValueError(f"manifest/dataset episode mismatch: missing={missing}, extra={extra}")

    roles: list[str] = []
    raw_weights: list[float] = []
    plans = {episode: _episode_plan(row) for episode, row in manifest.items()}
    for episode, frame in zip(data["episode_index"], data["frame_index"], strict=True):
        fallback, intervals = plans[int(episode)]
        role = frame_role(int(frame), intervals, fallback=fallback)
        roles.append(role)
        raw_weights.append(effective_weights[role])
    normalized = normalize_positive_mean(raw_weights)

    output = data.copy()
    output["segment_role"] = roles
    output["act_awr_weight_raw"] = raw_weights
    output["act_awr_weight"] = normalized
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(output_parquet, index=False)

    role_counts = Counter(roles)
    role_mass: dict[str, float] = defaultdict(float)
    for role, weight in zip(roles, normalized, strict=True):
        role_mass[role] += weight
    total_mass = sum(role_mass.values())
    report = {
        "schema_version": "radeon_oneloop.phase_targets.v1",
        "method": "phase_aware_per_frame_loss_weighting",
        "normalization": "positive_frame_mean_equals_one",
        "inputs": {
            "dataset_parquet": str(dataset_parquet),
            "dataset_parquet_sha256": file_sha256(dataset_parquet),
            "episode_manifest": str(episode_manifest),
            "episode_manifest_sha256": file_sha256(episode_manifest),
        },
        "output": {
            "targets": str(output_parquet),
            "targets_sha256": file_sha256(output_parquet),
        },
        "weights": effective_weights,
        "summary": {
            "episodes": len(observed),
            "frames": len(output),
            "positive_frames": sum(value > 0 for value in normalized),
            "zero_frames": sum(value == 0 for value in normalized),
            "mean_positive_weight": sum(value for value in normalized if value > 0)
            / max(sum(value > 0 for value in normalized), 1),
            "roles": {
                role: {
                    "frames": role_counts[role],
                    "gradient_mass": role_mass[role],
                    "gradient_mass_ratio": role_mass[role] / total_mass if total_mass else 0.0,
                }
                for role in sorted(role_counts)
            },
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-parquet", type=Path, required=True)
    parser.add_argument("--episode-manifest", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--weights-json", default=None)
    args = parser.parse_args()
    report = build_targets(
        dataset_parquet=args.dataset_parquet,
        episode_manifest=args.episode_manifest,
        output_parquet=args.output_parquet,
        report_path=args.report,
        weights=json.loads(args.weights_json) if args.weights_json else None,
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

