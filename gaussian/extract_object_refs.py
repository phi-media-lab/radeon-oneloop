#!/usr/bin/env python3
"""Extract traceable dual-camera object references from a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Iterable


CAMERAS = ("front_cam", "hand_cam")


def sample_timestamps(start: float, end: float, count: int) -> tuple[float, ...]:
    if not 0.0 <= start < end:
        raise ValueError("timestamps must satisfy 0 <= start < end")
    if count < 1:
        raise ValueError("sample count must be positive")
    # Avoid reset/transient frames at both ends while covering the full motion.
    return tuple(start + (end - start) * (index + 1) / (count + 1) for index in range(count))


def parse_episode_indices(value: str) -> tuple[int, ...]:
    indices = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not indices or min(indices) < 0 or len(indices) != len(set(indices)):
        raise ValueError("episode indices must be a non-empty unique CSV of non-negative integers")
    return indices


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"source video escaped dataset root: {path}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_episode_indices, required=True)
    parser.add_argument("--samples-per-episode", type=int, default=5)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()
    if not 1 <= args.samples_per_episode <= 60:
        raise ValueError("samples-per-episode must be between 1 and 60")

    import pandas as pd

    dataset_root = args.dataset_root.resolve()
    output = args.output.resolve()
    episodes_path = dataset_root / "meta/episodes/chunk-000/file-000.parquet"
    info_path = dataset_root / "meta/info.json"
    if not episodes_path.is_file() or not info_path.is_file():
        raise FileNotFoundError("dataset is missing LeRobot v3 episode metadata")
    episodes = pd.read_parquet(episodes_path).set_index("episode_index", drop=False)
    missing = sorted(set(args.episodes) - set(int(value) for value in episodes.index))
    if missing:
        raise ValueError(f"episodes are absent from metadata: {missing}")
    output.mkdir(parents=True, exist_ok=True)

    records = []
    for episode_index in args.episodes:
        row = episodes.loc[episode_index]
        for camera in CAMERAS:
            prefix = f"videos/observation.images.{camera}"
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            start = float(row[f"{prefix}/from_timestamp"])
            end = float(row[f"{prefix}/to_timestamp"])
            source = (
                dataset_root
                / "videos"
                / f"observation.images.{camera}"
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            for sample_index, timestamp in enumerate(
                sample_timestamps(start, end, args.samples_per_episode)
            ):
                destination = output / (
                    f"ep{episode_index:03d}_{camera}_s{sample_index:02d}_"
                    f"t{timestamp:010.3f}.jpg"
                )
                subprocess.run(
                    [
                        args.ffmpeg,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-ss",
                        f"{timestamp:.6f}",
                        "-i",
                        str(source),
                        "-frames:v",
                        "1",
                        "-q:v",
                        "2",
                        str(destination),
                    ],
                    check=True,
                )
                records.append(
                    {
                        "episode_index": episode_index,
                        "episode_success": str(row.get("episode_success", "unknown")),
                        "camera": camera,
                        "sample_index": sample_index,
                        "source_video": require_relative_to(source, dataset_root),
                        "source_timestamp_s": timestamp,
                        "image": destination.name,
                        "image_sha256": sha256_file(destination),
                    }
                )

    manifest = {
        "schema_version": "radeon_oneloop.private_object_refs.v1",
        "redistribution": False,
        "dataset_root_committed": False,
        "dataset_info_sha256": sha256_file(info_path),
        "episodes_metadata_sha256": sha256_file(episodes_path),
        "episodes": list(args.episodes),
        "cameras": list(CAMERAS),
        "samples_per_episode": args.samples_per_episode,
        "images": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), "images": len(records)}, indent=2))


if __name__ == "__main__":
    main()
