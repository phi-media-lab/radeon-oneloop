#!/usr/bin/env python3
"""Export an immutable Real2Sim capture from a LeRobot v3 HIL dataset.

The raw dataset stays outside the repository.  This command exports only the
selected RGB frames, synchronized joint trajectories, relative source names,
and content hashes needed by the COLMAP and Genesis workstreams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np


CAMERAS = ("front_cam", "hand_cam")
DATA_COLUMNS = (
    "action",
    "observation.state",
    "timestamp",
    "frame_index",
    "episode_index",
    "complementary_info.is_intervention",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_episode_spec(value: str) -> tuple[int, ...]:
    """Parse comma-separated episode indexes and inclusive ranges."""
    selected: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start < 0 or end < start:
                raise ValueError(f"invalid episode range: {token!r}")
            selected.update(range(start, end + 1))
        else:
            index = int(token)
            if index < 0:
                raise ValueError("episode indexes must be non-negative")
            selected.add(index)
    if not selected:
        raise ValueError("at least one episode must be selected")
    return tuple(sorted(selected))


def sampled_row_indexes(
    timestamps: Sequence[float], sample_hz: float
) -> tuple[int, ...]:
    """Select nearest data rows on a stable, episode-local time grid."""
    values = np.asarray(timestamps, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("timestamps must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any(np.diff(values) < 0.0):
        raise ValueError("timestamps must be finite and monotonic")
    if not math.isfinite(sample_hz) or sample_hz <= 0.0:
        raise ValueError("sample_hz must be finite and positive")
    targets = np.arange(values[0], values[-1] + 0.5 / sample_hz, 1.0 / sample_hz)
    positions = np.searchsorted(values, targets, side="left")
    positions = np.clip(positions, 0, values.size - 1)
    left = np.maximum(positions - 1, 0)
    choose_left = np.abs(values[left] - targets) <= np.abs(values[positions] - targets)
    nearest = np.where(choose_left, left, positions)
    return tuple(int(index) for index in np.unique(nearest))


def evenly_spaced_indexes(length: int, count: int) -> tuple[int, ...]:
    if length < 1:
        raise ValueError("length must be positive")
    if count < 0:
        raise ValueError("count must be non-negative")
    if count == 0:
        return ()
    count = min(count, length)
    return tuple(int(value) for value in np.unique(np.linspace(0, length - 1, count)))


def _require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - exercised on remote data host
        raise RuntimeError("pyarrow is required; install the project data extra") from exc
    return parquet


def load_episode_records(dataset: Path) -> dict[int, dict[str, Any]]:
    parquet = _require_pyarrow()
    paths = sorted((dataset / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not paths:
        raise FileNotFoundError("LeRobot v3 episode metadata was not found")
    records: dict[int, dict[str, Any]] = {}
    for path in paths:
        for record in parquet.read_table(path).to_pylist():
            index = int(record["episode_index"])
            if index in records:
                raise ValueError(f"duplicate episode_index {index}")
            records[index] = record
    return records


def relative_data_path(record: dict[str, Any]) -> Path:
    return Path("data") / f"chunk-{int(record['data/chunk_index']):03d}" / (
        f"file-{int(record['data/file_index']):03d}.parquet"
    )


def relative_video_path(camera: str, record: dict[str, Any]) -> Path:
    if camera not in CAMERAS:
        raise ValueError(f"unsupported camera: {camera}")
    prefix = f"videos/observation.images.{camera}"
    return Path(prefix) / f"chunk-{int(record[prefix + '/chunk_index']):03d}" / (
        f"file-{int(record[prefix + '/file_index']):03d}.mp4"
    )


def load_episode_rows(
    dataset: Path, record: dict[str, Any]
) -> dict[str, np.ndarray]:
    parquet = _require_pyarrow()
    path = dataset / relative_data_path(record)
    table = parquet.read_table(path, columns=list(DATA_COLUMNS))
    payload = table.to_pydict()
    episode_index = int(record["episode_index"])
    matching = np.flatnonzero(
        np.asarray(payload["episode_index"], dtype=np.int64) == episode_index
    )
    if matching.size != int(record["length"]):
        raise ValueError(
            f"episode {episode_index} metadata length {record['length']} does not "
            f"match {matching.size} data rows"
        )
    rows = {
        "action": np.asarray(payload["action"], dtype=np.float32)[matching],
        "observation_state": np.asarray(
            payload["observation.state"], dtype=np.float32
        )[matching],
        "timestamp": np.asarray(payload["timestamp"], dtype=np.float64)[matching],
        "frame_index": np.asarray(payload["frame_index"], dtype=np.int64)[matching],
        "intervention": np.asarray(
            payload["complementary_info.is_intervention"], dtype=np.float32
        )[matching],
    }
    if rows["action"].shape != (matching.size, 12):
        raise ValueError("action rows must have shape (frames, 12)")
    if rows["observation_state"].shape != (matching.size, 12):
        raise ValueError("observation.state rows must have shape (frames, 12)")
    expected_frames = np.arange(matching.size, dtype=np.int64)
    if not np.array_equal(rows["frame_index"], expected_frames):
        raise ValueError("episode-local frame_index is not contiguous from zero")
    return rows


def _extract_selected_frames(
    *,
    ffmpeg: str,
    video: Path,
    start_s: float,
    row_indexes: Sequence[int],
    source_fps: float,
    destination: Path,
    filename_prefix: str,
) -> list[Path]:
    if not row_indexes:
        return []
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as temporary:
        temporary_path = Path(temporary)
        expression = "+".join(
            f"eq(n\\,{int(round(index))})" for index in row_indexes
        )
        output_pattern = temporary_path / "%06d.jpg"
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start_s:.9f}",
            "-i",
            str(video),
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(len(row_indexes)),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(output_pattern),
        ]
        # FFmpeg's interactive command reader otherwise consumes the parent
        # shell's stdin (notably a remote heredoc) after extraction finishes.
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        extracted = sorted(temporary_path.glob("*.jpg"))
        if len(extracted) != len(row_indexes):
            raise RuntimeError(
                f"ffmpeg extracted {len(extracted)} frames, expected {len(row_indexes)} "
                f"from {video.name} at {source_fps:g} fps"
            )
        outputs: list[Path] = []
        for source, row_index in zip(extracted, row_indexes, strict=True):
            target = destination / f"{filename_prefix}_f{row_index:06d}.jpg"
            source.replace(target)
            outputs.append(target)
    return outputs


def _write_hash_manifest(root: Path) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        entries.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "hashes.sha256").write_text("\n".join(entries) + "\n", encoding="utf-8")


def export_capture(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing LeRobot info.json under {dataset}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "images").mkdir()
    (output / "front_anchors").mkdir()
    (output / "trajectories").mkdir()

    info = json.loads(info_path.read_text(encoding="utf-8"))
    source_fps = float(info["fps"])
    if args.sample_hz > source_fps:
        raise ValueError("sample-hz must not exceed the dataset frame rate")
    records = load_episode_records(dataset)
    missing = sorted(set(args.episodes) - set(records))
    if missing:
        raise ValueError(f"episode indexes do not exist: {missing}")

    source_hashes: dict[str, str] = {
        "meta/info.json": sha256_file(info_path),
    }
    frame_records: list[dict[str, Any]] = []
    episode_reports = []
    for episode_index in args.episodes:
        record = records[episode_index]
        if args.require_success and str(record.get("episode_success")) != "success":
            raise ValueError(f"episode {episode_index} is not marked success")
        rows = load_episode_rows(dataset, record)
        trajectory_path = output / "trajectories" / f"episode_{episode_index:03d}.npz"
        np.savez_compressed(trajectory_path, **rows)

        data_rel = relative_data_path(record)
        if data_rel.as_posix() not in source_hashes:
            source_hashes[data_rel.as_posix()] = sha256_file(dataset / data_rel)
        sampled = sampled_row_indexes(rows["timestamp"], args.sample_hz)
        camera = args.camera
        video_rel = relative_video_path(camera, record)
        video_path = dataset / video_rel
        if video_rel.as_posix() not in source_hashes:
            source_hashes[video_rel.as_posix()] = sha256_file(video_path)
        prefix = f"e{episode_index:03d}_{camera}"
        images = _extract_selected_frames(
            ffmpeg=args.ffmpeg,
            video=video_path,
            start_s=float(record[f"videos/observation.images.{camera}/from_timestamp"]),
            row_indexes=sampled,
            source_fps=source_fps,
            destination=output / "images",
            filename_prefix=prefix,
        )
        for image, row_index in zip(images, sampled, strict=True):
            timestamp = float(rows["timestamp"][row_index])
            frame_records.append(
                {
                    "image": image.relative_to(output).as_posix(),
                    "camera": camera,
                    "episode_index": episode_index,
                    "frame_index": int(rows["frame_index"][row_index]),
                    "timestamp_s": timestamp,
                    "video_timestamp_s": float(
                        record[f"videos/observation.images.{camera}/from_timestamp"]
                    )
                    + timestamp,
                    "trajectory": trajectory_path.relative_to(output).as_posix(),
                    "sha256": sha256_file(image),
                }
            )

        anchors = evenly_spaced_indexes(len(rows["timestamp"]), args.front_anchors)
        front_rel = relative_video_path("front_cam", record)
        front_path = dataset / front_rel
        if front_rel.as_posix() not in source_hashes:
            source_hashes[front_rel.as_posix()] = sha256_file(front_path)
        anchor_outputs = _extract_selected_frames(
            ffmpeg=args.ffmpeg,
            video=front_path,
            start_s=float(
                record["videos/observation.images.front_cam/from_timestamp"]
            ),
            row_indexes=anchors,
            source_fps=source_fps,
            destination=output / "front_anchors",
            filename_prefix=f"e{episode_index:03d}_front_cam",
        )
        episode_reports.append(
            {
                "episode_index": episode_index,
                "episode_success": record.get("episode_success"),
                "frames": int(len(rows["timestamp"])),
                "duration_s": float(rows["timestamp"][-1] - rows["timestamp"][0]),
                "sampled_images": len(images),
                "front_anchors": len(anchor_outputs),
                "intervention_frames": int(np.count_nonzero(rows["intervention"])),
                "trajectory": trajectory_path.relative_to(output).as_posix(),
            }
        )

    with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for record in frame_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "schema_version": "radeon_oneloop.hil_real2sim_capture.v1",
        "formal": False,
        "raw_data_redistributed": False,
        "dataset": {
            "robot_type": info.get("robot_type"),
            "fps": source_fps,
            "total_episodes": info.get("total_episodes"),
            "total_frames": info.get("total_frames"),
            "source_hashes": dict(sorted(source_hashes.items())),
        },
        "capture": {
            "camera": args.camera,
            "sample_hz": args.sample_hz,
            "image_count": len(frame_records),
            "front_anchor_count": sum(item["front_anchors"] for item in episode_reports),
            "episodes": episode_reports,
        },
        "coordinate_alignment": {
            "status": "pending_hand_eye_similarity",
            "metric_source": "SO-101 kinematics synchronized by frame timestamp",
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_hash_manifest(output)
    (output / "DONE").touch()
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_episode_spec, required=True)
    parser.add_argument("--camera", choices=CAMERAS, default="hand_cam")
    parser.add_argument("--sample-hz", type=float, default=2.0)
    parser.add_argument("--front-anchors", type=int, default=3)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = export_capture(args)
    except Exception:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED").touch()
        raise
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
