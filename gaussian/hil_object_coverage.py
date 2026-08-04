#!/usr/bin/env python3
"""Audit object-view coverage in private dual-camera HIL episodes.

The command deliberately does not attempt to turn dynamic handover footage
into calibrated object cameras.  It exports synchronized phase samples and
image-quality metadata so a reviewer can decide which *real* regions are
observed before any generated view is admitted to the completion branch.
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_frame_indexes(length: int, phase_samples: int) -> tuple[int, ...]:
    """Return deterministic interior samples including both episode ends."""
    if length < 1:
        raise ValueError("episode length must be positive")
    if phase_samples < 2:
        raise ValueError("phase-samples must be at least two")
    count = min(length, phase_samples)
    return tuple(
        int(value)
        for value in np.unique(np.rint(np.linspace(0, length - 1, count))).astype(int)
    )


def parse_episode_spec(value: str) -> tuple[int, ...]:
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
        raise ValueError("episode selection is empty")
    return tuple(sorted(selected))


def _require_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - remote data host dependency
        raise RuntimeError("pyarrow is required on the private data host") from exc
    return parquet


def load_episode_records(dataset: Path) -> dict[int, dict[str, Any]]:
    parquet = _require_parquet()
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


def successful_episode_indexes(records: dict[int, dict[str, Any]]) -> tuple[int, ...]:
    return tuple(
        index
        for index, record in sorted(records.items())
        if str(record.get("episode_success")) == "success"
    )


def relative_video_path(camera: str, record: dict[str, Any]) -> Path:
    if camera not in CAMERAS:
        raise ValueError(f"unsupported camera: {camera}")
    prefix = f"videos/observation.images.{camera}"
    return Path(prefix) / f"chunk-{int(record[prefix + '/chunk_index']):03d}" / (
        f"file-{int(record[prefix + '/file_index']):03d}.mp4"
    )


def _extract_frames(
    *,
    ffmpeg: str,
    video: Path,
    episode_start_s: float,
    frame_indexes: Sequence[int],
    destination: Path,
    prefix: str,
) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination) as temp:
        temp_path = Path(temp)
        expression = "+".join(f"eq(n\\,{index})" for index in frame_indexes)
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{episode_start_s:.9f}",
            "-i",
            str(video),
            "-vf",
            f"select={expression}",
            "-fps_mode",
            "vfr",
            "-frames:v",
            str(len(frame_indexes)),
            "-q:v",
            "2",
            "-start_number",
            "0",
            str(temp_path / "%04d.jpg"),
        ]
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)
        extracted = sorted(temp_path.glob("*.jpg"))
        if len(extracted) != len(frame_indexes):
            raise RuntimeError(
                f"extracted {len(extracted)} frames, expected {len(frame_indexes)} "
                f"from {video}"
            )
        outputs: list[Path] = []
        for phase, (source, frame_index) in enumerate(
            zip(extracted, frame_indexes, strict=True)
        ):
            target = destination / f"{prefix}_p{phase:02d}_f{frame_index:06d}.jpg"
            source.replace(target)
            outputs.append(target)
        return outputs


def _image_metrics(path: Path) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - remote data host dependency
        raise RuntimeError("opencv-python is required on the private data host") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not decode {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return {
        "image_size_wh": [int(image.shape[1]), int(image.shape[0])],
        "luma_mean": float(np.mean(gray) / 255.0),
        "luma_p01": float(np.percentile(gray, 1.0) / 255.0),
        "luma_p99": float(np.percentile(gray, 99.0) / 255.0),
        "saturation_mean": float(np.mean(hsv[..., 1]) / 255.0),
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


def _contact_sheet(
    *,
    image_paths: Sequence[Path],
    labels: Sequence[str],
    columns: int,
    output: Path,
    tile_width: int = 240,
) -> None:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required on the private data host") from exc
    if len(image_paths) != len(labels) or not image_paths:
        raise ValueError("contact sheet inputs must be non-empty and aligned")
    decoded = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    if any(image is None for image in decoded):
        raise ValueError("a contact-sheet image could not be decoded")
    first = decoded[0]
    assert first is not None
    tile_height = int(round(first.shape[0] * tile_width / first.shape[1])) + 28
    rows = math.ceil(len(decoded) / columns)
    sheet = np.zeros((rows * tile_height, columns * tile_width, 3), dtype=np.uint8)
    for index, (image, label) in enumerate(zip(decoded, labels, strict=True)):
        assert image is not None
        resized_h = tile_height - 28
        resized = cv2.resize(image, (tile_width, resized_h), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        x, y = column * tile_width, row * tile_height
        sheet[y : y + resized_h, x : x + tile_width] = resized
        cv2.putText(
            sheet,
            label,
            (x + 5, y + tile_height - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 91]):
        raise RuntimeError(f"could not write {output}")


def _write_hashes(root: Path) -> None:
    lines = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    (root / "hashes.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_coverage_audit(args: argparse.Namespace) -> dict[str, Any]:
    dataset = args.dataset.resolve()
    output = args.output.resolve()
    info_path = dataset / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"missing {info_path}")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("dataset fps must be finite and positive")
    records = load_episode_records(dataset)
    episodes = args.episodes or successful_episode_indexes(records)
    missing = sorted(set(episodes) - set(records))
    if missing:
        raise ValueError(f"episode indexes do not exist: {missing}")
    not_success = [
        index
        for index in episodes
        if str(records[index].get("episode_success")) != "success"
    ]
    if not_success:
        raise ValueError(f"selected episodes are not successful: {not_success}")

    frame_records: list[dict[str, Any]] = []
    source_video_hashes: dict[str, str] = {}
    episode_reports: list[dict[str, Any]] = []
    for episode_index in episodes:
        episode = records[episode_index]
        length = int(episode["length"])
        indexes = phase_frame_indexes(length, args.phase_samples)
        episode_images: list[Path] = []
        episode_labels: list[str] = []
        for camera in CAMERAS:
            rel_video = relative_video_path(camera, episode)
            video = dataset / rel_video
            if not video.is_file():
                raise FileNotFoundError(video)
            rel_text = rel_video.as_posix()
            if rel_text not in source_video_hashes:
                source_video_hashes[rel_text] = sha256_file(video)
            start_s = float(
                episode[f"videos/observation.images.{camera}/from_timestamp"]
            )
            extracted = _extract_frames(
                ffmpeg=args.ffmpeg,
                video=video,
                episode_start_s=start_s,
                frame_indexes=indexes,
                destination=output / "images" / camera,
                prefix=f"e{episode_index:03d}_{camera}",
            )
            for phase, (path, frame_index) in enumerate(
                zip(extracted, indexes, strict=True)
            ):
                record = {
                    "camera": camera,
                    "episode_index": episode_index,
                    "episode_success": "success",
                    "phase_index": phase,
                    "phase_fraction": float(frame_index / max(1, length - 1)),
                    "frame_index": frame_index,
                    "episode_timestamp_s": float(frame_index / fps),
                    "source_video_relpath": rel_text,
                    "source_video_timestamp_s": float(start_s + frame_index / fps),
                    "image_relpath": path.relative_to(output).as_posix(),
                    "image_sha256": sha256_file(path),
                    "quality": _image_metrics(path),
                    "object_review": {
                        "status": "pending_visual_review",
                        "view_label": None,
                        "visible_fraction": None,
                        "robot_occlusion": None,
                        "deformation_state": None,
                    },
                }
                frame_records.append(record)
                episode_images.append(path)
                episode_labels.append(f"{camera[:1]} p{phase:02d} f{frame_index}")
        _contact_sheet(
            image_paths=episode_images,
            labels=episode_labels,
            columns=len(indexes),
            output=output / "contact_sheets" / f"episode_{episode_index:03d}.jpg",
        )
        episode_reports.append(
            {
                "episode_index": episode_index,
                "length": length,
                "duration_s": float((length - 1) / fps),
                "phase_samples_per_camera": len(indexes),
            }
        )

    with (output / "frames.jsonl").open("w", encoding="utf-8") as stream:
        for record in frame_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    for camera in CAMERAS:
        selected = [
            (output / str(record["image_relpath"]), record)
            for record in frame_records
            if record["camera"] == camera
        ]
        _contact_sheet(
            image_paths=[item[0] for item in selected],
            labels=[
                f"e{item[1]['episode_index']:03d} p{item[1]['phase_index']:02d}"
                for item in selected
            ],
            columns=args.phase_samples,
            output=output / "contact_sheets" / f"all_{camera}.jpg",
            tile_width=180,
        )

    manifest = {
        "schema_version": "radeon_oneloop.hil_object_coverage.v1",
        "created_utc": args.created_utc,
        "formal": False,
        "eligible_for_metric_geometry": False,
        "eligible_for_heldout_real_metrics": False,
        "raw_data_redistributed": False,
        "dataset": {
            "info_sha256": sha256_file(info_path),
            "fps": fps,
            "total_episodes": int(info.get("total_episodes", len(records))),
            "total_frames": int(info.get("total_frames", 0)),
            "private_path_recorded": False,
            "source_video_hashes": dict(sorted(source_video_hashes.items())),
        },
        "capture": {
            "episodes": episode_reports,
            "successful_episode_count": len(episodes),
            "cameras": list(CAMERAS),
            "phase_samples_per_camera": args.phase_samples,
            "image_count": len(frame_records),
        },
        "review": {
            "status": "pending_visual_object_view_labeling",
            "required_labels": [
                "view_label",
                "visible_fraction",
                "robot_occlusion",
                "deformation_state",
            ],
            "generation_policy": (
                "no HIL frame becomes metric geometry or generated-fill supervision "
                "until an immutable visual review is attached"
            ),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_hashes(output)
    (output / "DONE").write_text(sha256_file(output / "hashes.sha256") + "\n")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=parse_episode_spec)
    parser.add_argument("--phase-samples", type=int, default=12)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg") or "ffmpeg")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        manifest = build_coverage_audit(args)
    except Exception as exc:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "FAILED").write_text(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}) + "\n",
            encoding="utf-8",
        )
        raise
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
