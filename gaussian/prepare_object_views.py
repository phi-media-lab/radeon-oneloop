#!/usr/bin/env python3
"""Freeze and normalize provenance-safe object views for the Real2Sim asset.

Source pixels and generated outputs stay outside the public repository.  The
committed YAML contains relative paths and expected public hashes only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable


CONFIG_SCHEMA_VERSION = "radeon_oneloop.object_asset_config.v1"
MANIFEST_SCHEMA_VERSION = "radeon_oneloop.object_asset_manifest.v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
VALID_TIERS = {"A", "B", "C", "G"}
VALID_PROVENANCE = {"observed", "generated"}
VALID_ROLES = {
    "pose",
    "photometric",
    "evaluation",
    "identity",
    "shape_prior",
    "generation_input",
    "domain_qa",
    "deformation_qa",
    "excluded",
}
MASK_REVIEW_STATUSES = {
    "pending_visual_review",
    "reviewed_pass",
    "reviewed_fail",
}


class ConfigError(ValueError):
    """Raised when provenance or split invariants are violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def require_relative_path(value: str, *, field: str) -> str:
    if not value or "\\" in value:
        raise ConfigError(f"{field} must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ConfigError(f"{field} must not be absolute or contain dot segments: {value}")
    return value


def relative_to_repo(path: Path, repo_root: Path, *, field: str) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise ConfigError(f"{field} must live in the public repository: {path}") from exc


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by the runtime environment
        raise RuntimeError("PyYAML is required to prepare object views") from exc
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfigError("asset config must be a YAML object")
    return value


def _require_keys(value: dict[str, Any], keys: Iterable[str], *, context: str) -> None:
    missing = sorted(set(keys) - set(value))
    if missing:
        raise ConfigError(f"{context} is missing keys: {missing}")


def validate_config(config: dict[str, Any]) -> None:
    _require_keys(
        config,
        {
            "schema_version",
            "asset_name",
            "formal",
            "redistribution",
            "coordinate_convention",
            "metric_anchor",
            "normalization",
            "views",
        },
        context="config",
    )
    if config["schema_version"] != CONFIG_SCHEMA_VERSION:
        raise ConfigError(f"unsupported config schema: {config['schema_version']}")
    if config["formal"] is not False or config["redistribution"] is not False:
        raise ConfigError("object preparation is a nonformal, non-redistributable stage")
    if not isinstance(config["asset_name"], str) or not config["asset_name"]:
        raise ConfigError("asset_name must be a non-empty string")

    coordinate = config["coordinate_convention"]
    if coordinate != {
        "front_axis": "+Y",
        "up_axis": "+Z",
        "viewer_left_axis": "+X",
        "unit": "m",
        "origin": "plush_body_center",
    }:
        raise ConfigError("coordinate convention must match the canonical object contract")

    metric = config["metric_anchor"]
    _require_keys(
        metric,
        {"kind", "dimension", "value_m", "uncertainty_m", "status"},
        context="metric_anchor",
    )
    if metric["kind"] != "product_specification" or metric["dimension"] != "overall_height":
        raise ConfigError("metric anchor must be the overall-height product specification")
    if metric["status"] != "user_confirmed_metric_anchor":
        raise ConfigError("metric anchor status must preserve user confirmation")
    if not isinstance(metric["value_m"], (int, float)) or metric["value_m"] <= 0:
        raise ConfigError("metric value_m must be positive")
    if not isinstance(metric["uncertainty_m"], (int, float)) or metric["uncertainty_m"] < 0:
        raise ConfigError("metric uncertainty_m must be non-negative")

    normalization = config["normalization"]
    _require_keys(
        normalization,
        {
            "output_size",
            "foreground_padding_fraction",
            "grabcut_iterations",
            "soft_alpha_sigma_px",
            "min_foreground_fraction",
            "max_foreground_fraction",
        },
        context="normalization",
    )
    if not 256 <= int(normalization["output_size"]) <= 4096:
        raise ConfigError("normalization output_size must be in [256, 4096]")
    if not 0.0 <= float(normalization["foreground_padding_fraction"]) <= 0.5:
        raise ConfigError("foreground padding must be in [0, 0.5]")
    if not 1 <= int(normalization["grabcut_iterations"]) <= 30:
        raise ConfigError("grabcut_iterations must be in [1, 30]")
    min_fraction = float(normalization["min_foreground_fraction"])
    max_fraction = float(normalization["max_foreground_fraction"])
    if not 0.0 < min_fraction < max_fraction < 1.0:
        raise ConfigError("foreground fraction limits must satisfy 0 < min < max < 1")

    views = config["views"]
    if not isinstance(views, list) or len(views) < 4:
        raise ConfigError("config must contain at least four views")
    ids: set[str] = set()
    source_paths: set[str] = set()
    photometric_instances: set[str] = set()
    pose_count = 0
    evaluation_count = 0
    generation_input_count = 0
    for index, view in enumerate(views):
        context = f"views[{index}]"
        if not isinstance(view, dict):
            raise ConfigError(f"{context} must be an object")
        _require_keys(
            view,
            {
                "id",
                "source_id",
                "instance_id",
                "source_relpath",
                "source_sha256",
                "tier",
                "provenance",
                "view_label",
                "roles",
                "canonical",
                "prepare",
            },
            context=context,
        )
        view_id = view["id"]
        if not isinstance(view_id, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", view_id):
            raise ConfigError(f"{context}.id is invalid: {view_id!r}")
        if view_id in ids:
            raise ConfigError(f"duplicate view id: {view_id}")
        ids.add(view_id)
        source_relpath = require_relative_path(view["source_relpath"], field=f"{context}.source_relpath")
        if source_relpath in source_paths:
            raise ConfigError(f"source image is listed more than once: {source_relpath}")
        source_paths.add(source_relpath)
        if not SHA256_PATTERN.fullmatch(str(view["source_sha256"])):
            raise ConfigError(f"{context}.source_sha256 is invalid")
        if view["tier"] not in VALID_TIERS:
            raise ConfigError(f"{context}.tier is invalid")
        if view["provenance"] not in VALID_PROVENANCE:
            raise ConfigError(f"{context}.provenance is invalid")
        roles = view["roles"]
        if not isinstance(roles, list) or not roles or len(roles) != len(set(roles)):
            raise ConfigError(f"{context}.roles must be a non-empty unique list")
        invalid_roles = sorted(set(roles) - VALID_ROLES)
        if invalid_roles:
            raise ConfigError(f"{context}.roles contains invalid values: {invalid_roles}")
        if not isinstance(view["canonical"], bool) or not isinstance(view["prepare"], bool):
            raise ConfigError(f"{context}.canonical and prepare must be booleans")
        if not view["canonical"] and not view.get("exclusion_reason"):
            raise ConfigError(f"{context} requires exclusion_reason when canonical is false")
        if view["provenance"] == "generated":
            if view["tier"] != "G":
                raise ConfigError(f"{context}: generated views must use tier G")
            if {"pose", "photometric", "evaluation"} & set(roles):
                raise ConfigError(f"{context}: generated views cannot supervise pose, photometric loss, or evaluation")
        elif view["tier"] == "G":
            raise ConfigError(f"{context}: tier G requires generated provenance")
        if "evaluation" in roles:
            evaluation_count += 1
            if view["provenance"] != "observed" or view["tier"] != "A":
                raise ConfigError(f"{context}: evaluation views must be observed tier A")
        if "photometric" in roles:
            photometric_instances.add(str(view["instance_id"]))
            if not view["prepare"]:
                raise ConfigError(f"{context}: photometric views must be prepared")
        if "pose" in roles:
            pose_count += 1
            if view["provenance"] != "observed" or not view["prepare"]:
                raise ConfigError(f"{context}: pose views must be prepared observed views")
        if "generation_input" in roles:
            generation_input_count += 1
            if view["provenance"] != "observed" or not view["prepare"]:
                raise ConfigError(f"{context}: generation inputs must be prepared observed views")
        if view["prepare"]:
            rect = view.get("grabcut_rect_fraction")
            if not isinstance(rect, list) or len(rect) != 4:
                raise ConfigError(f"{context}: prepared views require grabcut_rect_fraction")
            x0, y0, x1, y1 = (float(item) for item in rect)
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ConfigError(f"{context}: invalid grabcut_rect_fraction")
        if "manufacturing_error" in view_id:
            if view["canonical"] or roles != ["excluded"]:
                raise ConfigError("manufacturing-error evidence must be noncanonical and excluded-only")

    if len(photometric_instances) != 1:
        raise ConfigError(
            "photometric views must belong to exactly one coherent physical instance; "
            f"got {sorted(photometric_instances)}"
        )
    if pose_count < 4 or evaluation_count < 4 or generation_input_count < 4:
        raise ConfigError(
            "at least four observed prepared views are required for pose, evaluation, and generation input"
        )


def public_source_hashes(public_manifest: dict[str, Any]) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = {}
    for source in public_manifest.get("sources", []):
        source_id = source.get("id")
        if not isinstance(source_id, str):
            continue
        hashes[source_id] = {
            str(view["sha256"])
            for view in source.get("views", [])
            if isinstance(view, dict) and SHA256_PATTERN.fullmatch(str(view.get("sha256", "")))
        }
    return hashes


def validate_sources(
    config: dict[str, Any], source_root: Path, public_manifest: dict[str, Any]
) -> dict[str, Path]:
    source_root = source_root.resolve()
    public_hashes = public_source_hashes(public_manifest)
    resolved: dict[str, Path] = {}
    for view in config["views"]:
        source_id = str(view["source_id"])
        expected = str(view["source_sha256"])
        if source_id not in public_hashes or expected not in public_hashes[source_id]:
            raise ConfigError(
                f"{view['id']}: hash is not declared under source_id={source_id} in public manifest"
            )
        path = (source_root / str(view["source_relpath"])).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ConfigError(f"{view['id']}: source path escaped private source root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise ConfigError(f"{view['id']}: source hash mismatch: expected {expected}, got {actual}")
        resolved[str(view["id"])] = path
    return resolved


def _load_cv_modules() -> tuple[Any, Any]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "OpenCV and NumPy are required for mask generation; install the real2sim extras"
        ) from exc
    return cv2, np


def _largest_component(mask: Any, cv2: Any, np: Any) -> Any:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        raise RuntimeError("foreground mask is empty")
    foreground_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == foreground_label, 255, 0).astype(np.uint8)


def grabcut_foreground(image: Any, rect_fraction: list[float], iterations: int) -> Any:
    cv2, np = _load_cv_modules()
    height, width = image.shape[:2]
    x0 = max(1, min(width - 2, round(float(rect_fraction[0]) * width)))
    y0 = max(1, min(height - 2, round(float(rect_fraction[1]) * height)))
    x1 = max(x0 + 1, min(width - 1, round(float(rect_fraction[2]) * width)))
    y1 = max(y0 + 1, min(height - 1, round(float(rect_fraction[3]) * height)))
    mask = np.zeros((height, width), dtype=np.uint8)
    background = np.zeros((1, 65), dtype=np.float64)
    foreground = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        image,
        mask,
        (x0, y0, x1 - x0, y1 - y0),
        background,
        foreground,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    binary = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)
    kernel = np.ones((5, 5), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return _largest_component(binary, cv2, np)


def _foreground_bbox(mask: Any, np: Any) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("foreground mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def normalize_prepared_view(
    source: Path,
    view: dict[str, Any],
    normalization: dict[str, Any],
    staging: Path,
    external_mask: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], tuple[Any, Any, Any]]:
    cv2, np = _load_cv_modules()
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not decode {source}")
    height, width = image.shape[:2]
    if external_mask is None:
        mask = grabcut_foreground(
            image,
            [float(item) for item in view["grabcut_rect_fraction"]],
            int(normalization["grabcut_iterations"]),
        )
        mask_source = {"method": "grabcut_rect"}
    else:
        mask = cv2.imread(str(external_mask["path"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"OpenCV could not decode external mask for {view['id']}")
        if mask.shape != (height, width):
            raise RuntimeError(
                f"{view['id']}: external mask shape {mask.shape} does not match "
                f"source {(height, width)}"
            )
        mask = np.where(mask >= 128, 255, 0).astype(np.uint8)
        mask_source = {
            "method": "sam3_image_text_prompt",
            "candidate_manifest_sha256": external_mask["manifest_sha256"],
            "candidate_view_id": view["id"],
            "candidate_mask_sha256": external_mask["mask_sha256"],
        }
    fx0, fy0, fx1, fy1 = _foreground_bbox(mask, np)
    foreground_size = max(fx1 - fx0, fy1 - fy0)
    padding = int(round(foreground_size * float(normalization["foreground_padding_fraction"])))
    crop_x0 = max(0, fx0 - padding)
    crop_y0 = max(0, fy0 - padding)
    crop_x1 = min(width, fx1 + padding)
    crop_y1 = min(height, fy1 + padding)
    cropped_image = image[crop_y0:crop_y1, crop_x0:crop_x1]
    cropped_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1]
    crop_height, crop_width = cropped_image.shape[:2]
    side = max(crop_width, crop_height)
    pad_left = (side - crop_width) // 2
    pad_right = side - crop_width - pad_left
    pad_top = (side - crop_height) // 2
    pad_bottom = side - crop_height - pad_top
    squared_image = cv2.copyMakeBorder(
        cropped_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(255, 255, 255),
    )
    squared_mask = cv2.copyMakeBorder(
        cropped_mask,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    output_size = int(normalization["output_size"])
    normalized_image = cv2.resize(
        squared_image, (output_size, output_size), interpolation=cv2.INTER_AREA
    )
    normalized_mask = cv2.resize(
        squared_mask, (output_size, output_size), interpolation=cv2.INTER_NEAREST
    )
    normalized_mask = np.where(normalized_mask >= 128, 255, 0).astype(np.uint8)
    sigma = float(normalization["soft_alpha_sigma_px"])
    soft_alpha = cv2.GaussianBlur(normalized_mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha_f = soft_alpha.astype(np.float32)[:, :, None] / 255.0
    neutral = np.full_like(normalized_image, 127)
    neutral_image = np.clip(
        normalized_image.astype(np.float32) * alpha_f
        + neutral.astype(np.float32) * (1.0 - alpha_f),
        0,
        255,
    ).astype(np.uint8)

    output_paths = {
        "image": Path("01_normalized/rgb") / f"{view['id']}.png",
        "neutral_image": Path("01_normalized/neutral_rgb") / f"{view['id']}.png",
        "hard_mask": Path("01_normalized/masks") / f"{view['id']}.png",
        "soft_alpha": Path("01_normalized/alpha") / f"{view['id']}.png",
    }
    arrays = {
        "image": normalized_image,
        "neutral_image": neutral_image,
        "hard_mask": normalized_mask,
        "soft_alpha": soft_alpha,
    }
    for key, relpath in output_paths.items():
        destination = staging / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), arrays[key]):
            raise RuntimeError(f"failed to write {destination}")

    foreground_fraction = float(np.mean(normalized_mask > 0))
    min_fraction = float(normalization["min_foreground_fraction"])
    max_fraction = float(normalization["max_foreground_fraction"])
    if not min_fraction <= foreground_fraction <= max_fraction:
        raise RuntimeError(
            f"{view['id']}: foreground fraction {foreground_fraction:.4f} is outside "
            f"[{min_fraction}, {max_fraction}]"
        )
    touches = bool(
        np.any(normalized_mask[0, :])
        or np.any(normalized_mask[-1, :])
        or np.any(normalized_mask[:, 0])
        or np.any(normalized_mask[:, -1])
    )
    if touches:
        raise RuntimeError(f"{view['id']}: foreground touches normalized output border")
    out_bbox = _foreground_bbox(normalized_mask, np)
    scale = output_size / side
    tx = (-crop_x0 + pad_left) * scale
    ty = (-crop_y0 + pad_top) * scale
    record = {
        key: {"relpath": relpath.as_posix(), "sha256": sha256_file(staging / relpath)}
        for key, relpath in output_paths.items()
    }
    record["mask_source"] = mask_source
    record["normalization"] = {
        "source_width": width,
        "source_height": height,
        "crop_xyxy": [crop_x0, crop_y0, crop_x1, crop_y1],
        "square_padding_ltrb": [pad_left, pad_top, pad_right, pad_bottom],
        "output_width": output_size,
        "output_height": output_size,
        "source_to_output_affine_2x3": [
            [scale, 0.0, tx],
            [0.0, scale, ty],
        ],
    }
    record["mask_qa"] = {
        "foreground_fraction": foreground_fraction,
        "touches_output_border": touches,
        "foreground_bbox_xyxy": list(out_bbox),
    }

    overlay = normalized_image.copy()
    contours, _ = cv2.findContours(
        normalized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), max(2, output_size // 256))
    return record, (normalized_image, overlay, neutral_image)


def write_qa_montage(staging: Path, rows: list[tuple[str, tuple[Any, Any, Any]]]) -> None:
    if not rows:
        return
    cv2, np = _load_cv_modules()
    tile = 256
    rendered_rows = []
    for view_id, triplet in rows:
        tiles = [cv2.resize(image, (tile, tile), interpolation=cv2.INTER_AREA) for image in triplet]
        row = np.concatenate(tiles, axis=1)
        cv2.rectangle(row, (0, 0), (tile * 3, 28), (0, 0, 0), -1)
        cv2.putText(
            row,
            f"{view_id}: rgb | mask overlay | neutral generator input",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        rendered_rows.append(row)
    montage = np.concatenate(rendered_rows, axis=0)
    destination = staging / "01_normalized/qa/masks_montage.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), montage):
        raise RuntimeError(f"failed to write {destination}")


def _base_view_record(view: dict[str, Any], mask_review_status: str) -> dict[str, Any]:
    record = {
        key: view[key]
        for key in (
            "id",
            "source_id",
            "instance_id",
            "source_relpath",
            "source_sha256",
            "tier",
            "provenance",
            "view_label",
            "roles",
            "canonical",
        )
    }
    if "exclusion_reason" in view:
        record["exclusion_reason"] = view["exclusion_reason"]
    if "nominal_camera_orbit_deg" in view:
        record["nominal_camera_orbit_deg"] = view["nominal_camera_orbit_deg"]
    record["prepared"] = bool(view["prepare"])
    record["mask_status"] = mask_review_status if view["prepare"] else "not_applicable"
    return record


def build_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    public_manifest_path: Path,
    repo_root: Path,
    source_paths: dict[str, Path],
    staging: Path,
    mask_review_status: str,
    external_masks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    qa_rows: list[tuple[str, tuple[Any, Any, Any]]] = []
    for view in config["views"]:
        record = _base_view_record(view, mask_review_status)
        if view["prepare"]:
            prepared, qa = normalize_prepared_view(
                source_paths[str(view["id"])],
                view,
                config["normalization"],
                staging,
                external_masks.get(str(view["id"])),
            )
            record.update(prepared)
            qa_rows.append((str(view["id"]), qa))
        records.append(record)
    write_qa_montage(staging, qa_rows)

    tier_counts = Counter(str(view["tier"]) for view in config["views"])
    photometric_instances = sorted(
        {
            str(view["instance_id"])
            for view in config["views"]
            if "photometric" in view["roles"]
        }
    )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "asset_name": config["asset_name"],
        "created_utc": utc_now(),
        "formal": False,
        "redistribution": False,
        "source_root_committed": False,
        "coordinate_convention": config["coordinate_convention"],
        "metric_anchor": config["metric_anchor"],
        "inputs": {
            "config_relpath": relative_to_repo(config_path, repo_root, field="config"),
            "config_sha256": sha256_file(config_path),
            "public_manifest_relpath": relative_to_repo(
                public_manifest_path, repo_root, field="public manifest"
            ),
            "public_manifest_sha256": sha256_file(public_manifest_path),
        },
        "views": records,
        "summary": {
            "view_count": len(records),
            "prepared_count": sum(bool(view["prepare"]) for view in config["views"]),
            "observed_count": sum(view["provenance"] == "observed" for view in config["views"]),
            "generated_count": sum(view["provenance"] == "generated" for view in config["views"]),
            "tier_counts": {tier: tier_counts.get(tier, 0) for tier in ("A", "B", "C", "G")},
            "photometric_instance_ids": photometric_instances,
            "anchor_pose_view_count": sum("pose" in view["roles"] for view in config["views"]),
            "evaluation_view_count": sum(
                "evaluation" in view["roles"] for view in config["views"]
            ),
            "mask_review_status": mask_review_status,
        },
    }


def write_manifest_artifacts(staging: Path, manifest: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("jsonschema is required to validate the object manifest") from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(manifest)
    manifest_path = staging / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    source_manifest = staging / "00_sources/source_manifest.jsonl"
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.write_text(
        "".join(json.dumps(view, sort_keys=True) + "\n" for view in manifest["views"]),
        encoding="utf-8",
    )
    hashes = []
    for path in sorted(item for item in staging.rglob("*") if item.is_file()):
        if path.name in {"hashes.sha256", "DONE", "FAILED"}:
            continue
        hashes.append(f"{sha256_file(path)}  {path.relative_to(staging).as_posix()}\n")
    (staging / "hashes.sha256").write_text("".join(hashes), encoding="utf-8")
    (staging / "DONE").write_text(
        json.dumps(
            {
                "schema_version": "radeon_oneloop.object_asset_stage_done.v1",
                "stage": "M1_view_preparation",
                "manifest_sha256": sha256_file(manifest_path),
                "completed_utc": utc_now(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_external_masks(
    manifest_path: Path | None, config: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    if manifest_path is None:
        return {}
    manifest_path = manifest_path.resolve()
    value = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != "radeon_oneloop.sam3_object_masks.v1":
        raise ConfigError("external mask manifest must be a SAM3 object-mask manifest")
    if value.get("formal") is not False:
        raise ConfigError("external SAM3 masks must remain nonformal")
    by_id = {str(item.get("view_id")): item for item in value.get("views", [])}
    manifest_sha256 = sha256_file(manifest_path)
    result: dict[str, dict[str, Any]] = {}
    for view in config["views"]:
        if not view["prepare"]:
            continue
        view_id = str(view["id"])
        item = by_id.get(view_id)
        if item is None:
            raise ConfigError(f"external mask manifest is missing prepared view {view_id}")
        if item.get("source_sha256") != view["source_sha256"]:
            raise ConfigError(f"external mask source hash mismatch for {view_id}")
        output = item.get("outputs", {}).get("mask", {})
        relpath = require_relative_path(str(output.get("relpath", "")), field="mask relpath")
        mask_path = (manifest_path.parent / relpath).resolve()
        try:
            mask_path.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ConfigError(f"external mask escaped candidate root for {view_id}") from exc
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        actual = sha256_file(mask_path)
        if actual != output.get("sha256"):
            raise ConfigError(f"external mask hash mismatch for {view_id}")
        result[view_id] = {
            "path": mask_path,
            "manifest_sha256": manifest_sha256,
            "mask_sha256": actual,
        }
    return result


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config.resolve()
    public_manifest_path = args.public_manifest.resolve()
    schema_path = args.schema.resolve()
    repo_root = args.repo_root.resolve()
    source_root = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config = load_yaml(config_path)
    validate_config(config)
    public_manifest = json.loads(public_manifest_path.read_text(encoding="utf-8"))
    source_paths = validate_sources(config, source_root, public_manifest)
    external_masks = load_external_masks(args.external_mask_manifest, config)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    try:
        manifest = build_manifest(
            config=config,
            config_path=config_path,
            public_manifest_path=public_manifest_path,
            repo_root=repo_root,
            source_paths=source_paths,
            staging=staging,
            mask_review_status=args.mask_review_status,
            external_masks=external_masks,
        )
        write_manifest_artifacts(staging, manifest, schema_path)
        os.replace(staging, output)
        return manifest
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "stage": "M1_view_preparation",
                    "failed_utc": utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED.{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}")
        os.replace(staging, failed)
        raise


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=repo_root / "configs/graffiti_mickey_asset.yaml"
    )
    parser.add_argument(
        "--public-manifest",
        type=Path,
        default=repo_root / "data/graffiti_mickey_reference_sources.json",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        default=repo_root / "gaussian/object_asset_manifest.schema.json",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--external-mask-manifest",
        type=Path,
        help="validated SAM3 candidate manifest whose original-resolution masks replace GrabCut",
    )
    parser.add_argument(
        "--mask-review-status",
        choices=sorted(MASK_REVIEW_STATUSES),
        default="pending_visual_review",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = prepare(args)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "summary": manifest["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
