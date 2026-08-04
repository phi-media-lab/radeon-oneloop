#!/usr/bin/env python3
"""Build a real-authoritative COLMAP dataset from an accepted SEVA orbit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from gaussian.align_generated_mesh_four_views import VIEW_ORDER, _fit_projection
from gaussian.audit_seva_orbit import AUDIT_SCHEMA, ANCHORS
from gaussian.export_observed_initialization import SCHEMA as OBSERVED_INIT_SCHEMA, load_colmap_points
from gaussian.hybrid_pseudoview_colmap import _colmap_image_line
from gaussian.prepare_four_view_generation import sha256_file, validate_generation_input
from gaussian.provenance_quarantine import assert_not_quarantined
from gaussian.record_seva_four_view_run import SCHEMA_VERSION as SEVA_RUN_SCHEMA
from gaussian.record_seva_orbit_review import ACCEPTED, REVIEW_SCHEMA


SCHEMA = "radeon_oneloop.seva_pseudoview_colmap_dataset.v1"
DONE_SCHEMA = "radeon_oneloop.seva_pseudoview_colmap_dataset_done.v1"
IMAGE_SIZE_WH = (576, 576)
ANCHOR_INDEX_BY_LABEL = {label: index for label, index in ANCHORS}
SEVA_INPUT_INDEX_BY_LABEL = {"front": 0, "right": 1, "back": 2, "left": 3}


class SevaDatasetError(ValueError):
    """Raised when a SEVA pseudo-view dataset would weaken evidence boundaries."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SevaDatasetError(f"cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise SevaDatasetError(f"{label} must be a JSON object")
    return value


def _verify_hash_index(root: Path) -> None:
    path = root / "hashes.sha256"
    if not path.is_file():
        raise SevaDatasetError(f"missing hash index: {path}")
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        try:
            digest, relpath = line.split("  ", 1)
        except ValueError as exc:
            raise SevaDatasetError(f"malformed hash line {line_number}: {path}") from exc
        relative = Path(relpath)
        candidate = root / relative
        if relative.is_absolute() or ".." in relative.parts or not candidate.is_file():
            raise SevaDatasetError(f"unsafe or missing hash target: {relpath}")
        if sha256_file(candidate) != digest:
            raise SevaDatasetError(f"hash mismatch: {root.name}/{relpath}")


def select_generated_indices(count: int) -> list[int]:
    """Select evenly distributed pseudo-views while excluding four anchor targets."""

    excluded = set(ANCHOR_INDEX_BY_LABEL.values())
    candidates = [index for index in range(49) if index not in excluded]
    if count < 1 or count > len(candidates):
        raise SevaDatasetError(f"generated count must be in [1, {len(candidates)}]")
    positions = np.linspace(0, len(candidates) - 1, count).round().astype(np.int64)
    selected = [candidates[int(position)] for position in positions]
    if len(selected) != len(set(selected)):
        raise SevaDatasetError("generated pseudo-view selection contains duplicates")
    return selected


def opengl_c2w_to_metric_opencv_w2c(c2w: np.ndarray, radius_m: float) -> np.ndarray:
    """Convert SEVA OpenGL c2w to OpenCV w2c and resolve its normalized radius."""

    value = np.asarray(c2w, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise SevaDatasetError("SEVA camera must be one finite 4 x 4 matrix")
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise SevaDatasetError("metric camera radius must be positive")
    center_norm = float(np.linalg.norm(value[:3, 3]))
    if center_norm <= 0:
        raise SevaDatasetError("SEVA camera center has zero radius")
    metric_gl = value.copy()
    metric_gl[:3, 3] *= radius_m / center_norm
    gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
    opencv_c2w = metric_gl @ gl_to_cv
    return np.linalg.inv(opencv_c2w)


def _load_observed_initialization(root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    manifest_path = root / "manifest.json"
    done_path = root / "DONE"
    hashes_path = root / "hashes.sha256"
    if not manifest_path.is_file() or not done_path.is_file() or not hashes_path.is_file():
        raise SevaDatasetError("observed visual-hull initialization is incomplete")
    manifest = _load_json(manifest_path, "observed initialization manifest")
    done = _load_json(done_path, "observed initialization DONE")
    if manifest.get("schema_version") != OBSERVED_INIT_SCHEMA:
        raise SevaDatasetError("SEVA dataset requires the observed visual-hull initializer")
    provenance = manifest.get("provenance", {})
    required_false = (
        "generated_geometry",
        "generated_views",
        "learned_depth",
        "secondary_accelerator_artifacts",
    )
    if any(provenance.get(name) is not False for name in required_false):
        raise SevaDatasetError("observed initialization contains generated or learned geometry")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise SevaDatasetError("observed initialization DONE does not bind manifest")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise SevaDatasetError("observed initialization DONE does not bind hashes")
    _verify_hash_index(root)
    points_path = root / manifest["points"]["relpath"]
    if sha256_file(points_path) != manifest["points"]["sha256"]:
        raise SevaDatasetError("observed initialization point hash mismatch")
    vertices, colors = load_colmap_points(points_path)
    if len(vertices) != manifest["points"]["count"]:
        raise SevaDatasetError("observed initialization point count mismatch")
    return manifest, vertices, colors


def _load_rgba(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if value is None or value.shape != (576, 576, 4):
        raise SevaDatasetError(f"expected one 576p RGBA anchor: {path}")
    alpha = value[:, :, 3]
    rgb = cv2.cvtColor(value[:, :, :3], cv2.COLOR_BGR2RGB)
    weight = alpha.astype(np.float32)[..., None] / 255.0
    composite = np.rint(rgb * weight + 255.0 * (1.0 - weight)).astype(np.uint8)
    return composite, alpha


def _load_rgb(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if value is None or value.shape != (576, 576, 3):
        raise SevaDatasetError(f"expected one 576p generated frame: {path}")
    return cv2.cvtColor(value, cv2.COLOR_BGR2RGB)


def _load_mask(path: Path) -> np.ndarray:
    import cv2

    value = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if value is None or value.shape != (576, 576):
        raise SevaDatasetError(f"expected one 576p generated mask: {path}")
    return value


def _estimate_radius(
    vertices: np.ndarray,
    anchors: dict[str, tuple[np.ndarray, np.ndarray]],
    focal_px: float,
) -> tuple[float, dict[str, float]]:
    distances: dict[str, float] = {}
    for label in VIEW_ORDER:
        _, mask = anchors[label]
        _, fit, _ = _fit_projection(vertices, mask >= 128, label)
        pixels_per_m = float(fit["pixels_per_raw_unit"])
        if pixels_per_m <= 0:
            raise SevaDatasetError(f"invalid observed projection fit for {label}")
        distances[label] = focal_px / pixels_per_m
    radius = float(np.median(list(distances.values())))
    if not 0.05 <= radius <= 2.0:
        raise SevaDatasetError(f"fitted camera radius is implausible: {radius}")
    return radius, distances


def _write_hashes(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root).as_posix()}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    import cv2

    four_view_root = args.four_view_input.resolve()
    seva_root = args.seva_run.resolve()
    audit_root = args.audit.resolve()
    review_path = args.review.resolve()
    init_root = args.observed_initialization.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if args.real_repeat < 1 or args.max_points < 1000:
        raise SevaDatasetError("invalid real-repeat or max-points")

    four_view = validate_generation_input(four_view_root)
    seva = _load_json(seva_root / "manifest.json", "SEVA manifest")
    seva_done = _load_json(seva_root / "DONE", "SEVA DONE")
    if seva.get("schema_version") != SEVA_RUN_SCHEMA:
        raise SevaDatasetError("unexpected SEVA run schema")
    if seva_done.get("manifest_sha256") != sha256_file(seva_root / "manifest.json"):
        raise SevaDatasetError("SEVA DONE does not bind manifest")
    if seva_done.get("hashes_sha256") != sha256_file(seva_root / "hashes.sha256"):
        raise SevaDatasetError("SEVA DONE does not bind hashes")
    _verify_hash_index(seva_root)
    audit = _load_json(audit_root / "metrics.json", "SEVA audit")
    audit_done = _load_json(audit_root / "DONE", "SEVA audit DONE")
    if audit.get("schema_version") != AUDIT_SCHEMA:
        raise SevaDatasetError("unexpected SEVA audit schema")
    if audit_done.get("metrics_sha256") != sha256_file(audit_root / "metrics.json"):
        raise SevaDatasetError("SEVA audit DONE does not bind metrics")
    if audit_done.get("hashes_sha256") != sha256_file(audit_root / "hashes.sha256"):
        raise SevaDatasetError("SEVA audit DONE does not bind hashes")
    _verify_hash_index(audit_root)
    review = _load_json(review_path, "SEVA human review")
    if review.get("schema_version") != REVIEW_SCHEMA or review.get("decision") != ACCEPTED:
        raise SevaDatasetError("SEVA orbit was not accepted for pseudo-view training")
    if review.get("evidence", {}).get("audit_metrics_sha256") != sha256_file(
        audit_root / "metrics.json"
    ):
        raise SevaDatasetError("SEVA review does not bind the selected audit")
    if not all(review.get("human_checks", {}).values()) or not all(
        item.get("passed") for item in review.get("numeric_gates", {}).values()
    ):
        raise SevaDatasetError("SEVA review gates are incomplete")
    assert_not_quarantined(
        [
            ("four_view", four_view),
            ("seva", seva),
            ("seva_audit", audit),
            ("seva_review", review),
        ]
    )
    observed_init, vertices, colors = _load_observed_initialization(init_root)
    assert_not_quarantined([("observed_initialization", observed_init)])

    transforms_path = seva_root / "inference/transforms.json"
    transforms = _load_json(transforms_path, "SEVA output transforms")
    camera_frames = transforms.get("frames")
    if not isinstance(camera_frames, list) or len(camera_frames) != 53:
        raise SevaDatasetError("SEVA output requires four input and 49 target cameras")

    by_label = {item["generator_label"]: item for item in four_view["observed_inputs"]}
    anchors = {
        label: _load_rgba(four_view_root / by_label[label]["hunyuan_rgba_relpath"])
        for label in VIEW_ORDER
    }
    focal_values = [float(frame["fl_x"]) for frame in camera_frames]
    if max(focal_values) - min(focal_values) > 1e-6:
        raise SevaDatasetError("SEVA camera focal length is not constant")
    metric_radius_m, per_anchor_radius_m = _estimate_radius(
        vertices, anchors, statistics_fmean(focal_values)
    )

    valid = np.all(np.isfinite(vertices), axis=1)
    candidates = np.flatnonzero(valid)
    if len(candidates) < 1000:
        raise SevaDatasetError("observed initialization has too few finite points")
    if len(candidates) > args.max_points:
        generator = np.random.default_rng(args.sample_seed)
        candidates = np.sort(generator.choice(candidates, args.max_points, replace=False))
    vertices = vertices[candidates]
    colors = colors[candidates]

    selected_generated = select_generated_indices(args.generated_count)
    frame_record_by_index = {index: record for index, record in enumerate(seva["frames"])}
    staging_parent = output.parent
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=staging_parent))
    try:
        image_dir = staging / "images"
        mask_dir = staging / "masks"
        sparse_dir = staging / "sparse/0"
        for directory in (image_dir, mask_dir, sparse_dir):
            directory.mkdir(parents=True)

        entries: list[dict[str, Any]] = []
        generated_records = []
        for index in selected_generated:
            source_record = frame_record_by_index[index]
            rgb = _load_rgb(seva_root / source_record["relpath"])
            mask = _load_mask(audit_root / "generated_masks" / f"{index:05d}.png")
            camera = camera_frames[4 + index]
            intrinsic = np.asarray(
                [
                    [camera["fl_x"], 0.0, camera["cx"]],
                    [0.0, camera["fl_y"], camera["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            w2c = opengl_c2w_to_metric_opencv_w2c(
                np.asarray(camera["transform_matrix"], dtype=np.float64), metric_radius_m
            )
            name = f"gen_{index:05d}.png"
            entry = {
                "name": name,
                "rgb": rgb,
                "mask": mask,
                "intrinsic": intrinsic,
                "w2c": w2c,
                "provenance": "generated_low_confidence_SEVA_pseudoview",
                "source_index": index,
                "sampling_duplicate": False,
            }
            entries.append(entry)
            generated_records.append(
                {
                    "name": name,
                    "source_index": index,
                    "source_sha256": source_record["sha256"],
                }
            )
        entries.append(
            {
                **entries[0],
                "name": "000_eval_probe_generated.png",
                "provenance": "generated_eval_probe_duplicate_not_training_evidence",
                "sampling_duplicate": True,
            }
        )

        real_records = []
        for label in VIEW_ORDER:
            rgb, mask = anchors[label]
            camera = camera_frames[SEVA_INPUT_INDEX_BY_LABEL[label]]
            intrinsic = np.asarray(
                [
                    [camera["fl_x"], 0.0, camera["cx"]],
                    [0.0, camera["fl_y"], camera["cy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )
            w2c = opengl_c2w_to_metric_opencv_w2c(
                np.asarray(camera["transform_matrix"], dtype=np.float64), metric_radius_m
            )
            for repeat in range(args.real_repeat):
                entries.append(
                    {
                        "name": f"real_{label}_w{repeat:02d}.png",
                        "rgb": rgb,
                        "mask": mask,
                        "intrinsic": intrinsic,
                        "w2c": w2c,
                        "provenance": "observed_real_anchor_sampling_duplicate",
                        "source_index": None,
                        "sampling_duplicate": repeat > 0,
                    }
                )
            real_records.append(
                {
                    "view": label,
                    "source_id": by_label[label]["id"],
                    "source_hashes": by_label[label]["source_hashes"],
                    "training_repetitions": args.real_repeat,
                    "camera_role": "nominal_SEVA_input_orientation_metric_radius_fit_to_observed_mask",
                }
            )

        entries.sort(key=lambda item: item["name"])
        camera_lines = ["# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"]
        image_lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"]
        exported = []
        for image_id, entry in enumerate(entries, 1):
            name = entry["name"]
            if not cv2.imwrite(
                str(image_dir / name), cv2.cvtColor(entry["rgb"], cv2.COLOR_RGB2BGR)
            ):
                raise RuntimeError(f"failed to write {name}")
            if not cv2.imwrite(str(mask_dir / name), np.asarray(entry["mask"], dtype=np.uint8)):
                raise RuntimeError(f"failed to write mask {name}")
            camera_lines.append(_colmap_camera_line_square(image_id, entry["intrinsic"]))
            image_lines.append(_colmap_image_line(image_id, image_id, name, entry["w2c"]))
            exported.append(
                {
                    key: entry[key]
                    for key in ("name", "provenance", "source_index", "sampling_duplicate")
                }
            )
        (sparse_dir / "cameras.txt").write_text("".join(camera_lines), encoding="utf-8")
        (sparse_dir / "images.txt").write_text("".join(image_lines), encoding="utf-8")
        point_lines = ["# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n"]
        for point_id, (point, color) in enumerate(zip(vertices, colors, strict=True), 1):
            point_lines.append(
                f"{point_id} {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0\n"
            )
        (sparse_dir / "points3D.txt").write_text("".join(point_lines), encoding="utf-8")

        real_instances = 4 * args.real_repeat
        generated_instances = len(selected_generated)
        training_total = real_instances + generated_instances
        manifest = {
            "schema_version": SCHEMA,
            "created_utc": utc_now(),
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "asset_name": four_view["asset_name"],
            "lineage": {
                "four_view_manifest_sha256": sha256_file(four_view_root / "manifest.json"),
                "seva_manifest_sha256": sha256_file(seva_root / "manifest.json"),
                "seva_audit_metrics_sha256": sha256_file(audit_root / "metrics.json"),
                "seva_human_review_sha256": sha256_file(review_path),
                "observed_initialization_manifest_sha256": sha256_file(
                    init_root / "manifest.json"
                ),
                "inherited_mesh_or_procedural_geometry": None,
            },
            "sampling": {
                "real_unique_views": 4,
                "real_repetitions_per_view": args.real_repeat,
                "real_training_instances": real_instances,
                "generated_available_views": 49,
                "generated_selected_training_views": generated_instances,
                "generated_selected_indices": selected_generated,
                "generated_anchor_indices_excluded": sorted(ANCHOR_INDEX_BY_LABEL.values()),
                "generated_eval_probe_duplicates": 1,
                "training_instances_total": training_total,
                "nominal_real_sampling_probability": real_instances / training_total,
                "mechanism": "real_filename_repetition_plus_even_generated_subsampling",
            },
            "vksplat_split_contract": {
                "recommended_eval_interval": len(entries),
                "lexicographically_first_eval_image": entries[0]["name"],
                "all_unique_real_anchors_train": True,
                "eval_probe_is_generated_duplicate": True,
            },
            "camera_contract": {
                "orientation_and_intrinsics": "exact_SEVA_target_camera_contract",
                "normalized_radius_resolved_from_real_masks": True,
                "metric_radius_m": metric_radius_m,
                "per_anchor_radius_fit_m": per_anchor_radius_m,
                "photogrammetrically_calibrated": False,
            },
            "generated_views": {
                "source": "Stable_Virtual_Camera_v1.1_four_image_orbit",
                "records": generated_records,
                "mask_source": "audited_inferred_foreground",
                "confidence": "low",
                "allowed_role": "appearance_completion_training_only",
            },
            "real_views": real_records,
            "initial_points": {
                "count": int(len(vertices)),
                "source": "observed_real_mask_CPU_visual_hull",
                "source_points_sha256": observed_init["points"]["sha256"],
                "source_manifest_sha256": sha256_file(init_root / "manifest.json"),
                "generated_geometry_prior": False,
                "observed_visual_hull_prior": True,
                "metric_truth": False,
            },
            "images": exported,
            "required_training_profile": {
                "freeze_geometry": True,
                "disable_refinement": True,
                "keep_output_as_generated_fill_layer": True,
            },
            "provenance_boundary": [
                "SEVA frames are low-confidence pseudo-views, never observed evidence.",
                "Four generated anchor target frames are excluded so real anchors retain authority.",
                "Repeated real files change sampling frequency, not evidence count.",
                "Only the observed CPU visual hull initializes GS geometry.",
                "The output remains a generated-fill layer separate from the observed core.",
            ],
        }
        (staging / "dataset_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha = _write_hashes(staging)
        (staging / "DONE").write_text(
            json.dumps(
                {
                    "schema_version": DONE_SCHEMA,
                    "status": "done_nonformal_SEVA_pseudoview_dataset",
                    "manifest_sha256": sha256_file(staging / "dataset_manifest.json"),
                    "hashes_sha256": hashes_sha,
                    "completed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(staging, output)
        return manifest
    except BaseException as exc:
        (staging / "FAILED").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.seva_pseudoview_colmap_failure.v1",
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_utc": utc_now(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        failed = output.with_name(f"{output.name}.FAILED")
        if not failed.exists():
            os.replace(staging, failed)
        else:
            shutil.rmtree(staging)
        raise


def statistics_fmean(values: list[float]) -> float:
    return float(sum(values) / len(values))


def _colmap_camera_line_square(camera_id: int, intrinsic: np.ndarray) -> str:
    width, height = IMAGE_SIZE_WH
    return (
        f"{camera_id} PINHOLE {width} {height} {intrinsic[0, 0]:.17g} "
        f"{intrinsic[1, 1]:.17g} {intrinsic[0, 2]:.17g} {intrinsic[1, 2]:.17g}\n"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--four-view-input", type=Path, required=True)
    parser.add_argument("--seva-run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--observed-initialization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-repeat", type=int, default=12)
    parser.add_argument("--generated-count", type=int, default=24)
    parser.add_argument("--max-points", type=int, default=30000)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    return parser


def main() -> None:
    result = build_dataset(build_parser().parse_args())
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "sampling": result["sampling"],
                "camera_contract": result["camera_contract"],
                "initial_points": result["initial_points"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
