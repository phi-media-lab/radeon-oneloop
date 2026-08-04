#!/usr/bin/env python3
"""Render a static object Gaussian into a Vista4D-compatible conditioning bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

import numpy as np

from sim.genesis_so101.gaussian_appearance import (
    PinholeCamera,
    VkSplatAppearanceRenderer,
    nonformal_candidate_asset,
    observed_core_asset,
)
from sim.genesis_so101.gaussian_orbit_audit import (
    canonical_orbit_extrinsic,
    scaled_intrinsic,
)


VISTA4D_FRAMES = 49


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vista4d_camera_track(
    *,
    frames: int,
    intrinsic_3x3: np.ndarray,
    distance_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Vista4D external-convention c2w and [fx, fy, cx, cy].

    Vista4D's point-cloud renderer premultiplies stored c2w matrices by
    ``diag(-1, -1, 1, 1)`` before projection.  We apply the inverse of that
    conversion here so the resulting render-space camera is the accepted
    canonical OpenCV orbit.
    """

    if frames <= 0:
        raise ValueError("frames must be positive")
    intrinsic = np.asarray(intrinsic_3x3, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
        raise ValueError("intrinsic must be a finite 3x3 matrix")
    conversion = np.diag([-1.0, -1.0, 1.0, 1.0])
    stored_c2w = []
    for index in range(frames):
        azimuth_deg = 360.0 * index / frames
        camera_from_object = canonical_orbit_extrinsic(
            azimuth_deg, distance_m=distance_m
        )
        render_c2w = np.linalg.inv(camera_from_object)
        stored_c2w.append(conversion @ render_c2w)
    intrinsics = np.repeat(
        np.asarray(
            [[intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]]],
            dtype=np.float64,
        ),
        frames,
        axis=0,
    )
    return np.stack(stored_c2w), intrinsics


def _write_hash_index(root: Path) -> str:
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name not in {"hashes.sha256", "DONE", "FAILED"}:
            lines.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    target = root / "hashes.sha256"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(target)


def load_surface_carrier_source(
    root: Path,
    *,
    width: int,
    height: int,
    target_c2w: np.ndarray,
    target_intrinsics: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, object]]:
    """Load the rejected carrier orbit solely for reproducible ablation."""

    carrier_root = root.resolve()
    manifest_path = carrier_root / "manifest.json"
    done_path = carrier_root / "DONE"
    hashes_path = carrier_root / "hashes.sha256"
    for path in (manifest_path, done_path, hashes_path):
        if not path.is_file():
            raise FileNotFoundError(f"surface carrier is incomplete: {path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    done = json.loads(done_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "radeon_oneloop.surface_carrier.v1":
        raise ValueError("unsupported surface-carrier schema")
    if manifest.get("formal") is not False or manifest.get("accepted_numeric") is not True:
        raise ValueError("surface carrier must preserve its historical numeric record")
    if manifest.get("visual_review_required") is not True:
        raise ValueError("surface carrier must preserve the visual-review requirement")
    if done.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("surface-carrier DONE marker does not bind its manifest")
    if done.get("hashes_sha256") != sha256_file(hashes_path):
        raise ValueError("surface-carrier DONE marker does not bind its hash index")
    for line in hashes_path.read_text(encoding="utf-8").splitlines():
        expected, relpath = line.split("  ", maxsplit=1)
        path = carrier_root / relpath
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"surface-carrier hash mismatch: {relpath}")

    orbit = manifest.get("orbit", {})
    if orbit.get("frames") != VISTA4D_FRAMES:
        raise ValueError("surface carrier does not contain the required 49 frames")
    if orbit.get("image_size_wh") != [width, height]:
        raise ValueError("surface-carrier frame dimensions do not match conditioning")
    cameras_path = carrier_root / str(orbit["target_cameras_relpath"])
    with np.load(cameras_path, allow_pickle=False) as stored:
        stored_c2w = np.asarray(stored["cam_c2w"], dtype=np.float64)
        stored_intrinsics = np.asarray(stored["intrinsics"], dtype=np.float64)
    if not np.allclose(stored_c2w, target_c2w, atol=1.0e-10, rtol=0.0):
        raise ValueError("surface-carrier camera trajectory does not match Vista4D input")
    if not np.allclose(
        stored_intrinsics, target_intrinsics, atol=1.0e-10, rtol=0.0
    ):
        raise ValueError("surface-carrier intrinsics do not match Vista4D input")

    from PIL import Image

    frame_root = carrier_root / str(orbit["frames_relpath"])
    alpha_root = carrier_root / str(orbit["alpha_relpath"])
    frames = []
    alphas = []
    for index in range(VISTA4D_FRAMES):
        frame_path = frame_root / f"{index:05d}.png"
        alpha_path = alpha_root / f"{index:05d}.png"
        frame = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
        alpha = np.asarray(Image.open(alpha_path).convert("L"), dtype=np.float64) / 255.0
        if frame.shape != (height, width, 3) or alpha.shape != (height, width):
            raise ValueError(f"surface-carrier frame {index} has an invalid shape")
        frames.append(frame)
        alphas.append(alpha)
    return frames, alphas, {
        "manifest_sha256": sha256_file(manifest_path),
        "hashes_sha256": sha256_file(hashes_path),
        "carrier_ply_sha256": manifest["geometry"]["ply_sha256"],
        "carrier_role": manifest["carrier_role"],
        "real_view_silhouette_iou_mean": manifest["real_view_audit"][
            "silhouette_iou_mean"
        ],
        "observed_vertex_fraction": manifest["appearance"][
            "observed_vertex_fraction"
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--vksplat-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--distance-m", type=float, default=0.3)
    parser.add_argument("--background", type=float, default=1.0)
    parser.add_argument("--alpha-threshold", type=float, default=1.0e-3)
    parser.add_argument(
        "--real-view-root",
        type=Path,
        help=(
            "Optional 01_normalized folder containing neutral_rgb/ and alpha/; "
            "the four real views become identity keyframes in the source video."
        ),
    )
    parser.add_argument(
        "--surface-carrier-root",
        type=Path,
        help=(
            "Optional accepted surface-carrier artifact; its complete 49-frame "
            "orbit becomes the source video while the observed Gaussian remains "
            "the point-cloud condition."
        ),
    )
    parser.add_argument(
        "--allow-rejected-surface-carrier-ablation",
        action="store_true",
        help=(
            "Explicitly reproduce the rejected procedural-carrier negative control. "
            "This flag is forbidden in the learned-completion mainline."
        ),
    )
    parser.add_argument(
        "--candidate-nonformal",
        action="store_true",
        help="Use a self-bound formal=false candidate rather than the pinned default.",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.width, args.height) <= 0:
        raise ValueError("output dimensions must be positive")
    if not 1.0 <= args.fps <= 60.0:
        raise ValueError("fps must be between 1 and 60")
    if not 0.0 <= args.background <= 1.0:
        raise ValueError("background must be in [0, 1]")
    if not 0.0 < args.alpha_threshold < 1.0:
        raise ValueError("alpha threshold must be in (0, 1)")
    if not math.isfinite(args.distance_m) or args.distance_m <= 0.0:
        raise ValueError("camera distance must be finite and positive")
    if args.real_view_root is not None and args.surface_carrier_root is not None:
        raise ValueError("real-view keyframes and a surface-carrier source are exclusive")
    if (
        args.surface_carrier_root is not None
        and not args.allow_rejected_surface_carrier_ablation
    ):
        raise ValueError(
            "surface carrier is a rejected geometry prior; pass the explicit "
            "negative-control flag only to reproduce the historical ablation"
        )

    output = args.output.resolve()
    output.mkdir(parents=True)
    frame_dir = output / "frames"
    mask_names = (
        "alpha_mask_src",
        "dynamic_mask_src",
        "alpha_mask_pc",
        "dynamic_mask_pc",
    )
    frame_dir.mkdir()
    for name in mask_names:
        (output / name).mkdir()

    try:
        import imageio.v3 as iio
        from PIL import Image

        asset = (
            nonformal_candidate_asset(args.asset_root)
            if args.candidate_nonformal
            else observed_core_asset(args.asset_root)
        )
        asset_audit = asset.validate()
        cameras_document = json.loads(asset.cameras_path.read_text(encoding="utf-8"))
        front = cameras_document["cameras"][0]
        source_width, source_height = (int(value) for value in front["image_size_wh"])
        intrinsic = scaled_intrinsic(
            np.asarray(front["intrinsic_3x3"], dtype=np.float64),
            (source_width, source_height),
            (args.width, args.height),
        )
        target_c2w, target_intrinsics = vista4d_camera_track(
            frames=VISTA4D_FRAMES,
            intrinsic_3x3=intrinsic,
            distance_m=args.distance_m,
        )

        real_keyframes: dict[int, tuple[str, np.ndarray, np.ndarray, dict[str, str]]] = {}
        carrier_frames: list[np.ndarray] = []
        carrier_alphas: list[np.ndarray] = []
        carrier_record: dict[str, object] | None = None
        if args.surface_carrier_root is not None:
            carrier_frames, carrier_alphas, carrier_record = load_surface_carrier_source(
                args.surface_carrier_root,
                width=args.width,
                height=args.height,
                target_c2w=target_c2w,
                target_intrinsics=target_intrinsics,
            )
            (output / "frames_source").mkdir()
        if args.real_view_root is not None:
            real_view_root = args.real_view_root.resolve()
            keyframe_layout = {
                0: "anchor_front",
                12: "anchor_right",
                24: "anchor_rear",
                37: "anchor_left",
            }
            for frame_index, view_id in keyframe_layout.items():
                rgb_path = real_view_root / "neutral_rgb" / f"{view_id}.png"
                alpha_path = real_view_root / "alpha" / f"{view_id}.png"
                if not rgb_path.is_file() or not alpha_path.is_file():
                    raise FileNotFoundError(f"missing normalized real view for {view_id}")
                real_rgb = np.asarray(
                    Image.open(rgb_path)
                    .convert("RGB")
                    .resize((args.width, args.height), Image.Resampling.LANCZOS),
                    dtype=np.float64,
                ) / 255.0
                real_alpha = np.asarray(
                    Image.open(alpha_path)
                    .convert("L")
                    .resize((args.width, args.height), Image.Resampling.NEAREST),
                    dtype=np.float64,
                ) / 255.0
                real_keyframes[frame_index] = (
                    view_id,
                    np.round(real_rgb * 255.0).astype(np.uint8),
                    real_alpha,
                    {
                        "neutral_rgb_sha256": sha256_file(rgb_path),
                        "alpha_sha256": sha256_file(alpha_path),
                    },
                )
            (output / "frames_source").mkdir()

        source_frames: list[np.ndarray] = []
        point_frames: list[np.ndarray] = []
        source_alpha_masks: list[np.ndarray] = []
        point_alpha_masks: list[np.ndarray] = []
        render_ms: list[float] = []
        renderer = VkSplatAppearanceRenderer(asset, args.vksplat_root)
        try:
            for index in range(VISTA4D_FRAMES):
                azimuth_deg = 360.0 * index / VISTA4D_FRAMES
                camera = PinholeCamera(
                    width=args.width,
                    height=args.height,
                    intrinsic_3x3=intrinsic,
                    camera_from_object_opencv_4x4=canonical_orbit_extrinsic(
                        azimuth_deg, distance_m=args.distance_m
                    ),
                )
                frame = renderer.render(camera)
                composite = np.clip(
                    frame.premultiplied_rgb
                    + (1.0 - frame.alpha) * args.background,
                    0.0,
                    1.0,
                )
                rgb_u8 = np.round(composite * 255.0).astype(np.uint8)
                alpha = frame.alpha[..., 0] > args.alpha_threshold
                zero = np.zeros_like(alpha, dtype=np.uint8)
                alpha_u8 = alpha.astype(np.uint8) * 255
                iio.imwrite(frame_dir / f"{index:05d}.png", rgb_u8)
                source_rgb_u8 = rgb_u8
                source_alpha = alpha
                if carrier_record is not None:
                    source_rgb_u8 = carrier_frames[index]
                    source_alpha_soft = carrier_alphas[index]
                    source_alpha = source_alpha_soft > args.alpha_threshold
                if index in real_keyframes:
                    _, real_rgb_u8, real_alpha, _ = real_keyframes[index]
                    source_alpha_soft = real_alpha
                    source_alpha = source_alpha_soft > args.alpha_threshold
                    source_rgb_u8 = np.round(
                        np.clip(
                            (real_rgb_u8.astype(np.float64) / 255.0)
                            * source_alpha_soft[..., None]
                            + (1.0 - source_alpha_soft[..., None]) * args.background,
                            0.0,
                            1.0,
                        )
                        * 255.0
                    ).astype(np.uint8)
                source_alpha_u8 = source_alpha.astype(np.uint8) * 255
                if carrier_record is not None or index in real_keyframes:
                    iio.imwrite(output / "frames_source" / f"{index:05d}.png", source_rgb_u8)
                iio.imwrite(
                    output / "alpha_mask_src" / f"{index:05d}.png", source_alpha_u8
                )
                iio.imwrite(output / "alpha_mask_pc" / f"{index:05d}.png", alpha_u8)
                iio.imwrite(output / "dynamic_mask_src" / f"{index:05d}.png", zero)
                iio.imwrite(output / "dynamic_mask_pc" / f"{index:05d}.png", zero)
                source_frames.append(source_rgb_u8)
                point_frames.append(rgb_u8)
                source_alpha_masks.append(source_alpha)
                point_alpha_masks.append(alpha)
                render_ms.append(frame.render_ms)
            memory = renderer.memory_usage()
        finally:
            renderer.close()

        videos = {
            "video_src.mp4": np.stack(source_frames),
            "video_pc.mp4": np.stack(point_frames),
        }
        for name, video in videos.items():
            iio.imwrite(
                output / name,
                video,
                fps=args.fps,
                codec="libx264",
                pixelformat="yuv420p",
            )
        np.savez(
            output / "cameras_tgt.npz",
            cam_c2w=target_c2w,
            intrinsics=target_intrinsics,
        )
        point_support = [float(mask.mean()) for mask in point_alpha_masks]
        source_support = [float(mask.mean()) for mask in source_alpha_masks]
        keyframe_manifest = [
            {
                "frame_index": frame_index,
                "view_id": value[0],
                **value[3],
            }
            for frame_index, value in sorted(real_keyframes.items())
        ]
        manifest = {
            "schema_version": "radeon_oneloop.vista4d_object_conditioning.v1",
            "formal": False,
            "eligible_for_formal_metrics": False,
            "eligible_for_heldout_real_metrics": False,
            "host_role": "amd_apu_nonformal_conditioning_render",
            "frames": VISTA4D_FRAMES,
            "image_size_wh": [args.width, args.height],
            "fps": args.fps,
            "camera": {
                "trajectory": "closed_level_canonical_orbit",
                "distance_m": args.distance_m,
                "stored_convention": "vista4d_external_c2w",
                "render_conversion": "diag(-1,-1,1,1)_left_multiply",
                "canonical_render_convention": "opencv_camera_positive_z",
            },
            "source_video": {
                "role": (
                    "rejected_procedural_surface_carrier_negative_control_orbit"
                    if carrier_record is not None
                    else (
                        "real_photo_keyframes_plus_observed_gaussian_orbit"
                        if real_keyframes
                        else "real_photo_optimized_observed_gaussian_orbit"
                    )
                ),
                "relpath": "video_src.mp4",
                "alpha_support_fraction": {
                    "min": min(source_support),
                    "mean": statistics.fmean(source_support),
                    "max": max(source_support),
                },
                "real_identity_keyframes": keyframe_manifest,
                "surface_carrier": carrier_record,
            },
            "point_cloud_condition": {
                "role": "same_observed_gaussian_orbit_with_exact_alpha_support",
                "relpath": "video_pc.mp4",
                "alpha_support_fraction": {
                    "min": min(point_support),
                    "mean": statistics.fmean(point_support),
                    "max": max(point_support),
                },
                "dynamic": False,
            },
            "asset": asset_audit,
            "render_ms": {
                "mean": statistics.fmean(render_ms),
                "p95": float(np.percentile(render_ms, 95)),
                "max": max(render_ms),
            },
            "vksplat_memory": memory,
            "allowed_role": "vista4d_generated_appearance_pseudoview_conditioning",
            "not_proven": [
                "metric geometry completion",
                "held-out real-view quality",
                "physics collision geometry",
            ],
            "physical_output": False,
            "redistribution": False,
        }
        manifest_path = output / "input_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        hashes_sha256 = _write_hash_index(output)
        done = {
            "schema_version": manifest["schema_version"],
            "status": "complete",
            "manifest_sha256": sha256_file(manifest_path),
            "hashes_sha256": hashes_sha256,
        }
        (output / "DONE").write_text(
            json.dumps(done, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except Exception as error:
        (output / "FAILED").write_text(
            json.dumps(
                {
                    "schema_version": "radeon_oneloop.vista4d_object_conditioning.v1",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
