#!/usr/bin/env python3
"""Generate text-grounded object masks with the existing MI300X SAM3 stack."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import types
from typing import Any


VIEW_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_image(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--image must have the form view_id=/path/image.png")
    view_id, raw_path = value.split("=", 1)
    if not VIEW_NAME_PATTERN.fullmatch(view_id):
        raise argparse.ArgumentTypeError(f"invalid view id: {view_id}")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"image does not exist: {path}")
    return view_id, path


def select_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = [
        item
        for item in candidates
        if 0.03 <= float(item["area_fraction"]) <= 0.75
        and not bool(item["touches_border"])
        and float(item["score"]) >= 0.25
    ]
    if not accepted:
        raise RuntimeError("SAM3 produced no candidate that passed area, border, and score gates")
    return max(accepted, key=lambda item: (float(item["score"]), -float(item["area_fraction"])))


def _install_pkg_resources_compat(vista4d_root: Path) -> None:
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        module = types.ModuleType("pkg_resources")
        module.resource_filename = lambda package, relative: str(
            vista4d_root / package / relative
        )
        sys.modules["pkg_resources"] = module


def _mask_bbox(mask: Any, np: Any) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _touches_border(mask: Any, np: Any) -> bool:
    return bool(
        np.any(mask[0, :])
        or np.any(mask[-1, :])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _fill_small_holes(mask: Any, max_area_px: int, cv2: Any, np: Any) -> tuple[Any, dict[str, int]]:
    inverse = (~mask).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(inverse, connectivity=8)
    cleaned = mask.copy()
    holes_filled = 0
    pixels_filled = 0
    height, width = mask.shape
    for label in range(1, count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        touches = (
            x == 0
            or y == 0
            or x + component_width >= width
            or y + component_height >= height
        )
        if not touches and area <= max_area_px:
            cleaned[labels == label] = True
            holes_filled += 1
            pixels_filled += area
    return cleaned, {"holes_filled": holes_filled, "pixels_filled": pixels_filled}


def run(args: argparse.Namespace) -> dict[str, Any]:
    vista4d_root = args.vista4d_root.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (vista4d_root / "sam3/model_builder.py").is_file():
        raise FileNotFoundError("Vista4D SAM3 source tree is incomplete")
    image_pairs = args.image
    view_ids = [item[0] for item in image_pairs]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("duplicate --image view id")

    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageFilter

    sys.path.insert(0, str(vista4d_root))
    _install_pkg_resources_compat(vista4d_root)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("SAM3 object masking requires exactly one visible ROCm device")
    hardware = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "device_count": torch.cuda.device_count(),
    }
    bpe_path = vista4d_root / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"
    started = time.monotonic()
    model = build_sam3_image_model(
        bpe_path=str(bpe_path),
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging.", dir=output.parent))
    results: list[dict[str, Any]] = []
    montage_rows = []
    try:
        for view_id, image_path in image_pairs:
            image = Image.open(image_path).convert("RGB")
            image_array = np.asarray(image)
            state = processor.set_image(image)
            candidates: list[dict[str, Any]] = []
            masks: dict[str, Any] = {}
            for prompt in args.prompt:
                processor.reset_all_prompts(state)
                state = processor.set_text_prompt(prompt, state)
                scores = state["scores"].detach().float().cpu().tolist()
                prompt_masks = state["masks"].detach().cpu().numpy()
                boxes = state["boxes"].detach().float().cpu().tolist()
                for index, (score, raw_mask, box) in enumerate(
                    zip(scores, prompt_masks, boxes, strict=True)
                ):
                    mask = np.asarray(raw_mask).squeeze().astype(bool)
                    candidate_id = f"{len(candidates):03d}"
                    masks[candidate_id] = mask
                    candidates.append(
                        {
                            "candidate_id": candidate_id,
                            "prompt": prompt,
                            "prompt_index": index,
                            "score": float(score),
                            "area_fraction": float(mask.mean()),
                            "touches_border": _touches_border(mask, np),
                            "bbox_xyxy": [float(value) for value in box],
                            "mask_bbox_xyxy": _mask_bbox(mask, np),
                        }
                    )
            selected = select_candidate(candidates)
            mask = masks[str(selected["candidate_id"])]
            mask, hole_fill = _fill_small_holes(
                mask, args.max_hole_area_px, cv2, np
            )
            selected = {
                **selected,
                "postprocess": {
                    "method": "fill_enclosed_components_below_area",
                    "max_hole_area_px": args.max_hole_area_px,
                    **hole_fill,
                    "final_area_fraction": float(mask.mean()),
                    "final_mask_bbox_xyxy": _mask_bbox(mask, np),
                },
            }
            mask_u8 = (mask.astype(np.uint8) * 255)
            mask_image = Image.fromarray(mask_u8, mode="L")
            alpha_image = mask_image.filter(ImageFilter.GaussianBlur(radius=args.alpha_blur_px))
            alpha = np.asarray(alpha_image).astype(np.float32)[:, :, None] / 255.0
            neutral = np.full_like(image_array, args.neutral_value)
            neutral_array = np.clip(
                image_array.astype(np.float32) * alpha
                + neutral.astype(np.float32) * (1.0 - alpha),
                0,
                255,
            ).astype(np.uint8)
            edge = np.asarray(mask_image.filter(ImageFilter.MaxFilter(7))) > np.asarray(
                mask_image.filter(ImageFilter.MinFilter(7))
            )
            overlay = image_array.copy()
            overlay[edge] = np.array([0, 255, 0], dtype=np.uint8)

            relpaths = {
                "mask": Path("masks") / f"{view_id}.png",
                "alpha": Path("alpha") / f"{view_id}.png",
                "neutral_rgb": Path("neutral_rgb") / f"{view_id}.png",
                "overlay": Path("qa") / f"{view_id}_overlay.png",
            }
            images = {
                "mask": mask_image,
                "alpha": alpha_image,
                "neutral_rgb": Image.fromarray(neutral_array, mode="RGB"),
                "overlay": Image.fromarray(overlay, mode="RGB"),
            }
            for key, relpath in relpaths.items():
                destination = staging / relpath
                destination.parent.mkdir(parents=True, exist_ok=True)
                images[key].save(destination)
            results.append(
                {
                    "view_id": view_id,
                    "source_basename": image_path.name,
                    "source_sha256": sha256_file(image_path),
                    "selected": selected,
                    "candidates": candidates,
                    "outputs": {
                        key: {
                            "relpath": relpath.as_posix(),
                            "sha256": sha256_file(staging / relpath),
                        }
                        for key, relpath in relpaths.items()
                    },
                }
            )
            tile = 320
            original_tile = image.resize((tile, tile))
            overlay_tile = images["overlay"].resize((tile, tile))
            neutral_tile = images["neutral_rgb"].resize((tile, tile))
            row = np.concatenate(
                [np.asarray(original_tile), np.asarray(overlay_tile), np.asarray(neutral_tile)],
                axis=1,
            )
            montage_rows.append(row)

        montage = np.concatenate(montage_rows, axis=0)
        montage_path = staging / "qa/masks_montage.png"
        Image.fromarray(montage, mode="RGB").save(montage_path)
        manifest = {
            "schema_version": "radeon_oneloop.sam3_object_masks.v1",
            "created_utc": utc_now(),
            "formal": False,
            "host_role": "phi_amd_work_mi300x_nonformal_generation_preprocess",
            "method": "sam3_image_text_prompt",
            "checkpoint_sha256": sha256_file(checkpoint),
            "prompts": list(args.prompt),
            "confidence_threshold": args.confidence_threshold,
            "alpha_blur_px": args.alpha_blur_px,
            "neutral_value": args.neutral_value,
            "max_hole_area_px": args.max_hole_area_px,
            "hardware": hardware,
            "runtime_s": time.monotonic() - started,
            "views": results,
            "mask_review_status": "pending_visual_review",
            "qa_montage": {
                "relpath": "qa/masks_montage.png",
                "sha256": sha256_file(montage_path),
            },
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
                    "stage": "MI300X_SAM3_object_masks",
                    "completed_utc": utc_now(),
                    "manifest_sha256": sha256_file(manifest_path),
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
                {"failed_utc": utc_now(), "error_type": type(exc).__name__, "error": str(exc)},
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", action="append", type=parse_image, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vista4d-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prompt", action="append", default=["plush toy", "Mickey Mouse plush toy"]
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.25)
    parser.add_argument("--alpha-blur-px", type=float, default=2.0)
    parser.add_argument("--neutral-value", type=int, default=127)
    parser.add_argument("--max-hole-area-px", type=int, default=1500)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise ValueError("confidence threshold must be in [0, 1]")
    if not 0.0 <= args.alpha_blur_px <= 20.0:
        raise ValueError("alpha blur must be in [0, 20]")
    if not 0 <= args.neutral_value <= 255:
        raise ValueError("neutral value must be in [0, 255]")
    if not 0 <= args.max_hole_area_px <= 100000:
        raise ValueError("max hole area must be in [0, 100000]")
    manifest = run(args)
    print(json.dumps({"output": str(args.output), "views": len(manifest["views"])}, indent=2))


if __name__ == "__main__":
    main()
