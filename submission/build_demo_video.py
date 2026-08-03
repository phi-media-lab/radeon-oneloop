#!/usr/bin/env python3
"""Build the evidence-backed 3-5 minute Radeon OneLoop demo video.

The renderer reads only registered formal result JSON, creates 1080p slides,
uses the local macOS speech synthesizer, and joins H.264/AAC segments with
ffmpeg.  Private robot video is deliberately not consumed.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


WIDTH, HEIGHT = 1920, 1080
INK = "#172235"
MUTED = "#667085"
ACCENT = "#B91C3B"
NAVY = "#31587A"
PALE = "#F4F6F9"
WHITE = "#FFFFFF"


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        candidates = [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        ]
    elif bold:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(require(path).read_text(encoding="utf-8"))


def command(arguments: list[str]) -> None:
    subprocess.run(arguments, check=True)


def fit_text(draw: ImageDraw.ImageDraw, value: str, width: int, text_font: ImageFont.FreeTypeFont) -> list[str]:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and draw.textlength(candidate, font=text_font) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def slide(
    path: Path,
    *,
    kicker: str,
    title: str,
    bullets: list[str],
    metric: str | None = None,
    image_path: Path | None = None,
    mono_lines: list[str] | None = None,
) -> None:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 18), fill=ACCENT)
    draw.text((110, 72), kicker.upper(), fill=ACCENT, font=font(24, bold=True))
    draw.text((110, 122), title, fill=INK, font=font(58, bold=True))
    draw.line((110, 215, 1810, 215), fill="#D8DDE6", width=3)

    if metric:
        draw.rounded_rectangle((1120, 260, 1810, 405), radius=18, fill=PALE)
        metric_font = font(39, bold=True)
        lines = fit_text(draw, metric, 620, metric_font)
        for index, line in enumerate(lines[:2]):
            draw.text((1160, 287 + index * 52), line, fill=ACCENT, font=metric_font)

    text_right = 1040 if image_path or metric else 1750
    bullet_font = font(33)
    y = 285
    for value in bullets:
        wrapped = fit_text(draw, value, text_right - 170, bullet_font)
        draw.ellipse((112, y + 14, 130, y + 32), fill=NAVY)
        for index, line in enumerate(wrapped):
            draw.text((155, y + index * 48), line, fill=INK, font=bullet_font)
        y += len(wrapped) * 48 + 42

    if image_path:
        picture = Image.open(require(image_path)).convert("RGB")
        picture = ImageOps.contain(picture, (690, 480))
        x = 1115 + (690 - picture.width) // 2
        y_image = 435 + (480 - picture.height) // 2
        canvas.paste(picture, (x, y_image))
        draw.rectangle((x, y_image, x + picture.width, y_image + picture.height), outline="#D8DDE6", width=3)

    if mono_lines:
        y_mono = 760
        draw.rounded_rectangle((110, y_mono - 25, 1810, 955), radius=14, fill="#EEF1F5")
        mono = font(22, mono=True)
        for index, line in enumerate(mono_lines):
            draw.text((145, y_mono + index * 48), line, fill=INK, font=mono)

    draw.text((110, 1015), "Phi Media Lab · Radeon OneLoop · Track 3", fill=MUTED, font=font(20))
    draw.text((1535, 1015), "One Radeon. One robot loop.", fill=ACCENT, font=font(20, bold=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format="PNG", optimize=True)


def speech_segment(
    slide_path: Path,
    narration: str,
    output: Path,
    *,
    voice: str,
    rate: int,
    audio_path: Path,
) -> None:
    command(["say", "-v", voice, "-r", str(rate), "-o", str(audio_path), narration])
    command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(slide_path),
            "-i",
            str(audio_path),
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def duration(path: Path) -> float:
    value = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        text=True,
    )
    return float(value.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/formal"))
    parser.add_argument("--figures-root", type=Path, default=Path("reports/figures"))
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/video-build"))
    parser.add_argument(
        "--output", type=Path, default=Path("output/video/radeon-oneloop-demo.mp4")
    )
    parser.add_argument("--voice", default="Samantha")
    parser.add_argument("--rate", type=int, default=260)
    args = parser.parse_args()

    for executable in ("say", "ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is unavailable: {executable}")

    evidence = args.evidence_root.resolve()
    figures = args.figures_root.resolve()
    paired = read_json(evidence / "paired_training_summary.json")
    base_latency = read_json(evidence / "baseline_latency/metrics.json")
    phase_latency = read_json(evidence / "phase_latency/metrics.json")
    base_recon = read_json(evidence / "baseline_reconstruction/metrics.json")
    phase_recon = read_json(evidence / "phase_reconstruction/metrics.json")
    genesis = read_json(evidence / "genesis_camera_corrected/metrics.json")
    targets = read_json(evidence / "dataset/phase_targets_report.json")

    base = paired["baseline"]
    phase = paired["phase_aware"]
    base_hash = base["checkpoint"]["tree_sha256"]
    phase_hash = phase["checkpoint"]["tree_sha256"]
    base_l1 = base_recon["aggregate_equal_role_stratified_samples"]["normalized_chunk_l1"]["mean"]
    phase_l1 = phase_recon["aggregate_equal_role_stratified_samples"]["normalized_chunk_l1"]["mean"]
    base_correction_l1 = base_recon["roles"]["correction"]["normalized_chunk_l1"]["mean"]
    phase_correction_l1 = phase_recon["roles"]["correction"]["normalized_chunk_l1"]["mean"]
    correction = targets["summary"]["roles"]["correction"]

    work = args.work_dir.resolve()
    if work.exists():
        shutil.rmtree(work)
    slides = work / "slides"
    audio = work / "audio"
    segments = work / "segments"
    for directory in (slides, audio, segments):
        directory.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "kicker": "AMD Radeon Hackathon 2026 · Track 3",
            "title": "Radeon OneLoop",
            "bullets": [
                "Single-Radeon phase-aware learning for a real bimanual handover",
                "Environment, training, inference, and evidence in one auditable loop",
            ],
            "metric": "ONE RADEON · ONE ROBOT LOOP",
            "narration": "Radeon OneLoop is Phi Media Lab's Track Three entry for a contact-rich bimanual handover. One low-cost arm presents a soft object, the second receives it, and the system places it into a target zone. We provide an auditable path from environment, through intervention-aware training, to real-time execution on one AMD Radeon.",
        },
        {
            "kicker": "The physical-AI task",
            "title": "Coordination at the handover boundary",
            "bullets": [
                "Two SO-101 arms, two RGB views, twelve joint and gripper commands",
                "The receiver must arrive before the giver releases support",
                "Prior 37 of 45 reviewed runs prove the inherited loop, not the new checkpoints",
            ],
            "metric": "30 Hz control contract",
            "narration": "The setup uses two SO-101 arms and two RGB cameras. Observations carry both images plus twelve joint and gripper values, and actions command both arms at thirty hertz. The receiver must establish support before release. A prior reviewed batch completed thirty-seven of forty-five handovers; that is historical evidence for the inherited loop, never a result of the new checkpoints.",
        },
        {
            "kicker": "Competition boundary",
            "title": "Exactly one formal accelerator",
            "bullets": [
                "One gfx1100 Radeon, ROCm 7.2.1, AMD PyTorch 2.9.1",
                "Genesis, both ACT training jobs, and ACT inference use that same device",
                "CPU handles decoding, cameras, robot I/O, watchdog, limits, and E-stop",
            ],
            "metric": "51.52 GB visible VRAM · 1 GPU",
            "narration": "Exactly one gfx eleven-hundred Radeon is exposed, with ROCm seven point two point one and AMD PyTorch two point nine point one. Genesis, both from-scratch ACT training jobs, and real-observation policy chunks use this device. The CPU performs decoding, robot input-output, validation, watchdog, limits, and emergency stop, never fallback inference. Results from our other AMD machines are excluded.",
        },
        {
            "kicker": "Reproducible environment",
            "title": "Dual-SO-101 Genesis on Radeon",
            "bullets": [
                "Hash-verified robot model and meshes",
                "Two 480 by 640 camera observations plus finite twelve-value state",
                "Scripted sweep is an interface test, not a handover success claim",
            ],
            "metric": f"p50 {genesis['step_ms']['p50']:.2f} ms · p99 {genesis['step_ms']['p99']:.2f} ms",
            "image_path": evidence / "genesis_camera_corrected/camera_pair.png",
            "narration": f"The Genesis environment verifies the official SO-101 assets, then builds two arms, a table, an object, and two cameras with the real pipeline's keys. The corrected formal run completed one thousand Radeon steps with two four-eighty by six-forty views and finite state. Median step time was {genesis['step_ms']['p50']:.2f} milliseconds and p ninety-nine was {genesis['step_ms']['p99']:.2f}, including capture steps. This validates interfaces, not a learned handover.",
        },
        {
            "kicker": "Reviewed intervention data",
            "title": "Short corrections carry dense signal",
            "bullets": [
                "124 real episodes and 178,465 train-only frames",
                "Correction weight 4.0; failed autonomous prefix weight 0.05",
                "Positive weights normalize to mean one; all other settings stay fixed",
            ],
            "metric": f"{correction['frames']:,} corrections → {100 * correction['gradient_mass_ratio']:.2f}% gradient mass",
            "image_path": figures / "phase_weighting.png",
            "narration": f"The immutable real dataset has one hundred twenty-four episodes and one hundred seventy-eight thousand four hundred sixty-five frames. Corrections are only {100 * correction['frames'] / 178465:.2f} percent of frames but concentrate around difficult transitions. We assign them weight four, failed prefixes point zero five, and other frames one. Normalization gives corrections {100 * correction['gradient_mass_ratio']:.2f} percent of gradient mass without duplicating data. Every other training setting stays identical.",
        },
        {
            "kicker": "Controlled experiment",
            "title": "Baseline versus phase-aware ACT",
            "bullets": [
                f"Both models: 10,000 updates, batch 16, seed 20260803",
                f"Wall time: baseline {base['elapsed_seconds']/60:.1f} min · phase-aware {phase['elapsed_seconds']/60:.1f} min",
                f"Terminal logged loss: {base['terminal_loss']:.4f} · {phase['terminal_loss']:.4f}",
            ],
            "metric": "Final-step checkpoint predeclared",
            "image_path": figures / "formal_training_loss.png",
            "narration": f"Both models start randomly with the same seed, batch size sixteen, ACT architecture, optimizer, data, and ten thousand updates. Baseline training took {base['elapsed_seconds']/60:.1f} minutes with terminal logged loss {base['terminal_loss']:.4f}; phase-aware training took {phase['elapsed_seconds']/60:.1f} minutes with loss {phase['terminal_loss']:.4f}. The final-step rule was declared in advance. Intermediate checkpoints were never searched, and both complete hashes appear next.",
        },
        {
            "kicker": "Immutable model artifacts",
            "title": "The selected checkpoints are content-addressed",
            "bullets": [
                "Each hash covers model weights, policy config, train config, and processors",
                "The ledger is path-ordered and independently reproducible",
            ],
            "metric": "SHA-256 over the complete artifact tree",
            "mono_lines": [f"baseline  {base_hash}", f"phase     {phase_hash}"],
            "narration": "These are the complete content hashes of the two final checkpoints. Each ledger covers weights, configurations, processors, and tensor state. Files are sorted by relative path before the digest, so reviewers can reproduce each identifier without trusting a filename or upload service. The report links both hashes to their training jobs and Radeon record.",
        },
        {
            "kicker": "Real-time policy path",
            "title": "Full chunks on Radeon, queued actions at the edge",
            "bullets": [
                f"Full 100-action chunk p50: {base_latency['chunk_generation']['p50_ms']:.2f} ms · {phase_latency['chunk_generation']['p50_ms']:.2f} ms",
                f"Correction-frame normalized chunk L1: {base_correction_l1:.4f} · {phase_correction_l1:.4f}",
                f"Stratified train-frame chunk L1: {base_l1:.4f} · {phase_l1:.4f}",
            ],
            "metric": "200 synchronized calls after warm-up",
            "image_path": figures / "formal_inference_latency.png",
            "narration": f"A real observation drives one-hundred-action chunk generation. After warm-up, two hundred synchronized calls give median latency {base_latency['chunk_generation']['p50_ms']:.2f} milliseconds for baseline and {phase_latency['chunk_generation']['p50_ms']:.2f} for phase-aware ACT. Queued actions avoid regeneration. On correction frames, normalized chunk L one is {base_correction_l1:.4f} versus {phase_correction_l1:.4f}; equal-role values are {base_l1:.4f} and {phase_l1:.4f}. These training-frame diagnostics are not task success.",
        },
        {
            "kicker": "Fail-closed deployment",
            "title": "Safety before robot I/O",
            "bullets": [
                "Observation and action sequence IDs must match and increase monotonically",
                "Timeouts, non-finite values, joint limits, and per-step deltas are enforced",
                "Any violation latches E-stop; software cannot silently re-arm",
            ],
            "metric": "CPU safety edge · no fallback inference",
            "narration": "Before robot input-output, the safety kernel validates sequence identifiers, observation correspondence, freshness, shape, finite values, joint limits, and maximum per-step motion. A stale, reordered, mismatched, or unsafe packet latches emergency stop. Recovery requires a new controller after physical reset, so software cannot silently re-arm. Manufacturer limits and a human physical-stop operator remain required.",
        },
        {
            "kicker": "Evidence, not storytelling",
            "title": "Every claim has a lineage",
            "bullets": [
                "GPU UID, source commit, config hash, dataset hash, seed, command, and raw logs",
                "One-second ROCm samples plus deterministic checkpoint-tree hashes",
                "No held-out data means no generalization or new task-success claim",
            ],
            "metric": "Failed and superseded runs remain visible",
            "mono_lines": [
                "$ ./ops/dispatch.sh radeon-c act_train configs/act_phase_aware.yaml ...",
                "INFO Loaded 178465 ACT-AWR weights: min=0.041529 mean=1 max=3.322282",
                "INFO Checkpoint policy after step 10000 · End of training",
            ],
            "narration": "Every formal GPU job takes an exclusive lock and records the Radeon UID, source, configuration, data, seed, command, environment, ROCm samples, raw logs, metrics, and terminal marker. Failed and superseded runs stay visible. With no held-out split and no calibrated sim-to-real benchmark, we make no generalization or new physical-success claim. That boundary is a result, not fine print.",
        },
        {
            "kicker": "Reproduce the loop",
            "title": "Open source, auditable, Radeon-native",
            "bullets": [
                "Hash-pinned environment and Genesis assets",
                "Immutable data builder, frozen pair, inference diagnostics, and safety tests",
                "github.com/phi-media-lab/radeon-oneloop · Apache-2.0",
            ],
            "metric": "./ops/validate_scaffold.sh",
            "narration": "The public repository contains pinned AMD setup, verified Genesis assets, the immutable data builder, frozen experiment pair, formal evidence, inference diagnostics, safety tests, and the English report. Raw robot video remains access-controlled, while its schema and hashes are public. Radeon OneLoop shows one Radeon carrying physical AI from environment, through learning, to real-time deployment, with honest claim boundaries. Thank you.",
        },
    ]

    segment_paths: list[Path] = []
    for index, spec in enumerate(specs, start=1):
        slide_path = slides / f"{index:02d}.png"
        segment_path = segments / f"{index:02d}.mp4"
        audio_path = audio / f"{index:02d}.aiff"
        slide(slide_path, **{key: value for key, value in spec.items() if key != "narration"})
        speech_segment(
            slide_path,
            spec["narration"],
            segment_path,
            voice=args.voice,
            rate=args.rate,
            audio_path=audio_path,
        )
        segment_paths.append(segment_path)

    concat = work / "concat.txt"
    concat.write_text(
        "".join(f"file '{path.as_posix()}'\n" for path in segment_paths),
        encoding="utf-8",
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )
    seconds = duration(output)
    if not 180 <= seconds <= 300:
        raise RuntimeError(f"video duration {seconds:.2f}s is outside the required 3-5 minutes")
    print(json.dumps({"output": str(output), "duration_seconds": seconds}, indent=2))


if __name__ == "__main__":
    main()
