#!/usr/bin/env python3
"""Render publication figures from the machine-readable formal evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


INK = "#172235"
MUTED = "#667085"
ACCENT = "#B91C3B"
NAVY = "#31587A"
GRID = "#D8DDE6"
PALE = "#F4F6F9"
WHITE = "#FFFFFF"
ROLE_COLORS = ["#31587A", "#7196B5", "#D8A0AF", "#B91C3B"]


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


def canvas(title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (1600, 900), WHITE)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 1600, 14), fill=ACCENT)
    draw.text((90, 55), title, fill=INK, font=font(44, bold=True))
    draw.text((90, 118), subtitle, fill=MUTED, font=font(22))
    return image, draw


def save(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def training_loss(summary: dict[str, Any], path: Path) -> None:
    image, draw = canvas(
        "Formal ACT training loss",
        "Same Radeon, dataset, architecture, seed, batch size, optimizer, and 10,000-step budget",
    )
    left, top, right, bottom = 150, 210, 1500, 760
    series = [
        ("Baseline ACT", ACCENT, summary["baseline"]["progress"]),
        ("Phase-aware ACT", NAVY, summary["phase_aware"]["progress"]),
    ]
    values = [float(point["loss"]) for _, _, points in series for point in points]
    minimum = max(min(values) * 0.8, 1e-4)
    maximum = max(values) * 1.15
    log_min, log_max = math.log10(minimum), math.log10(maximum)

    draw.rectangle((left, top, right, bottom), fill=PALE)
    for step in range(0, 10001, 2000):
        x = left + (right - left) * step / 10000
        draw.line((x, top, x, bottom), fill=GRID, width=2)
        label = f"{step // 1000}k" if step else "0"
        draw.text((x - 18, bottom + 16), label, fill=MUTED, font=font(18))
    tick_power = math.floor(log_min)
    while tick_power <= math.ceil(log_max):
        value = 10**tick_power
        if minimum <= value <= maximum:
            y = bottom - (bottom - top) * (math.log10(value) - log_min) / (log_max - log_min)
            draw.line((left, y, right, y), fill=GRID, width=2)
            draw.text((58, y - 11), f"{value:g}", fill=MUTED, font=font(18))
        tick_power += 1
    for name, color, points in series:
        coordinates = []
        for point in points:
            x = left + (right - left) * float(point["step"]) / 10000
            y = bottom - (bottom - top) * (
                math.log10(float(point["loss"])) - log_min
            ) / (log_max - log_min)
            coordinates.append((x, y))
        draw.line(coordinates, fill=color, width=6, joint="curve")
    draw.text((left, 790), "Optimizer updates", fill=INK, font=font(20, bold=True))
    draw.text((25, 235), "Loss (log scale)", fill=INK, font=font(18, bold=True))
    for index, (name, color, points) in enumerate(series):
        x = 980 + index * 260
        draw.line((x, 175, x + 55, 175), fill=color, width=7)
        draw.text((x + 66, 161), name, fill=INK, font=font(18, bold=True))
        draw.text(
            (x + 66, 184),
            f"terminal {float(points[-1]['loss']):.4f}",
            fill=MUTED,
            font=font(16),
        )
    save(image, path)


def latency_figure(baseline: dict[str, Any], phase: dict[str, Any], path: Path) -> None:
    image, draw = canvas(
        "Single-Radeon ACT inference latency",
        "200 synchronized full-chunk calls after warm-up; real dataset observation; 100-action horizon",
    )
    groups = [
        ("Full chunk p50", baseline["chunk_generation"]["p50_ms"], phase["chunk_generation"]["p50_ms"]),
        ("Full chunk p95", baseline["chunk_generation"]["p95_ms"], phase["chunk_generation"]["p95_ms"]),
        ("Queued p50", baseline["queued_action_dispatch"]["p50_ms"], phase["queued_action_dispatch"]["p50_ms"]),
    ]
    left, top, right, bottom = 150, 220, 1500, 760
    maximum = max(value for _, base, phased in groups for value in (base, phased)) * 1.2
    draw.rectangle((left, top, right, bottom), fill=PALE)
    for index in range(6):
        value = maximum * index / 5
        y = bottom - (bottom - top) * index / 5
        draw.line((left, y, right, y), fill=GRID, width=2)
        draw.text((72, y - 10), f"{value:.0f}", fill=MUTED, font=font(18))
    group_width = (right - left) / len(groups)
    for group_index, (label, base, phased) in enumerate(groups):
        center = left + group_width * (group_index + 0.5)
        for offset, value, color, name in (
            (-70, base, ACCENT, "Baseline"),
            (10, phased, NAVY, "Phase-aware"),
        ):
            x0, x1 = center + offset, center + offset + 60
            y0 = bottom - (bottom - top) * float(value) / maximum
            draw.rounded_rectangle((x0, y0, x1, bottom), radius=7, fill=color)
            draw.text((x0 - 5, y0 - 31), f"{float(value):.2f}", fill=INK, font=font(17, bold=True))
        width = draw.textlength(label, font=font(19, bold=True))
        draw.text((center - width / 2, bottom + 22), label, fill=INK, font=font(19, bold=True))
    draw.text((35, 235), "ms", fill=INK, font=font(20, bold=True))
    draw.rectangle((1040, 164, 1070, 190), fill=ACCENT)
    draw.text((1082, 165), "Baseline ACT", fill=INK, font=font(18, bold=True))
    draw.rectangle((1260, 164, 1290, 190), fill=NAVY)
    draw.text((1302, 165), "Phase-aware ACT", fill=INK, font=font(18, bold=True))
    save(image, path)


def weighting_figure(report: dict[str, Any], path: Path) -> None:
    image, draw = canvas(
        "Where phase-aware ACT spends its gradient",
        "Correction frames are 8.91% of data but receive 29.61% of normalized gradient mass",
    )
    roles = report["summary"]["roles"]
    order = ["bc_demonstration", "success_policy", "failed_policy_prefix", "correction"]
    labels = ["Human demonstrations", "Successful policy", "Failed policy prefix", "Human correction"]
    frames_total = sum(int(roles[role]["frames"]) for role in order)
    frame_shares = [int(roles[role]["frames"]) / frames_total for role in order]
    mass_shares = [float(roles[role]["gradient_mass_ratio"]) for role in order]

    left, right = 320, 1480
    for y, title, shares in (
        (290, "Share of training frames", frame_shares),
        (510, "Share of gradient mass", mass_shares),
    ):
        draw.text((90, y - 70), title, fill=INK, font=font(27, bold=True))
        x = left
        for share, color in zip(shares, ROLE_COLORS, strict=True):
            width = (right - left) * share
            draw.rectangle((x, y, x + width, y + 110), fill=color)
            if width > 90:
                draw.text((x + 12, y + 38), f"{share * 100:.1f}%", fill=WHITE, font=font(22, bold=True))
            x += width
    for index, (label, color) in enumerate(zip(labels, ROLE_COLORS, strict=True)):
        x = 110 + (index % 2) * 560
        y = 700 + (index // 2) * 62
        draw.rectangle((x, y, x + 34, y + 34), fill=color)
        draw.text((x + 48, y + 4), label, fill=INK, font=font(21, bold=True))
    save(image, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-root", type=Path, default=Path("artifacts/formal"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()
    root = args.evidence_root.resolve()
    output = args.output_dir.resolve()
    paired = json.loads((root / "paired_training_summary.json").read_text(encoding="utf-8"))
    baseline_latency = json.loads((root / "baseline_latency/metrics.json").read_text(encoding="utf-8"))
    phase_latency = json.loads((root / "phase_latency/metrics.json").read_text(encoding="utf-8"))
    targets = json.loads((root / "dataset/phase_targets_report.json").read_text(encoding="utf-8"))
    training_loss(paired, output / "formal_training_loss.png")
    latency_figure(baseline_latency, phase_latency, output / "formal_inference_latency.png")
    weighting_figure(targets, output / "phase_weighting.png")
    for path in sorted(output.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
