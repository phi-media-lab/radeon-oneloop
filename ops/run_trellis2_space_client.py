#!/usr/bin/env python3
"""Generate a textured GLB through the official TRELLIS.2 Gradio Space.

The script accepts either the Hugging Face Space id or its direct ``hf.space``
URL.  The latter is useful on compute nodes that can reach Space replicas but
cannot reach the Hugging Face Hub frontend.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from gradio_client import Client, handle_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--space",
        default="https://microsoft-trellis-2.hf.space",
        help="Official Space id or direct URL",
    )
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--resolution", choices=("512", "1024", "1536"), default="1024")
    parser.add_argument("--decimation-target", type=int, default=300_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--output-name", default="trellis2.glb")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    client = Client(args.space, verbose=True)
    client.predict(api_name="/start_session")
    preprocessed = client.predict(handle_file(str(source)), api_name="/preprocess_image")
    shutil.copy2(preprocessed, output_dir / "preprocessed.png")

    client.predict(
        handle_file(preprocessed),
        args.seed,
        args.resolution,
        7.5,
        0.7,
        12,
        5.0,
        7.5,
        0.5,
        12,
        3.0,
        1.0,
        0.0,
        12,
        3.0,
        api_name="/image_to_3d",
    )
    model_preview, model_download = client.predict(
        args.decimation_target,
        args.texture_size,
        api_name="/extract_glb",
    )
    model_path = model_download or model_preview
    target = output_dir / args.output_name
    shutil.copy2(model_path, target)
    print(target)


if __name__ == "__main__":
    main()
