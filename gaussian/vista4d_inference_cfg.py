#!/usr/bin/env python3
"""Run the official Vista4D CLI with an explicit, auditable CFG override."""

from __future__ import annotations

import functools
import math
import os
import runpy
from typing import Any, Callable


def install_cfg_scale_override(
    pipeline_class: type,
    cfg_scale: float,
) -> Callable[..., Any]:
    """Set CFG only when the official caller did not provide it explicitly."""

    if not math.isfinite(cfg_scale) or not 1.0 <= cfg_scale <= 10.0:
        raise ValueError("Vista4D CFG scale must be finite and in [1, 10]")
    original = pipeline_class.__call__

    @functools.wraps(original)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("cfg_scale", cfg_scale)
        return original(self, *args, **kwargs)

    pipeline_class.__call__ = wrapped
    return original


def main() -> None:
    cfg_scale = float(os.environ.get("ONELOOP_VISTA4D_CFG_SCALE", "5.0"))
    from diffsynth.pipelines.wan_video_vista4d import Vista4DPipeline

    install_cfg_scale_override(Vista4DPipeline, cfg_scale)
    runpy.run_module("scripts.inference.inference", run_name="__main__")


if __name__ == "__main__":
    main()
