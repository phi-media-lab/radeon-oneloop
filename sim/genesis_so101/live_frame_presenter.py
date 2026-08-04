"""Loopback-only HTTP presenter for a live RGB simulation frame."""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
import statistics
import threading
import time
from typing import Any
from urllib.parse import urlsplit

import numpy as np


@dataclass
class _PresenterState:
    title: str
    jpeg_quality: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    latest_jpeg: bytes | None = None
    frame_count: int = 0
    image_size_wh: tuple[int, int] | None = None
    publish_times_ms: list[float] = field(default_factory=list)
    page_requests: int = 0
    frame_requests: int = 0
    health_requests: int = 0


def _page(title: str) -> bytes:
    safe_title = escape(title)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{safe_title}</title>
  <style>
    html, body {{ margin: 0; min-height: 100%; background: #0b0e12; color: #eef3f8;
      font-family: system-ui, sans-serif; }}
    main {{ display: grid; place-items: center; padding: 18px; gap: 10px; }}
    h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
    .labels {{ width: min(96vw, 1280px); display: grid; grid-template-columns: 1fr 1fr;
      text-align: center; color: #9bd5ff; font-weight: 600; }}
    img {{ width: min(96vw, 1280px); height: auto; border: 1px solid #35404c;
      border-radius: 8px; background: #171b20; }}
    footer {{ color: #9ca8b5; font-size: 13px; text-align: center; }}
  </style>
</head>
<body><main>
  <h1>{safe_title}</h1>
  <div class="labels"><span>Front camera</span><span>Hand camera</span></div>
  <img id="feed" alt="Waiting for the first Gaussian composite frame">
  <footer>30k Gaussian appearance · Genesis depth/occlusion · non-authoritative renderer · physical output disabled</footer>
</main>
<script>
  const feed = document.getElementById('feed');
  function refresh() {{
    feed.src = '/frame.jpg?now=' + Date.now();
  }}
  feed.onload = () => setTimeout(refresh, 80);
  feed.onerror = () => setTimeout(refresh, 250);
  refresh();
</script></body></html>""".encode("utf-8")


def _handler(state: _PresenterState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
            route = urlsplit(self.path).path
            if route == "/":
                with state.lock:
                    state.page_requests += 1
                payload = _page(state.title)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif route == "/frame.jpg":
                with state.lock:
                    state.frame_requests += 1
                    payload = state.latest_jpeg
                if payload is None:
                    self.send_error(503, "first frame is not ready")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Cache-Control", "no-store, max-age=0")
            elif route == "/health.json":
                with state.lock:
                    state.health_requests += 1
                    health = {
                        "status": "ready",
                        "frames_published": state.frame_count,
                        "image_size_wh": list(state.image_size_wh) if state.image_size_wh else None,
                        "physical_output": False,
                    }
                payload = json.dumps(health, sort_keys=True).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
            else:
                self.send_error(404)
                return
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        def log_message(self, _format: str, *_args: Any) -> None:
            return

    return Handler


class LiveFrameHttpPresenter:
    """Publish the latest RGB frame on one loopback HTTP endpoint."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        title: str = "Radeon OneLoop Real2Sim",
        jpeg_quality: int = 90,
    ):
        if host not in {"127.0.0.1", "::1", "localhost"}:
            raise ValueError("live presenter must bind to a loopback host")
        if not 0 <= port <= 65535:
            raise ValueError("presenter port must be between 0 and 65535")
        if not 50 <= jpeg_quality <= 100:
            raise ValueError("jpeg quality must be between 50 and 100")
        self._state = _PresenterState(title=title, jpeg_quality=jpeg_quality)
        self._server = ThreadingHTTPServer((host, port), _handler(self._state))
        self._server.daemon_threads = True
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="oneloop-live-frame-presenter",
            daemon=True,
        )
        self._closed = False

    @property
    def url(self) -> str:
        host, port = self._server.server_address[:2]
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        return f"http://{display_host}:{port}/"

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("presenter is closed")
        self._thread.start()

    def publish(self, rgb_u8: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("presenter is closed")
        rgb = np.asarray(rgb_u8)
        if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
            raise ValueError("presenter frame must be uint8 HxWx3 RGB")
        from PIL import Image

        started = time.perf_counter()
        encoded = BytesIO()
        Image.fromarray(rgb).save(
            encoded,
            format="JPEG",
            quality=self._state.jpeg_quality,
            optimize=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        with self._state.lock:
            self._state.latest_jpeg = encoded.getvalue()
            self._state.frame_count += 1
            self._state.image_size_wh = (int(rgb.shape[1]), int(rgb.shape[0]))
            self._state.publish_times_ms.append(elapsed_ms)

    def metrics(self) -> dict[str, Any]:
        with self._state.lock:
            times = list(self._state.publish_times_ms)
            return {
                "enabled": True,
                "url": self.url,
                "frames_published": self._state.frame_count,
                "image_size_wh": (
                    list(self._state.image_size_wh)
                    if self._state.image_size_wh is not None
                    else None
                ),
                "requests": {
                    "page": self._state.page_requests,
                    "frame": self._state.frame_requests,
                    "health": self._state.health_requests,
                },
                "jpeg_encode_ms": {
                    "mean": statistics.fmean(times) if times else None,
                    "p95": float(np.percentile(times, 95)) if times else None,
                    "max": max(times) if times else None,
                },
                "bind_scope": "loopback_only",
                "physical_output": False,
            }

    def close(self) -> None:
        if self._closed:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5.0)
        self._closed = True

    def __enter__(self) -> "LiveFrameHttpPresenter":
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
