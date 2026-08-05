"""Rigid Gaussian appearance binding for the Genesis handover object.

The control path never depends on this module.  It converts the current
Genesis object/camera transforms into the OpenCV world-to-camera convention
used by the pinned VkSplat renderer and returns premultiplied RGB plus alpha.
Callers can therefore composite the appearance over a Genesis raster frame or
drop to the ordinary debug mesh when the optional renderer is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from gaussian.provenance_quarantine import (
    QuarantinedLineageError,
    assert_not_quarantined,
)


VKSPLAT_COMMIT = "e26c254938c81ff85998cd357a9e005e255d9b03"
OBSERVED_CORE_PLY_SHA256 = (
    "7f01c1e6d8253d7f15162e2cb51e18845676fa1015983266b7d356d9b21aa706"
)
OBSERVED_CORE_CAMERAS_SHA256 = (
    "050891df1cfc5ef33070f7ab6becdd168267e5951143523519601f38963cbc26"
)
OBSERVED_CORE_PROVENANCE_SHA256 = (
    "c2c95e0f7b5e51ffb5e9aeda7ecdc45229d75b4adf8683b09e6319c638795528"
)
OBSERVED_CORE_GAUSSIANS = 30_000

# A point expressed in an OpenCV camera frame (x right, y down, z forward)
# has these coordinates in Genesis' OpenGL camera frame (x right, y up,
# z backward).  Genesis Camera.get_transform() is camera-to-world.
OPENGL_CAMERA_FROM_OPENCV = np.diag((1.0, -1.0, -1.0, 1.0))


class GaussianAppearanceError(RuntimeError):
    """Raised for an invalid asset or a recoverable renderer failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array(value: Any, *, dtype: np.dtype[Any] = np.dtype(np.float64)) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def validate_rigid_transform(value: Any, *, name: str = "transform") -> np.ndarray:
    """Return a finite SE(3) matrix after rejecting scale/reflection/shear."""

    transform = _array(value)
    if transform.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {transform.shape}")
    if not np.isfinite(transform).all():
        raise ValueError(f"{name} must be finite")
    if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2.0e-5):
        raise ValueError(f"{name} rotation contains scale or shear")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=2.0e-5):
        raise ValueError(f"{name} rotation is not right-handed")
    return transform


def transform_from_pos_quat_wxyz(position: Any, quaternion: Any) -> np.ndarray:
    """Build an object-to-world transform from Genesis' WXYZ quaternion."""

    pos = _array(position).reshape(-1)
    quat = _array(quaternion).reshape(-1)
    if pos.shape != (3,) or quat.shape != (4,):
        raise ValueError("position and WXYZ quaternion must have shapes (3,) and (4,)")
    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
        raise ValueError("position and quaternion must be finite")
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        raise ValueError("quaternion norm is zero")
    w, x, y, z = quat / norm
    rotation = np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = pos
    return validate_rigid_transform(transform, name="T_world_object")


def object_to_camera_opencv(
    world_from_camera_opengl: Any,
    world_from_object: Any,
) -> np.ndarray:
    """Return the canonical-object to OpenCV-camera transform for VkSplat.

    Genesis exposes ``T_world_camera_gl``.  Right multiplication by
    ``diag(1,-1,-1,1)`` changes only the camera basis and gives
    ``T_world_camera_cv``.  VkSplat expects its inverse composed with the
    rigid object's canonical-to-world pose::

        T_camera_object_cv = inverse(T_world_camera_gl * C_gl_from_cv)
                             * T_world_object
    """

    world_camera_gl = validate_rigid_transform(
        world_from_camera_opengl, name="T_world_camera_opengl"
    )
    world_object = validate_rigid_transform(world_from_object, name="T_world_object")
    world_camera_cv = world_camera_gl @ OPENGL_CAMERA_FROM_OPENCV
    result = np.linalg.solve(world_camera_cv, world_object)
    return validate_rigid_transform(result, name="T_camera_object_opencv")


@dataclass(frozen=True)
class PinholeCamera:
    width: int
    height: int
    intrinsic_3x3: np.ndarray
    camera_from_object_opencv_4x4: np.ndarray

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera dimensions must be positive")
        intrinsic = _array(self.intrinsic_3x3)
        if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
            raise ValueError("intrinsic_3x3 must be a finite 3x3 matrix")
        if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), atol=1.0e-6):
            raise ValueError("intrinsic_3x3 has an invalid homogeneous row")
        transform = validate_rigid_transform(
            self.camera_from_object_opencv_4x4,
            name="T_camera_object_opencv",
        )
        object.__setattr__(self, "intrinsic_3x3", intrinsic)
        object.__setattr__(self, "camera_from_object_opencv_4x4", transform)

    @classmethod
    def from_genesis(cls, camera: Any, object_entity: Any) -> "PinholeCamera":
        world_camera = _array(camera.get_transform())
        world_object = transform_from_pos_quat_wxyz(
            object_entity.get_pos(), object_entity.get_quat()
        )
        width, height = (int(value) for value in camera.res)
        intrinsic = np.asarray(
            ((float(camera.f), 0.0, float(camera.cx)),
             (0.0, float(camera.f), float(camera.cy)),
             (0.0, 0.0, 1.0)),
            dtype=np.float64,
        )
        return cls(
            width=width,
            height=height,
            intrinsic_3x3=intrinsic,
            camera_from_object_opencv_4x4=object_to_camera_opencv(
                world_camera, world_object
            ),
        )


@dataclass(frozen=True)
class ObservedCoreAsset:
    ply_path: Path
    cameras_path: Path
    provenance_path: Path
    expected_ply_sha256: str = OBSERVED_CORE_PLY_SHA256
    expected_cameras_sha256: str = OBSERVED_CORE_CAMERAS_SHA256
    expected_provenance_sha256: str = OBSERVED_CORE_PROVENANCE_SHA256
    expected_gaussians: int = OBSERVED_CORE_GAUSSIANS
    expected_formal: bool = True
    expected_provenance_schema: str = "radeon_oneloop.observed_core_canonicalization.v1"
    expected_provenance_class: str = "observed_core_candidate"
    required_observed_only_training: bool = True

    def validate(self) -> dict[str, Any]:
        paths = (self.ply_path, self.cameras_path, self.provenance_path)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise GaussianAppearanceError(f"observed-core files are missing: {missing}")
        hashes = {
            "ply": sha256_file(self.ply_path),
            "cameras": sha256_file(self.cameras_path),
            "provenance": sha256_file(self.provenance_path),
        }
        expected = {
            "ply": self.expected_ply_sha256,
            "cameras": self.expected_cameras_sha256,
            "provenance": self.expected_provenance_sha256,
        }
        mismatches = {
            name: {"expected": expected[name], "actual": value}
            for name, value in hashes.items()
            if value != expected[name]
        }
        if mismatches:
            raise GaussianAppearanceError(f"observed-core hash mismatch: {mismatches}")
        provenance = json.loads(self.provenance_path.read_text(encoding="utf-8"))
        if provenance.get("schema_version") != self.expected_provenance_schema:
            raise GaussianAppearanceError("unsupported observed-core provenance schema")
        if provenance.get("output_ply_sha256") != hashes["ply"]:
            raise GaussianAppearanceError("provenance does not bind the canonical PLY")
        if provenance.get("gaussian_count") != self.expected_gaussians:
            raise GaussianAppearanceError("unexpected observed-core Gaussian count")
        if provenance.get("observed_only_training") is not self.required_observed_only_training:
            raise GaussianAppearanceError("appearance training class does not match the asset role")
        if provenance.get("provenance_class") != self.expected_provenance_class:
            raise GaussianAppearanceError("appearance provenance class does not match the asset role")
        if provenance.get("formal") is not self.expected_formal:
            raise GaussianAppearanceError(
                "observed-core formal status does not match the pinned asset class"
            )
        cameras = json.loads(self.cameras_path.read_text(encoding="utf-8"))
        if cameras.get("camera_model") != "PINHOLE_OPENCV":
            raise GaussianAppearanceError("canonical cameras are not PINHOLE_OPENCV")
        return {
            "hashes": hashes,
            "gaussian_count": self.expected_gaussians,
            "formal": self.expected_formal,
            "provenance_class": provenance.get("provenance_class"),
            "camera_count": len(cameras.get("cameras", [])),
        }


@dataclass(frozen=True)
class CapabilityProbe:
    available: bool
    backend: str
    reason: str
    details: dict[str, Any]


def probe_nyx() -> CapabilityProbe:
    """Probe Nyx without making it a dependency of the control process."""

    try:
        spec = importlib.util.find_spec("genesis.renderers.nyx_renderer")
    except Exception as error:  # optional environment may not import torch
        return CapabilityProbe(False, "genesis_nyx", f"probe failed: {error}", {})
    if spec is None:
        return CapabilityProbe(
            False,
            "genesis_nyx",
            "genesis.renderers.nyx_renderer is not installed",
            {},
        )
    try:
        module = importlib.import_module("genesis.renderers.nyx_renderer")
    except Exception as error:
        return CapabilityProbe(False, "genesis_nyx", f"import failed: {error}", {})
    has_light_field = hasattr(module, "LightFieldAsset")
    return CapabilityProbe(
        has_light_field,
        "genesis_nyx",
        "LightFieldAsset available" if has_light_field else "LightFieldAsset missing",
        {"module": str(spec.origin)},
    )


@dataclass(frozen=True)
class AppearanceFrame:
    premultiplied_rgb: np.ndarray
    alpha: np.ndarray
    render_ms: float
    backend: str
    depth_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        rgb = np.asarray(self.premultiplied_rgb, dtype=np.float32)
        alpha = np.asarray(self.alpha, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("premultiplied_rgb must have shape HxWx3")
        if alpha.shape == rgb.shape[:2]:
            alpha = alpha[..., None]
        if alpha.shape != (*rgb.shape[:2], 1):
            raise ValueError("alpha must have shape HxW or HxWx1")
        if not np.isfinite(rgb).all() or not np.isfinite(alpha).all():
            raise ValueError("appearance output must be finite")
        depth = self.depth_m
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
            if depth.shape != rgb.shape[:2]:
                raise ValueError("depth_m must have shape HxW")
            if np.any(np.isnan(depth)) or np.any(depth <= 0.0):
                finite = np.isfinite(depth)
                if np.any(depth[finite] <= 0.0) or np.any(np.isnan(depth)):
                    raise ValueError("finite appearance depth values must be positive")
            object.__setattr__(self, "depth_m", depth)
        object.__setattr__(self, "premultiplied_rgb", np.clip(rgb, 0.0, 1.0))
        object.__setattr__(self, "alpha", np.clip(alpha, 0.0, 1.0))


def approximate_gaussian_depth(
    projected_xy: Any,
    center_depth_m: Any,
    radii_px: Any,
    alpha: Any,
    *,
    fill_radius_px: int = 3,
    alpha_threshold: float = 1.0e-3,
) -> np.ndarray:
    """Approximate the front Gaussian surface from projected center depths.

    VkSplat exposes its projected Gaussian centers and camera-space depths but
    not an accumulated per-pixel depth buffer.  The trained surface is dense,
    so a conservative small-radius nearest-depth fill gives an object z-buffer
    without consulting the procedural collision proxy.
    """

    xy = np.asarray(projected_xy, dtype=np.float32)
    depth = np.asarray(center_depth_m, dtype=np.float32).reshape(-1)
    radii = np.asarray(radii_px).reshape(-1)
    raw_alpha = np.asarray(alpha, dtype=np.float32)
    if raw_alpha.ndim == 3 and raw_alpha.shape[-1] == 1:
        raw_alpha = raw_alpha[..., 0]
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("projected_xy must have shape Nx2")
    if len(xy) != len(depth) or len(xy) != len(radii):
        raise ValueError("projected centers, depths, and radii must have equal lengths")
    if raw_alpha.ndim != 2:
        raise ValueError("alpha must have shape HxW or HxWx1")
    if fill_radius_px < 0 or fill_radius_px > 8:
        raise ValueError("fill_radius_px must be between zero and eight")
    if not 0.0 < alpha_threshold < 1.0:
        raise ValueError("alpha_threshold must be in (0, 1)")

    height, width = raw_alpha.shape
    px = np.rint(xy[:, 0]).astype(np.int64)
    py = np.rint(xy[:, 1]).astype(np.int64)
    valid = (
        np.all(np.isfinite(xy), axis=1)
        & np.isfinite(depth)
        & (depth > 0.0)
        & (radii > 0)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    inverse_depth = np.zeros((height, width), dtype=np.float32)
    good = np.flatnonzero(valid)
    if len(good):
        np.maximum.at(inverse_depth, (py[good], px[good]), 1.0 / depth[good])
    if fill_radius_px:
        padded = np.pad(inverse_depth, fill_radius_px, mode="constant")
        filled = np.zeros_like(inverse_depth)
        for dy in range(2 * fill_radius_px + 1):
            for dx in range(2 * fill_radius_px + 1):
                np.maximum(
                    filled,
                    padded[dy : dy + height, dx : dx + width],
                    out=filled,
                )
        inverse_depth = filled
    support = raw_alpha > alpha_threshold
    result = np.full((height, width), np.inf, dtype=np.float32)
    resolved = support & (inverse_depth > 0.0)
    result[resolved] = 1.0 / inverse_depth[resolved]
    return result


class AppearanceRenderer(Protocol):
    backend: str

    def render(self, camera: PinholeCamera) -> AppearanceFrame: ...

    def close(self) -> None: ...


class VkSplatAppearanceRenderer:
    """Load one canonical PLY and render per-frame rigid camera transforms."""

    backend = "vksplat_radv"

    def __init__(
        self,
        asset: ObservedCoreAsset,
        vksplat_root: Path,
        *,
        active_sh_degree: int = 0,
        device_index: int = -1,
    ):
        if active_sh_degree not in range(4):
            raise ValueError("active_sh_degree must be between zero and three")
        self.asset_audit = asset.validate()
        self.asset = asset
        self.vksplat_root = vksplat_root.resolve()
        self.active_sh_degree = active_sh_degree
        self._lock = threading.Lock()
        self._closed = False

        build_dir = self.vksplat_root / "build"
        package_dir = self.vksplat_root / "vksplat"
        for path in (build_dir, package_dir):
            if path.is_dir() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
        shader_dir = package_dir / "shader"
        if not shader_dir.is_dir():
            raise GaussianAppearanceError(f"VkSplat shader directory is missing: {shader_dir}")
        try:
            vksplat = importlib.import_module("vksplat")
        except Exception as error:
            raise GaussianAppearanceError(f"VkSplat import failed: {error}") from error

        from gaussian.vksplat_render_ply import read_3dgs_ply

        gaussians = read_3dgs_ply(asset.ply_path)
        if len(gaussians["xyz"]) != asset.expected_gaussians:
            raise GaussianAppearanceError("decoded Gaussian count does not match provenance")
        self._module = vksplat.VkSplat()
        try:
            self._module.initialize(str(shader_dir.resolve()) + "//", device_index)
            self._module.set_gauss_params(
                gaussians["xyz"],
                gaussians["rotations"],
                gaussians["scales"],
                gaussians["opacities"],
                gaussians["sh"],
            )
        except Exception:
            try:
                self._module.cleanup()
            finally:
                self._closed = True
            raise

    def render(self, camera: PinholeCamera) -> AppearanceFrame:
        if self._closed:
            raise GaussianAppearanceError("VkSplat renderer is closed")
        intrinsic = camera.intrinsic_3x3
        started = time.perf_counter()
        with self._lock:
            self._module.set_uniforms(
                self.active_sh_degree,
                np.asarray(camera.camera_from_object_opencv_4x4, dtype=np.float32),
                camera.height,
                camera.width,
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
                False,
            )
            self._module.forward()
            pixel_state = np.asarray(self._module.pixel_state, dtype=np.float32).copy()
            projected_xy = np.asarray(self._module.xy_vs, dtype=np.float32).copy()
            center_depth_m = np.asarray(self._module.depths, dtype=np.float32).copy()
            radii_px = np.asarray(self._module.radii).copy()
        expected_shape = (camera.height, camera.width, 4)
        if pixel_state.shape != expected_shape:
            raise GaussianAppearanceError(
                f"unexpected VkSplat output {pixel_state.shape}; expected {expected_shape}"
            )
        transmittance = np.clip(pixel_state[..., 3:4], 0.0, 1.0)
        alpha = 1.0 - transmittance
        depth_m = approximate_gaussian_depth(
            projected_xy, center_depth_m, radii_px, alpha
        )
        render_ms = (time.perf_counter() - started) * 1000.0
        return AppearanceFrame(
            premultiplied_rgb=pixel_state[..., :3],
            alpha=alpha,
            render_ms=render_ms,
            backend=self.backend,
            depth_m=depth_m,
        )

    def memory_usage(self) -> dict[str, int]:
        return {
            "current_bytes": int(self._module.get_vram_usage()),
            "peak_bytes": int(self._module.get_peak_vram_usage()),
        }

    def close(self) -> None:
        if self._closed:
            return
        with self._lock:
            self._module.cleanup()
            self._closed = True

    def __enter__(self) -> "VkSplatAppearanceRenderer":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


@dataclass(frozen=True)
class BindingResult:
    frame: AppearanceFrame | None
    fallback: str
    error: str | None


class SafeAppearanceBinding:
    """Latch to the Genesis debug mesh after any recoverable render failure."""

    def __init__(
        self,
        renderer: AppearanceRenderer | None,
        *,
        initialization_error: str | None = None,
        fallback: str = "genesis_debug_mesh",
    ):
        self.renderer = renderer
        self.fallback = fallback
        self.initialization_error = initialization_error
        self.latched_error = initialization_error
        self.attempts = 0
        self.successes = 0
        self.failures = int(initialization_error is not None)

    @classmethod
    def create(
        cls,
        factory: Callable[[], AppearanceRenderer],
        *,
        fallback: str = "genesis_debug_mesh",
    ) -> "SafeAppearanceBinding":
        try:
            return cls(factory(), fallback=fallback)
        except Exception as error:
            return cls(None, initialization_error=f"{type(error).__name__}: {error}", fallback=fallback)

    def render(self, camera: PinholeCamera) -> BindingResult:
        if self.renderer is None or self.latched_error is not None:
            return BindingResult(None, self.fallback, self.latched_error)
        self.attempts += 1
        try:
            frame = self.renderer.render(camera)
        except Exception as error:
            self.failures += 1
            self.latched_error = f"{type(error).__name__}: {error}"
            try:
                self.renderer.close()
            except Exception:
                pass
            return BindingResult(None, self.fallback, self.latched_error)
        self.successes += 1
        return BindingResult(frame, self.fallback, None)

    def render_from_genesis(self, camera: Any, object_entity: Any) -> BindingResult:
        try:
            model = PinholeCamera.from_genesis(camera, object_entity)
        except Exception as error:
            self.failures += 1
            self.latched_error = f"{type(error).__name__}: {error}"
            return BindingResult(None, self.fallback, self.latched_error)
        return self.render(model)

    def metrics(self) -> dict[str, Any]:
        return {
            "backend": getattr(self.renderer, "backend", None),
            "fallback": self.fallback,
            "attempts": self.attempts,
            "successes": self.successes,
            "failures": self.failures,
            "latched_error": self.latched_error,
        }

    def close(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass


@dataclass(frozen=True)
class CompositeResult:
    rgb_u8: np.ndarray
    effective_alpha: np.ndarray
    visible_proxy_fraction: float
    gaussian_alpha_clipped_fraction: float
    compositor: str = "proxy_matte"


def composite_with_proxy_depth(
    genesis_rgb: Any,
    genesis_depth_m: Any,
    visible_object_mask: Any,
    appearance: AppearanceFrame,
) -> CompositeResult:
    """Composite Gaussian color only where the front-most proxy is visible.

    The pinned VkSplat output has premultiplied RGB and transmittance but no
    per-pixel depth.  Genesis' aligned collision/visual proxy therefore acts as
    the conservative object-depth matte.  A gripper or table pixel in front of
    the proxy is absent from ``visible_object_mask`` and remains untouched.
    """

    raw_rgb = np.asarray(genesis_rgb)
    if raw_rgb.ndim != 3 or raw_rgb.shape[-1] != 3:
        raise ValueError("genesis_rgb must have shape HxWx3")
    if np.issubdtype(raw_rgb.dtype, np.integer):
        base = raw_rgb.astype(np.float32) / np.iinfo(raw_rgb.dtype).max
    else:
        base = raw_rgb.astype(np.float32)
        if float(np.nanmax(base, initial=0.0)) > 1.0 + 1.0e-6:
            base /= 255.0
    depth = np.asarray(genesis_depth_m, dtype=np.float32)
    mask = np.asarray(visible_object_mask, dtype=bool)
    shape = raw_rgb.shape[:2]
    if depth.shape != shape or mask.shape != shape:
        raise ValueError("depth and object mask must match the RGB height/width")
    if appearance.premultiplied_rgb.shape[:2] != shape:
        raise ValueError("appearance and Genesis frame dimensions differ")

    visible_proxy = mask & np.isfinite(depth) & (depth > 0.0)
    matte = visible_proxy[..., None].astype(np.float32)
    effective_alpha = appearance.alpha * matte
    premultiplied = appearance.premultiplied_rgb * matte
    composite = premultiplied + base * (1.0 - effective_alpha)
    gaussian_support = appearance.alpha[..., 0] > 1.0e-3
    clipped = gaussian_support & ~visible_proxy
    support_count = int(np.count_nonzero(gaussian_support))
    return CompositeResult(
        rgb_u8=np.round(np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8),
        effective_alpha=effective_alpha,
        visible_proxy_fraction=float(np.mean(visible_proxy)),
        gaussian_alpha_clipped_fraction=(
            float(np.count_nonzero(clipped) / support_count) if support_count else 0.0
        ),
        compositor="proxy_matte",
    )


def composite_with_gaussian_depth(
    genesis_rgb: Any,
    genesis_depth_m: Any,
    appearance: AppearanceFrame,
    *,
    depth_tolerance_m: float = 0.004,
) -> CompositeResult:
    """Composite a self-matted Gaussian against the object-free Genesis depth."""

    raw_rgb = np.asarray(genesis_rgb)
    if raw_rgb.ndim != 3 or raw_rgb.shape[-1] != 3:
        raise ValueError("genesis_rgb must have shape HxWx3")
    if np.issubdtype(raw_rgb.dtype, np.integer):
        base = raw_rgb.astype(np.float32) / np.iinfo(raw_rgb.dtype).max
    else:
        base = raw_rgb.astype(np.float32)
        if float(np.nanmax(base, initial=0.0)) > 1.0 + 1.0e-6:
            base /= 255.0
    scene_depth = np.asarray(genesis_depth_m, dtype=np.float32)
    shape = raw_rgb.shape[:2]
    if scene_depth.shape != shape:
        raise ValueError("Genesis depth must match the RGB height/width")
    if appearance.premultiplied_rgb.shape[:2] != shape:
        raise ValueError("appearance and Genesis frame dimensions differ")
    if appearance.depth_m is None:
        raise ValueError("Gaussian self-depth is required for proxy-free compositing")
    gaussian_depth = np.asarray(appearance.depth_m, dtype=np.float32)
    if gaussian_depth.shape != shape:
        raise ValueError("Gaussian depth must match the RGB height/width")
    if not np.isfinite(depth_tolerance_m) or depth_tolerance_m < 0.0:
        raise ValueError("depth tolerance must be finite and non-negative")

    support = appearance.alpha[..., 0] > 1.0e-3
    gaussian_valid = np.isfinite(gaussian_depth) & (gaussian_depth > 0.0)
    scene_valid = np.isfinite(scene_depth) & (scene_depth > 0.0)
    visible = (
        support
        & gaussian_valid
        & (~scene_valid | (gaussian_depth <= scene_depth + depth_tolerance_m))
    )
    matte = visible[..., None].astype(np.float32)
    effective_alpha = appearance.alpha * matte
    premultiplied = appearance.premultiplied_rgb * matte
    composite = premultiplied + base * (1.0 - effective_alpha)
    support_count = int(np.count_nonzero(support))
    clipped = support & ~visible
    return CompositeResult(
        rgb_u8=np.round(np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8),
        effective_alpha=effective_alpha,
        visible_proxy_fraction=(
            float(np.count_nonzero(visible) / support_count) if support_count else 0.0
        ),
        gaussian_alpha_clipped_fraction=(
            float(np.count_nonzero(clipped) / support_count) if support_count else 0.0
        ),
        compositor="gaussian_self_depth",
    )


def entity_segmentation_index(scene: Any, entity: Any) -> int:
    """Resolve the single segmentation index of a one-link entity."""

    entity_index = int(entity.idx)
    matches = [
        int(segmentation_index)
        for segmentation_index, key in scene.segmentation_idx_dict.items()
        if key == entity_index
        or (isinstance(key, tuple) and len(key) >= 1 and key[0] == entity_index)
    ]
    if len(matches) != 1:
        raise GaussianAppearanceError(
            f"expected one segmentation index for entity {entity_index}, got {matches}"
        )
    return matches[0]


def link_segmentation_index(scene: Any, entity: Any, link: Any) -> int:
    """Resolve one link-level Genesis segmentation index after scene build."""

    entity_index = int(entity.idx)
    link_index = int(link.idx)
    matches = [
        int(segmentation_index)
        for segmentation_index, key in scene.segmentation_idx_dict.items()
        if isinstance(key, tuple)
        and len(key) >= 2
        and key[0] == entity_index
        and key[1] == link_index
    ]
    if len(matches) != 1:
        raise GaussianAppearanceError(
            "expected one segmentation index for "
            f"entity/link {entity_index}/{link_index}, got {matches}"
        )
    return matches[0]


def observed_core_asset(root: Path) -> ObservedCoreAsset:
    root = root.resolve()
    return ObservedCoreAsset(
        ply_path=root / "appearance_observed_canonical.ply",
        cameras_path=root / "cameras_observed.json",
        provenance_path=root / "provenance.json",
    )


def nonformal_candidate_asset(root: Path) -> ObservedCoreAsset:
    """Load a self-bound candidate without weakening the pinned formal asset.

    Candidate use must be an explicit caller choice.  The three local hashes
    bind the exact files for the current nonformal run, while provenance keeps
    the asset out of held-out-real and formal evidence claims.
    """

    resolved = root.resolve()
    ply = resolved / "appearance_observed_canonical.ply"
    cameras = resolved / "cameras_observed.json"
    provenance_path = resolved / "provenance.json"
    missing = [
        str(path)
        for path in (ply, cameras, provenance_path)
        if not path.is_file()
    ]
    if missing:
        raise GaussianAppearanceError(f"candidate files are missing: {missing}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("formal") is not False:
        raise GaussianAppearanceError("candidate requires formal=false provenance")
    if provenance.get("eligible_for_heldout_real_metrics") is not False:
        raise GaussianAppearanceError(
            "candidate must remain ineligible for held-out real metrics"
        )
    gaussian_count = provenance.get("gaussian_count")
    if not isinstance(gaussian_count, int) or gaussian_count <= 0:
        raise GaussianAppearanceError("candidate Gaussian count must be positive")
    return ObservedCoreAsset(
        ply_path=ply,
        cameras_path=cameras,
        provenance_path=provenance_path,
        expected_ply_sha256=sha256_file(ply),
        expected_cameras_sha256=sha256_file(cameras),
        expected_provenance_sha256=sha256_file(provenance_path),
        expected_gaussians=gaussian_count,
        expected_formal=False,
    )


def layered_preview_asset(root: Path) -> ObservedCoreAsset:
    """Load an explicit generated-fill preview without relabeling it observed-only."""

    resolved = root.resolve()
    ply = resolved / "appearance_fused_preview.ply"
    cameras = resolved / "cameras_observed.json"
    provenance_path = resolved / "appearance_fused_preview.provenance.json"
    missing = [str(path) for path in (ply, cameras, provenance_path) if not path.is_file()]
    if missing:
        raise GaussianAppearanceError(f"layered-preview files are missing: {missing}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if provenance.get("formal") is not False:
        raise GaussianAppearanceError("layered preview requires formal=false provenance")
    if provenance.get("eligible_for_heldout_real_metrics") is not False:
        raise GaussianAppearanceError("layered preview cannot be held-out-real evidence")
    gaussian_count = provenance.get("gaussian_count")
    if not isinstance(gaussian_count, int) or gaussian_count <= 0:
        raise GaussianAppearanceError("layered-preview Gaussian count must be positive")
    return ObservedCoreAsset(
        ply_path=ply,
        cameras_path=cameras,
        provenance_path=provenance_path,
        expected_ply_sha256=sha256_file(ply),
        expected_cameras_sha256=sha256_file(cameras),
        expected_provenance_sha256=sha256_file(provenance_path),
        expected_gaussians=gaussian_count,
        expected_formal=False,
        expected_provenance_schema="radeon_oneloop.layered_gaussian_provenance.v1",
        expected_provenance_class="confidence_fused_candidate",
        required_observed_only_training=False,
    )


def completed_appearance_asset(root: Path) -> ObservedCoreAsset:
    """Load one unified, full-orbit appearance distilled from reviewed views.

    This is deliberately a separate asset class from both the observed-only
    formal baseline and the optional generated-fill layer.  It is one rigid
    Gaussian field at runtime, while its provenance remains explicit that
    generated SEVA pseudo-views contributed appearance supervision.
    """

    resolved = root.resolve()
    ply = resolved / "appearance_completed_canonical.ply"
    cameras = resolved / "cameras_completed.json"
    provenance_path = resolved / "provenance.json"
    missing = [str(path) for path in (ply, cameras, provenance_path) if not path.is_file()]
    if missing:
        raise GaussianAppearanceError(f"completed-appearance files are missing: {missing}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    try:
        assert_not_quarantined(
            [
                (
                    "completed_appearance_runtime_asset",
                    {
                        "ply_sha256": sha256_file(ply),
                        "cameras_sha256": sha256_file(cameras),
                        "provenance_sha256": sha256_file(provenance_path),
                        "provenance": provenance,
                    },
                )
            ]
        )
    except QuarantinedLineageError as error:
        raise GaussianAppearanceError(
            "completed appearance is quarantined and cannot be a live asset: "
            f"{error}"
        ) from error
    if provenance.get("formal") is not False:
        raise GaussianAppearanceError("completed appearance requires formal=false provenance")
    if provenance.get("eligible_for_heldout_real_metrics") is not False:
        raise GaussianAppearanceError("completed appearance cannot be held-out-real evidence")
    gaussian_count = provenance.get("gaussian_count")
    if not isinstance(gaussian_count, int) or gaussian_count <= 0:
        raise GaussianAppearanceError("completed-appearance Gaussian count must be positive")
    return ObservedCoreAsset(
        ply_path=ply,
        cameras_path=cameras,
        provenance_path=provenance_path,
        expected_ply_sha256=sha256_file(ply),
        expected_cameras_sha256=sha256_file(cameras),
        expected_provenance_sha256=sha256_file(provenance_path),
        expected_gaussians=gaussian_count,
        expected_formal=False,
        expected_provenance_schema="radeon_oneloop.gaussian_canonicalization.v2",
        expected_provenance_class="completed_appearance_candidate",
        required_observed_only_training=False,
    )


def full_geometry_candidate_asset(root: Path) -> ObservedCoreAsset:
    """Load a generated full-geometry visual candidate pending human audit."""

    resolved = root.resolve()
    ply = resolved / "appearance_full_geometry_canonical.ply"
    cameras = resolved / "cameras_full_geometry.json"
    provenance_path = resolved / "provenance.json"
    missing = [str(path) for path in (ply, cameras, provenance_path) if not path.is_file()]
    if missing:
        raise GaussianAppearanceError(f"full-geometry candidate files are missing: {missing}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    try:
        assert_not_quarantined(
            [
                (
                    "full_geometry_runtime_asset",
                    {
                        "ply_sha256": sha256_file(ply),
                        "cameras_sha256": sha256_file(cameras),
                        "provenance_sha256": sha256_file(provenance_path),
                        "provenance": provenance,
                    },
                )
            ]
        )
    except QuarantinedLineageError as error:
        raise GaussianAppearanceError(
            "full-geometry candidate contains quarantined lineage: " f"{error}"
        ) from error
    if provenance.get("formal") is not False:
        raise GaussianAppearanceError("full-geometry candidate requires formal=false")
    if provenance.get("eligible_for_heldout_real_metrics") is not False:
        raise GaussianAppearanceError("full-geometry candidate cannot be held-out evidence")
    lineage = provenance.get("training_lineage", {})
    if lineage.get("secondary_accelerator_artifacts") is not True:
        raise GaussianAppearanceError("generated geometry lineage must remain explicit")
    if lineage.get("training_job_role") != (
        "SEVA_49_view_generated_geometry_hypothesis_variable_3DGS"
    ):
        raise GaussianAppearanceError("unexpected full-geometry training role")
    gaussian_count = provenance.get("gaussian_count")
    if not isinstance(gaussian_count, int) or gaussian_count <= 0:
        raise GaussianAppearanceError("full-geometry Gaussian count must be positive")
    return ObservedCoreAsset(
        ply_path=ply,
        cameras_path=cameras,
        provenance_path=provenance_path,
        expected_ply_sha256=sha256_file(ply),
        expected_cameras_sha256=sha256_file(cameras),
        expected_provenance_sha256=sha256_file(provenance_path),
        expected_gaussians=gaussian_count,
        expected_formal=False,
        expected_provenance_schema="radeon_oneloop.gaussian_canonicalization.v2",
        expected_provenance_class="generated_full_geometry_candidate",
        required_observed_only_training=False,
    )
