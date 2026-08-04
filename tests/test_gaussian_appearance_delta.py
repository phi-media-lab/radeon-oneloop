import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.gaussian_appearance_delta import (
    AppearanceDeltaError,
    apply_delta,
    create_delta,
    sha256_file,
)


DTYPE = np.dtype(
    [(name, "<f4") for name in ("x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity")]
    + [("source_view", "u1")]
)


def write_ply(path: Path, vertices: np.ndarray) -> None:
    properties = "\n".join(
        f"property {'uchar' if dtype == np.dtype('u1') else 'float'} {name}"
        for name, (dtype, _) in vertices.dtype.fields.items()
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n{properties}\nend_header\n"
    ).encode("ascii")
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())


class GaussianAppearanceDeltaTests(unittest.TestCase):
    def test_round_trip_preserves_exact_target_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_values = np.zeros(4, dtype=DTYPE)
            base_values["x"] = np.arange(4)
            base_values["source_view"] = [0, 1, 2, 3]
            target_values = base_values.copy()
            target_values["f_dc_0"] = [0.1, 0.2, 0.3, 0.4]
            target_values["opacity"] = [1.0, 2.0, 3.0, 4.0]
            base = root / "base.ply"
            target = root / "target.ply"
            delta = root / "appearance.npz"
            output = root / "output.ply"
            write_ply(base, base_values)
            write_ply(target, target_values)
            create_delta(base, target, delta)
            result = apply_delta(base, delta, output)
            self.assertEqual(result["output_ply_sha256"], sha256_file(target))

    def test_rejects_geometry_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_values = np.zeros(1, dtype=DTYPE)
            target_values = base_values.copy()
            target_values["x"] = 1.0
            base = root / "base.ply"
            target = root / "target.ply"
            write_ply(base, base_values)
            write_ply(target, target_values)
            with self.assertRaises(AppearanceDeltaError):
                create_delta(base, target, root / "delta.npz")


if __name__ == "__main__":
    unittest.main()
