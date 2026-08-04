import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gaussian.vksplat_render_ply import (
    VkSplatRenderError,
    read_3dgs_ply,
    validate_source_provenance,
)


class VkSplatRenderPlyTests(unittest.TestCase):
    def test_formal_render_requires_formal_radeon_lineage(self):
        provenance = {
            "formal": True,
            "training_lineage": {
                "training_formal": True,
                "secondary_accelerator_artifacts": False,
            },
        }
        validate_source_provenance(
            provenance, formal=True, host_role="radeon_c_gpu0_gfx1100_formal"
        )
        with self.assertRaisesRegex(VkSplatRenderError, "formal host role"):
            validate_source_provenance(provenance, formal=True, host_role="radeon_f")

    def test_binary_ply_decodes_logits_and_ignores_audit_fields(self):
        fields = [
            "x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
            "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
        ]
        header = ["ply", "format binary_little_endian 1.0", "element vertex 1"]
        header.extend(f"property float {name}" for name in fields)
        header.append("property uchar source_view")
        header.append("end_header")
        values = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.0, np.log(0.01), np.log(0.02), np.log(0.03), 1.0, 0.0, 0.0, 0.0]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.ply"
            path.write_bytes(("\n".join(header) + "\n").encode("ascii") + struct.pack("<14fB", *values, 2))
            result = read_3dgs_ply(path)
        np.testing.assert_allclose(result["xyz"], [[1.0, 2.0, 3.0]])
        np.testing.assert_allclose(result["scales"], [[0.01, 0.02, 0.03]], rtol=1.0e-6)
        np.testing.assert_allclose(result["opacities"], [[0.5]])
        np.testing.assert_allclose(result["rotations"], [[1.0, 0.0, 0.0, 0.0]])
        np.testing.assert_allclose(result["sh"][0, 0], [0.1, 0.2, 0.3])
        self.assertEqual(result["sh"].shape, (1, 16, 3))


if __name__ == "__main__":
    unittest.main()
