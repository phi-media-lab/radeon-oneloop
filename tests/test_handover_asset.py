import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Local scaffold checks can run before project dependencies.
    np = None

if np is not None:
    from sim.genesis_so101.handover_asset import (
        DEFAULT_CONFIG,
        MATERIALS,
        build_mesh,
        build_visual_parts,
        load_spec,
        signed_mesh_volume,
        write_asset,
    )


@unittest.skipIf(np is None, "numpy is not installed")
class HandoverAssetTests(unittest.TestCase):
    def test_soft_body_mesh_is_closed_and_has_expected_scale(self):
        spec = load_spec(DEFAULT_CONFIG)
        vertices, faces = build_mesh(spec)
        edges = {}
        for face in faces:
            for start, stop in zip(face, np.roll(face, -1)):
                edge = tuple(sorted((int(start), int(stop))))
                edges[edge] = edges.get(edge, 0) + 1
        self.assertTrue(edges)
        self.assertEqual(set(edges.values()), {2})
        np.testing.assert_allclose(
            np.ptp(vertices, axis=0),
            np.asarray(spec.semi_axes_m) * 2.0,
            rtol=0.035,
        )
        self.assertLess(
            abs(signed_mesh_volume(vertices, faces) / spec.analytic_volume_m3 - 1.0),
            0.03,
        )

    def test_visual_asset_has_observed_hybrid_parts_and_orientation(self):
        spec = load_spec(DEFAULT_CONFIG)
        parts = build_visual_parts(spec, include_accessories=False)
        by_name = {part.name: part for part in parts}
        self.assertTrue(
            {
                "plush_body",
                "pink_face_left_lobe",
                "pink_face_right_lobe",
                "pink_face_muzzle",
                "viewer_left_ear",
                "viewer_right_ear",
                "left_hand",
                "right_hand",
                "left_shoe",
                "right_shoe",
            }.issubset(by_name)
        )
        self.assertEqual(by_name["viewer_left_ear"].material, "vinyl_black")
        self.assertEqual(by_name["viewer_right_ear"].material, "ear_blue")
        # The canonical camera sits on +Y and looks toward the origin, so its
        # screen-left direction is world +X.
        self.assertGreater(by_name["viewer_left_ear"].vertices[:, 0].mean(), 0.0)
        self.assertLess(by_name["viewer_right_ear"].vertices[:, 0].mean(), 0.0)
        self.assertGreater(by_name["pink_face_muzzle"].vertices[:, 1].mean(), 0.0)

    def test_display_asset_adds_flexible_accessories(self):
        spec = load_spec(DEFAULT_CONFIG)
        simulation = build_visual_parts(spec, include_accessories=False)
        display = build_visual_parts(spec, include_accessories=True)
        simulation_names = {part.name for part in simulation}
        display_names = {part.name for part in display}
        self.assertNotIn("mickey_strap", simulation_names)
        self.assertIn("mickey_strap", display_names)
        self.assertIn("keyring_head", display_names)
        self.assertGreater(len(display), len(simulation))
        self.assertIn("graffiti_pink", MATERIALS)

    def test_generated_asset_is_content_hashed(self):
        spec = load_spec(DEFAULT_CONFIG)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = write_asset(spec, root / "asset.obj", root / "asset.json")
            self.assertEqual(report["collision"]["vertices"], len(build_mesh(spec)[0]))
            self.assertEqual(len(report["sim_visual"]["sha256"]), 64)
            self.assertTrue((root / "asset.obj").is_file())
            self.assertTrue((root / "asset_collision.obj").is_file())
            self.assertTrue((root / "asset_display.obj").is_file())
            self.assertTrue((root / "asset.mtl").is_file())


if __name__ == "__main__":
    unittest.main()
