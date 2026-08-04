import copy
import json
from pathlib import Path
import unittest

from gaussian.prepare_object_views import (
    ConfigError,
    public_source_hashes,
    require_relative_path,
    validate_config,
)
from gaussian.sam3_object_masks import select_candidate


HASHES = [f"{index:064x}" for index in range(1, 9)]


def valid_config():
    views = []
    for index, label in enumerate(("front", "right", "rear", "left")):
        views.append(
            {
                "id": f"anchor_{label}",
                "source_id": "same_listing",
                "instance_id": "same_physical_unit",
                "source_relpath": f"listing/{label}.png",
                "source_sha256": HASHES[index],
                "tier": "A",
                "provenance": "observed",
                "view_label": label,
                "roles": [
                    "pose",
                    "photometric",
                    "evaluation",
                    "identity",
                    "generation_input",
                ],
                "canonical": True,
                "prepare": True,
                "grabcut_rect_fraction": [0.1, 0.1, 0.9, 0.9],
            }
        )
    return {
        "schema_version": "radeon_oneloop.object_asset_config.v1",
        "asset_name": "test_asset",
        "formal": False,
        "redistribution": False,
        "coordinate_convention": {
            "front_axis": "+Y",
            "up_axis": "+Z",
            "viewer_left_axis": "+X",
            "unit": "m",
            "origin": "plush_body_center",
        },
        "metric_anchor": {
            "kind": "product_specification",
            "dimension": "overall_height",
            "value_m": 0.095,
            "uncertainty_m": 0.005,
            "status": "user_confirmed_metric_anchor",
        },
        "normalization": {
            "output_size": 1024,
            "foreground_padding_fraction": 0.1,
            "grabcut_iterations": 8,
            "soft_alpha_sigma_px": 2.0,
            "min_foreground_fraction": 0.08,
            "max_foreground_fraction": 0.82,
        },
        "views": views,
    }


class ObjectViewConfigTests(unittest.TestCase):
    def test_valid_four_view_contract(self):
        validate_config(valid_config())

    def test_cross_instance_photometric_mix_is_rejected(self):
        config = valid_config()
        config["views"][3]["instance_id"] = "different_unit"
        with self.assertRaisesRegex(ConfigError, "coherent physical instance"):
            validate_config(config)

    def test_generated_view_cannot_enter_real_metrics(self):
        config = valid_config()
        config["views"][0]["provenance"] = "generated"
        config["views"][0]["tier"] = "G"
        with self.assertRaisesRegex(ConfigError, "generated views cannot supervise"):
            validate_config(config)

    def test_manufacturing_error_is_excluded_only(self):
        config = valid_config()
        bad = copy.deepcopy(config["views"][0])
        bad.update(
            {
                "id": "excluded_manufacturing_error",
                "source_relpath": "listing/error.png",
                "source_sha256": HASHES[7],
                "canonical": False,
                "exclusion_reason": "manufacturing error",
                "roles": ["identity"],
                "prepare": False,
            }
        )
        config["views"].append(bad)
        with self.assertRaisesRegex(ConfigError, "manufacturing-error"):
            validate_config(config)

    def test_private_source_path_cannot_escape(self):
        for invalid in ("../secret.png", "/absolute/secret.png", "folder/../secret.png"):
            with self.assertRaises(ConfigError):
                require_relative_path(invalid, field="source_relpath")

    def test_public_hashes_are_scoped_by_source(self):
        manifest = {
            "sources": [
                {"id": "a", "views": [{"sha256": HASHES[0]}]},
                {"id": "b", "views": [{"sha256": HASHES[1]}]},
            ]
        }
        self.assertEqual(public_source_hashes(manifest), {"a": {HASHES[0]}, "b": {HASHES[1]}})


class ObjectAssetManifestSchemaTests(unittest.TestCase):
    def test_prepared_view_requires_neutral_generator_input(self):
        schema = json.loads(
            Path("gaussian/object_asset_manifest.schema.json").read_text(encoding="utf-8")
        )
        prepared_rule = schema["$defs"]["view"]["allOf"][0]
        self.assertIn("neutral_image", prepared_rule["then"]["required"])

    def test_generated_provenance_forces_tier_g(self):
        schema = json.loads(
            Path("gaussian/object_asset_manifest.schema.json").read_text(encoding="utf-8")
        )
        generated_rule = schema["$defs"]["view"]["allOf"][1]
        self.assertEqual(
            generated_rule["then"]["properties"]["tier"]["const"],
            "G",
        )


class Sam3MaskCandidateTests(unittest.TestCase):
    def test_highest_scoring_safe_candidate_is_selected(self):
        selected = select_candidate(
            [
                {"score": 0.8, "area_fraction": 0.2, "touches_border": False},
                {"score": 0.9, "area_fraction": 0.3, "touches_border": False},
                {"score": 0.99, "area_fraction": 0.4, "touches_border": True},
            ]
        )
        self.assertEqual(selected["score"], 0.9)

    def test_floor_or_full_frame_masks_are_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "no candidate"):
            select_candidate(
                [
                    {"score": 0.99, "area_fraction": 0.9, "touches_border": False},
                    {"score": 0.95, "area_fraction": 0.4, "touches_border": True},
                ]
            )


if __name__ == "__main__":
    unittest.main()
