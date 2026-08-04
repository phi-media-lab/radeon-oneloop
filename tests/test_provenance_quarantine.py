import json
from pathlib import Path

import pytest

from gaussian.provenance_quarantine import (
    DEFAULT_REGISTRY,
    QuarantinedLineageError,
    assert_not_quarantined,
    find_quarantine_matches,
    load_registry,
)


def test_default_registry_is_valid_and_has_unique_triggers() -> None:
    registry = load_registry()
    assert registry["policy"]["applies_to_descendants"] is True
    triggers = []
    for entry in registry["entries"]:
        triggers.extend(entry.get("run_ids", []))
        triggers.extend(entry.get("sha256", []))
        triggers.extend(entry.get("dataset_hashes", []))
    assert len(triggers) == len(set(triggers))


def test_nested_descendant_lineage_is_rejected() -> None:
    payload = {
        "lineage": {
            "texture_manifest_sha256": (
                "af509b8a43687534d7f06bc4d798085f1731f415842064bb758a324cf14ae426"
            )
        }
    }
    matches = find_quarantine_matches([("dataset_manifest", payload)])
    assert [item.entry_id for item in matches] == ["distorted_hunyuan3d_2mv_seed_mesh_v2"]
    with pytest.raises(QuarantinedLineageError, match="negative control"):
        assert_not_quarantined([("dataset_manifest", payload)])


def test_quarantined_dataset_hash_is_rejected() -> None:
    with pytest.raises(QuarantinedLineageError, match="dataset_hash"):
        assert_not_quarantined(
            [
                (
                    "vksplat_dataset",
                    "f3c2c58d47d805daa74eb91376283c654c2aa2cf1e99fa8e25834aeeceb03014",
                )
            ]
        )


def test_procedural_surface_carrier_descendant_is_rejected() -> None:
    payload = {
        "source": {
            "surface_carrier_manifest_sha256": (
                "c30186ef633f786995651dc3f5b4021f4cc04d2e4fb7aca8d04b991dbfc9d5de"
            )
        }
    }
    matches = find_quarantine_matches([("vista4d_input", payload)])
    assert [item.entry_id for item in matches] == [
        "procedural_surface_carrier_invalid_complete_prior_v1"
    ]
    with pytest.raises(QuarantinedLineageError, match="complete_prior"):
        assert_not_quarantined([("vista4d_input", payload)])


def test_surface_carrier_vista_and_genesis_descendants_are_rejected() -> None:
    payload = {
        "video_sha256": (
            "3e2063f09915c03e6ad200ade2d228c3ca2689cb451cf357cb51e3c3610c1a3d"
        ),
        "glb_sha256": (
            "3ac021c0a091081c6a03af3b016b9ce37978267ff4c4e6ebe5666d7b8fddad0f"
        ),
    }
    matches = find_quarantine_matches([("candidate", payload)])
    assert [item.entry_id for item in matches] == [
        "procedural_surface_carrier_genesis_glb_descendant_v1",
        "procedural_surface_carrier_vista4d_descendants_v1",
    ]


def test_rejected_seed10030_vista_reshoot_is_quarantined_without_rejecting_mesh() -> None:
    with pytest.raises(QuarantinedLineageError, match="seed10030_vista4d"):
        assert_not_quarantined(
            [
                (
                    "vista4d_review",
                    "47629db7c680e29d0208f89e498a0971a0c55ba78a387ef95abcf501da0f1081",
                )
            ]
        )
    # The independent accepted Hunyuan seed10030 mesh remains a nonformal baseline.
    assert_not_quarantined(
        [
            (
                "hunyuan_mesh",
                "cc67f708",  # deliberately not a quarantine trigger
            )
        ]
    )


def test_unrelated_observed_only_manifest_passes(tmp_path: Path) -> None:
    observed = {
        "schema_version": "radeon_oneloop.object_colmap_dataset.v1",
        "formal": False,
        "provenance": "observed_real_only",
        "manifest_sha256": "0" * 64,
    }
    assert_not_quarantined([("observed", observed)])
    registry_copy = tmp_path / "registry.json"
    registry_copy.write_text(json.dumps(json.loads(DEFAULT_REGISTRY.read_text())) + "\n")
    assert load_registry(registry_copy)["schema_version"].endswith(".v1")
