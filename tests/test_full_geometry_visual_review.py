from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from gaussian.full_geometry_visual_review import (
    FullGeometryVisualReviewError,
    NEXT_STAGE,
    authorize_receipt,
    build_packet,
    seal_receipt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _index(path: Path, records: list[tuple[str, Path]]) -> None:
    path.write_text(
        "".join(f"{_sha(source)}  ./{relative}\n" for relative, source in records),
        encoding="utf-8",
    )


class FullGeometryVisualReviewTests(unittest.TestCase):
    def make_inputs(self, root: Path):
        asset = root / "asset"
        source = root / "source"
        orbit = root / "orbit"
        live = root / "live"
        for directory in (asset, source, orbit / "artifacts", live / "renderer"):
            directory.mkdir(parents=True)

        ply = asset / "appearance_full_geometry_canonical.ply"
        cameras = asset / "cameras_full_geometry.json"
        provenance = asset / "provenance.json"
        ply.write_bytes(b"ply-candidate")
        cameras.write_text("{}\n", encoding="utf-8")
        asset_hashes = {
            "ply": _sha(ply),
            "cameras": _sha(cameras),
        }
        _json(
            provenance,
            {
                "schema_version": "radeon_oneloop.gaussian_canonicalization.v2",
                "provenance_class": "generated_full_geometry_candidate",
                "formal": False,
                "eligible_for_heldout_real_metrics": False,
                "gaussian_count": 39072,
                "output_ply_sha256": asset_hashes["ply"],
                "training_lineage": {
                    "training_run_id": "run",
                    "dataset_manifest_sha256": "1" * 64,
                    "vksplat_commit": "2" * 40,
                },
            },
        )
        asset_hashes["provenance"] = _sha(provenance)

        anchors = source / "real_generated_difference_anchors.png"
        source_video = source / "samples-rgb.mp4"
        anchors.write_bytes(b"anchor-comparison")
        source_video.write_bytes(b"49-view-video")
        _json(
            source / "accepted_review.json",
            {
                "schema_version": "radeon_oneloop.seva_four_view_orbit_review.v1",
                "decision": "accepted_low_confidence_pseudoviews",
                "accepted_role": "generated_low_confidence_appearance_pseudoviews",
                "evidence": {"anchor_comparison_sha256": _sha(anchors)},
            },
        )
        _json(source / "metrics.json", {"frames": 49, "formal": False})

        orbit_metrics = orbit / "artifacts/metrics.json"
        orbit_contact = orbit / "artifacts/orbit_contact_sheet.png"
        orbit_video = orbit / "artifacts/orbit_360.mp4"
        orbit_contact.write_bytes(b"orbit-contact")
        orbit_video.write_bytes(b"orbit-video")
        _json(
            orbit_metrics,
            {
                "accepted_numeric": True,
                "formal": False,
                "eligible_for_heldout_real_metrics": False,
                "frames": [{"index": index} for index in range(73)],
                "asset": {"hashes": asset_hashes},
            },
        )
        (orbit / "DONE").write_text("{}\n", encoding="utf-8")
        _index(
            orbit / "hashes.sha256",
            [
                ("artifacts/metrics.json", orbit_metrics),
                ("artifacts/orbit_contact_sheet.png", orbit_contact),
                ("artifacts/orbit_360.mp4", orbit_video),
            ],
        )

        live_video = live / "live_gaussian.mp4"
        live_first = live / "live_gaussian_first.png"
        live_final = live / "live_gaussian_final.png"
        live_video.write_bytes(b"live-video")
        live_first.write_bytes(b"live-first")
        live_final.write_bytes(b"live-final")
        checks = {"check_a": True, "check_b": True}
        appearance = {
            "object_visualization": False,
            "compositor": "gaussian_self_depth",
            "object_mesh_path": "/private/asset_collision.obj",
        }
        gate = live / "gate.json"
        renderer_metrics = live / "renderer/metrics.json"
        _json(
            gate,
            {
                "schema_version": "radeon_oneloop.gaussian_authoritative_synthetic_gate.v1",
                "accepted": True,
                "checks": checks,
                "physical_output": False,
                "serial_or_usb_access": False,
                "renderer_appearance": appearance,
            },
        )
        _json(
            renderer_metrics,
            {
                "asset": {"hashes": asset_hashes},
                "full_geometry_candidate": True,
            },
        )
        _index(
            live / "hashes.sha256",
            [
                ("gate.json", gate),
                ("renderer/metrics.json", renderer_metrics),
                ("renderer/live_gaussian.mp4", live_video),
                ("renderer/live_gaussian_first.png", live_first),
                ("renderer/live_gaussian_final.png", live_final),
            ],
        )
        return source, orbit, live, asset, asset_hashes

    def test_packet_and_receipt_authorize_only_exact_monitor_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, orbit, live, asset, hashes = self.make_inputs(root)
            packet_dir = root / "packet"
            packet = build_packet(
                source_review_root=source,
                orbit_run_root=orbit,
                live_run_root=live,
                asset_root=asset,
                output=packet_dir,
            )
            self.assertEqual(packet["status"], "pending_project_owner_visual_review")
            self.assertEqual(packet["asset"]["hashes"], hashes)
            self.assertTrue(all(value is None for value in packet["required_human_checks"].values()))

            receipt_dir = root / "receipt"
            receipt = seal_receipt(
                packet_dir=packet_dir,
                output=receipt_dir,
                decision="accepted",
                correct_identity_orientation=True,
                complete_orbit_acceptable=True,
                legacy_obj_not_visible=True,
                live_registration_acceptable=True,
            )
            self.assertTrue(receipt["accepted"])
            self.assertFalse(receipt["physical_output_commands"])
            self.assertFalse(receipt["haptic_output_authorized"])
            authorized = authorize_receipt(
                receipt_dir=receipt_dir,
                expected_asset_ply_sha256=hashes["ply"],
                target_stage=NEXT_STAGE,
            )
            self.assertEqual(authorized["next_authorized_stage"], NEXT_STAGE)
            with self.assertRaisesRegex(FullGeometryVisualReviewError, "different PLY"):
                authorize_receipt(
                    receipt_dir=receipt_dir,
                    expected_asset_ply_sha256="0" * 64,
                    target_stage=NEXT_STAGE,
                )

    def test_missing_human_check_preserves_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, orbit, live, asset, hashes = self.make_inputs(root)
            packet_dir = root / "packet"
            build_packet(
                source_review_root=source,
                orbit_run_root=orbit,
                live_run_root=live,
                asset_root=asset,
                output=packet_dir,
            )
            receipt_dir = root / "receipt"
            receipt = seal_receipt(
                packet_dir=packet_dir,
                output=receipt_dir,
                decision="accepted",
                correct_identity_orientation=True,
                complete_orbit_acceptable=False,
                legacy_obj_not_visible=True,
                live_registration_acceptable=True,
            )
            self.assertFalse(receipt["accepted"])
            with self.assertRaisesRegex(FullGeometryVisualReviewError, "not accepted"):
                authorize_receipt(
                    receipt_dir=receipt_dir,
                    expected_asset_ply_sha256=hashes["ply"],
                    target_stage=NEXT_STAGE,
                )

    def test_tampered_packet_cannot_be_sealed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, orbit, live, asset, _ = self.make_inputs(root)
            packet_dir = root / "packet"
            build_packet(
                source_review_root=source,
                orbit_run_root=orbit,
                live_run_root=live,
                asset_root=asset,
                output=packet_dir,
            )
            (packet_dir / "07_live_gaussian.mp4").write_bytes(b"tampered")
            with self.assertRaisesRegex(FullGeometryVisualReviewError, "hash mismatch"):
                seal_receipt(
                    packet_dir=packet_dir,
                    output=root / "receipt",
                    decision="accepted",
                    correct_identity_orientation=True,
                    complete_orbit_acceptable=True,
                    legacy_obj_not_visible=True,
                    live_registration_acceptable=True,
                )


if __name__ == "__main__":
    unittest.main()
