from __future__ import annotations

from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = REPO_ROOT / "ops/run_amd_seva_completed_appearance.sh"


def test_completed_appearance_entrypoint_is_a_fail_closed_negative_control() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")

    assert "freeze-geometry" not in source
    assert "vksplat_train" not in source
    assert "quarantined" in source

    result = subprocess.run(
        ["bash", str(ENTRYPOINT)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 65
    assert "REJECTED" in result.stderr
    assert "run_amd_seva_full_geometry.sh" in result.stderr
