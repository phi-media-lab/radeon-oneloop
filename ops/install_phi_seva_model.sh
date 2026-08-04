#!/usr/bin/env bash
set -euo pipefail
umask 077

if [[ $# -ne 3 ]]; then
  printf 'usage: %s <local-model-root> <install-run-root> <python-bin>\n' "$0" >&2
  exit 64
fi

model_root=$1
run_root=$2
python_bin=$3
repo_id=stabilityai/stable-virtual-camera
revision=e538e251c1009e9a41cf8b7fee5f21332a1960de
expected_hf_user=fbsh96
run_id="seva_model_install_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}"
run_dir="$run_root/$run_id"

[[ -x "$python_bin" ]]
[[ "$revision" =~ ^[0-9a-f]{40}$ ]]
mkdir -p "$model_root" "$run_root"
[[ ! -e "$run_dir" ]]
mkdir "$run_dir"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status = sys.argv[1:]
Path(path).write_text(
    json.dumps(
        {
            "schema_version": "radeon_oneloop.seva_model_install_failure.v1",
            "formal": False,
            "stage": "authorized_fixed_revision_model_install",
            "status": "failed",
            "exit_code": int(status),
            "failed_utc": datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z"),
            "credential_material_recorded": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
PY
    (
      cd "$run_dir"
      find . -type f ! -name hashes.sha256 -print0 | sort -z | xargs -0 sha256sum >hashes.sha256
    )
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$python_bin" - "$model_root" "$repo_id" "$revision" "$expected_hf_user" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log" <<'PY'
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

model_root = Path(sys.argv[1]).resolve()
repo_id, revision, expected_hf_user = sys.argv[2:]

# This intentionally uses only the credential previously entered through the
# interactive Hugging Face CLI.  No token is accepted on this command line.
identity = HfApi().whoami()
actual_hf_user = identity.get("name") or identity.get("fullname")
if actual_hf_user != expected_hf_user:
    raise RuntimeError(
        f"wrong Hugging Face account: expected {expected_hf_user}, got {actual_hf_user}"
    )
info = HfApi().model_info(repo_id=repo_id, revision=revision)
if info.sha != revision:
    raise RuntimeError(f"resolved model revision drifted: {info.sha}")
for filename in ("modelv1.1.safetensors", "config.yaml"):
    resolved = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            local_dir=model_root,
        )
    ).resolve()
    if resolved.parent != model_root or not resolved.is_file() or resolved.stat().st_size == 0:
        raise RuntimeError(f"invalid downloaded model file: {resolved}")
print("authorized fixed-revision SEVA files are present")
PY

"$python_bin" - "$model_root" "$run_dir/manifest.json" "$repo_id" "$revision" "$expected_hf_user" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root, output = map(Path, sys.argv[1:3])
repo_id, revision, hf_user = sys.argv[3:]

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

files = {}
for name in ("modelv1.1.safetensors", "config.yaml"):
    path = root / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"installed SEVA file is missing: {name}")
    files[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
manifest = {
    "schema_version": "radeon_oneloop.seva_model_install.v1",
    "created_utc": datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z"),
    "formal": False,
    "host_role": "phi_amd_work_mi300x_nonformal_generation_lab",
    "repo_id": repo_id,
    "revision": revision,
    "huggingface_user": hf_user,
    "access": "authorized_huggingface_credential_from_interactive_cli",
    "credential_material_recorded": False,
    "files": files,
}
output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(
  cd "$run_dir"
  sha256sum manifest.json stderr.log stdout.log >hashes.sha256
)
"$python_bin" - "$run_dir" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
done = {
    "schema_version": "radeon_oneloop.seva_model_install_done.v1",
    "formal": False,
    "stage": "authorized_fixed_revision_model_install",
    "status": "done",
    "completed_utc": datetime.now(timezone.utc)
    .isoformat(timespec="seconds")
    .replace("+00:00", "Z"),
    "manifest_sha256": sha(root / "manifest.json"),
    "hashes_sha256": sha(root / "hashes.sha256"),
    "credential_material_recorded": False,
}
(root / "DONE").write_text(json.dumps(done, indent=2, sort_keys=True) + "\n")
PY
trap - EXIT INT TERM
printf 'authorized fixed-revision SEVA model installed: %s\n' "$run_dir"
