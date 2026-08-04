#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'usage: %s <conditioning-folder> <output-root> <vista4d-root> <python-bin>\n' "$0" >&2
  exit 64
}

[[ $# -eq 4 ]] || usage
input_folder=$1
output_root=$2
vista4d_root=$3
python_bin=$4
seed=${ONELOOP_VISTA4D_SEED:-10027}
cfg_scale=${ONELOOP_VISTA4D_CFG_SCALE:-5.0}
prompt=${ONELOOP_VISTA4D_PROMPT:-A studio turntable video of one small oval white plush Graffiti Mickey doll, with a pink Mickey face on the front, one viewer-left black vinyl ear with pink graffiti markings, one viewer-right cyan-blue vinyl ear, yellow shoes, and soft fuzzy fabric. The same static object rotates smoothly on a clean white background with stable identity, markings, and proportions.}
negative_prompt=${ONELOOP_VISTA4D_NEGATIVE_PROMPT:-multiple objects, duplicate doll, duplicated ears, extra face, missing ears, morphing, deformation, changing markings, changing identity, floating parts, text, watermark, hands, people, cluttered background, blur, low quality}
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "$runner_dir/.." && pwd)
script_path="$runner_dir/$(basename "${BASH_SOURCE[0]}")"
wrapper_path=$(cd "$runner_dir/../gaussian" && pwd)/vista4d_inference_cfg.py
checkpoint_dir="$vista4d_root/checkpoints/vista4d/384p49_step=30000"
wan_root="$vista4d_root/checkpoints/wan"
wan_name=Wan2.1-T2V-14B
wan_paths="${wan_name}:diffusion_pytorch_model*.safetensors,${wan_name}:models_t5_umt5-xxl-enc-bf16.pth,${wan_name}:Wan2.1_VAE.pth"
tokenizer_paths="${wan_name}:google/*"
run_id="vista4d_object_$(date -u +%Y%m%dT%H%M%SZ)_${BASHPID}_seed${seed}"
run_dir="$output_root/$run_id"
inference_dir="$run_dir/inference"

mkdir -p "$output_root"
[[ ! -e "$run_dir" ]]
mkdir -p "$inference_dir"

mark_failure() {
  status=$?
  if [[ $status -ne 0 && ! -e "$run_dir/DONE" ]]; then
    "$python_bin" - "$run_dir/FAILED" "$status" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

path, status = sys.argv[1:]
value = {
    "schema_version": "radeon_oneloop.vista4d_object_completion_failure.v1",
    "stage": "MI300X_Vista4D_object_appearance_completion",
    "status": "failed",
    "exit_code": int(status),
    "failed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  fi
  exit "$status"
}
trap mark_failure EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ "$seed" =~ ^[0-9]+$ ]]
[[ -d "$input_folder" ]]
[[ -f "$input_folder/input_manifest.json" ]]
[[ -f "$input_folder/hashes.sha256" ]]
[[ -f "$input_folder/DONE" ]]
[[ -d "$vista4d_root/.git" ]]
[[ -x "$python_bin" ]]
[[ -f "$wrapper_path" ]]
[[ -f "$checkpoint_dir/dit.pth" ]]
[[ -f "$checkpoint_dir/config.yaml" ]]
[[ -f "$wan_root/$wan_name/Wan2.1_VAE.pth" ]]
[[ -f "$wan_root/$wan_name/models_t5_umt5-xxl-enc-bf16.pth" ]]
[[ -f "$vista4d_root/scripts/inference/inference.py" ]]

PYTHONPATH="$repo_root" "$python_bin" -m gaussian.provenance_quarantine \
  --check-json "$input_folder/input_manifest.json"

exec 9>/tmp/radeon-oneloop-gpu0.lock
if ! flock -n 9; then
  printf '%s\n' 'GPU0 is locked by another Radeon OneLoop job' >&2
  exit 75
fi

(cd "$input_folder" && sha256sum -c hashes.sha256 >/dev/null)
"$python_bin" - "$input_folder/input_manifest.json" "$input_folder/DONE" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
done = json.loads(Path(sys.argv[2]).read_text())
if manifest.get("schema_version") != "radeon_oneloop.vista4d_object_conditioning.v1":
    raise SystemExit("unexpected conditioning schema")
if manifest.get("frames") != 49 or manifest.get("image_size_wh") != [672, 384]:
    raise SystemExit("conditioning shape is not the Vista4D 384p49 contract")
if manifest.get("formal") is not False or manifest.get("physical_output") is not False:
    raise SystemExit("conditioning provenance boundary was weakened")
if manifest.get("eligible_for_heldout_real_metrics") is not False:
    raise SystemExit("generated conditioning cannot be held-out evidence")
if done.get("status") != "complete":
    raise SystemExit("conditioning bundle is incomplete")
if manifest.get("source_video", {}).get("role") == "four_real_view_projected_Hunyuan_learned_mesh_orbit":
    if manifest.get("source_video", {}).get("surface_carrier") is not None:
        raise SystemExit("learned-mesh mainline cannot contain a procedural carrier")
    if manifest.get("asset", {}).get("inherited_procedural_geometry") is not None:
        raise SystemExit("learned-mesh mainline inherited procedural geometry")
    if manifest.get("camera", {}).get("endpoint_duplicate") is not False:
        raise SystemExit("learned-mesh mainline duplicated the orbit endpoint")
    if manifest.get("camera", {}).get("render_camera_model") != "PINHOLE_OPENCV_fixed_intrinsic":
        raise SystemExit("learned-mesh mainline source is not camera-bound")
    review = manifest.get("manual_visual_review", {})
    if review.get("decision") != "accepted_conditioning_only":
        raise SystemExit("learned-mesh mainline lacks an accepted external visual review")
    if not review.get("review_sha256") or not all(review.get("checks", {}).values()):
        raise SystemExit("learned-mesh mainline review is incomplete or unbound")
PY

hardware_json=$(
  "$python_bin" - <<'PY'
import json
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("Vista4D job requires exactly one visible ROCm device")
name = torch.cuda.get_device_name(0)
if "MI300X" not in name:
    raise SystemExit(f"expected MI300X, got {name}")
print(json.dumps({
    "device": name,
    "device_count": torch.cuda.device_count(),
    "torch": torch.__version__,
    "hip": torch.version.hip,
}))
PY
)
"$python_bin" - "$cfg_scale" <<'PY'
import math
import sys

value = float(sys.argv[1])
if not math.isfinite(value) or not 1.0 <= value <= 10.0:
    raise SystemExit("Vista4D CFG scale must be finite and in [1, 10]")
PY

vista4d_commit=$(git -C "$vista4d_root" rev-parse HEAD)
git -C "$vista4d_root" status --porcelain=v1 >"$run_dir/vista4d_git_status.txt"
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
started_epoch=$(date +%s)

command=(
  "$python_bin" "$wrapper_path"
  --model_id_with_origin_paths "$wan_paths"
  --tokenizer_id_with_origin_path "$tokenizer_paths"
  --local_model_folder "$wan_root"
  --vista4d_checkpoint "$checkpoint_dir/dit.pth"
  --vista4d_config_path "$checkpoint_dir/config.yaml"
  --input_folder "$input_folder"
  --output_folder "$inference_dir"
  --prompt "$prompt"
  --negative_prompt "$negative_prompt"
  --height 384 --width 672 --num_frames 49
  --seed "$seed"
)
printf '%q ' "${command[@]}" >"$run_dir/command.sh"
printf '\n' >>"$run_dir/command.sh"

cd "$vista4d_root"
export ONELOOP_VISTA4D_CFG_SCALE="$cfg_scale"
PYTHONPATH="$vista4d_root${PYTHONPATH:+:$PYTHONPATH}" \
timeout --signal=TERM --kill-after=60 2400 \
  "${command[@]}" \
  >"$run_dir/stdout.log" 2>"$run_dir/stderr.log"

finished_epoch=$(date +%s)
finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
output_video="$inference_dir/video_seed=${seed}.mp4"
[[ -s "$output_video" ]]
[[ -s "$inference_dir/source.mp4" ]]
[[ -s "$inference_dir/point_cloud.mp4" ]]
[[ -s "$inference_dir/point_cloud_masks.mp4" ]]

"$python_bin" - "$run_dir/manifest.json" "$run_id" "$input_folder" \
  "$checkpoint_dir/dit.pth" "$checkpoint_dir/config.yaml" "$vista4d_root" \
  "$vista4d_commit" "$hardware_json" "$script_path" "$wrapper_path" "$seed" "$cfg_scale" "$prompt" \
  "$negative_prompt" "$started_utc" "$finished_utc" "$((finished_epoch - started_epoch))" \
  "$inference_dir" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

(
    manifest_path, run_id, input_folder, checkpoint, config, vista4d_root,
    vista4d_commit, hardware_json, runner, wrapper, seed, cfg_scale, prompt, negative_prompt,
    started_utc, finished_utc, runtime_s, inference_dir,
) = sys.argv[1:]

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def video_probe(path):
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(command, text=True))["streams"][0]

input_root = Path(input_folder)
inference_root = Path(inference_dir)
input_manifest = json.loads((input_root / "input_manifest.json").read_text())
outputs = []
for path in sorted(inference_root.glob("*.mp4")):
    outputs.append({
        "basename": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "video": video_probe(path),
    })
generated_name = f"video_seed={seed}.mp4"
generated = next(item for item in outputs if item["basename"] == generated_name)
if int(generated["video"]["nb_read_frames"]) != 49:
    raise SystemExit("generated Vista4D video does not contain 49 frames")
if [int(generated["video"]["width"]), int(generated["video"]["height"])] != [672, 384]:
    raise SystemExit("generated Vista4D video has the wrong resolution")

git_status = (Path(manifest_path).parent / "vista4d_git_status.txt").read_text()
value = {
    "schema_version": "radeon_oneloop.vista4d_object_completion_proposal.v1",
    "run_id": run_id,
    "formal": False,
    "eligible_for_formal_metrics": False,
    "eligible_for_heldout_real_metrics": False,
    "physical_output": False,
    "host_role": "phi_amd_work_mi300x_nonformal_generation_lab",
    "stage": "MI300X_Vista4D_object_appearance_completion",
    "hardware": json.loads(hardware_json),
    "started_utc": started_utc,
    "finished_utc": finished_utc,
    "runtime_s": int(runtime_s),
    "model": {
        "name": "Vista4D-384p49",
        "seed": int(seed),
        "checkpoint_sha256": sha256(checkpoint),
        "config_sha256": sha256(config),
        "vista4d_commit": vista4d_commit,
        "vista4d_worktree_dirty": bool(git_status),
        "vista4d_git_status_sha256": sha256(Path(manifest_path).parent / "vista4d_git_status.txt"),
        "runner_sha256": sha256(runner),
        "cfg_wrapper_sha256": sha256(wrapper),
        "cfg_scale": float(cfg_scale),
    },
    "conditioning": {
        "schema_version": input_manifest["schema_version"],
        "manifest_sha256": sha256(input_root / "input_manifest.json"),
        "hash_index_sha256": sha256(input_root / "hashes.sha256"),
        "conditioning_geometry_sha256": input_manifest["asset"]["hashes"]["ply"],
        "observed_gaussian_sha256": (
            input_manifest["asset"]["hashes"]["ply"]
            if "observed_gaussian" in input_manifest["point_cloud_condition"]["role"]
            else None
        ),
        "learned_mesh": (
            input_manifest["asset"]
            if "learned_mesh" in input_manifest["source_video"]["role"]
            else None
        ),
        "camera_trajectory": input_manifest["camera"]["trajectory"],
        "frames": input_manifest["frames"],
        "image_size_wh": input_manifest["image_size_wh"],
        "source_video_role": input_manifest["source_video"]["role"],
        "surface_carrier": input_manifest["source_video"].get("surface_carrier"),
    },
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "outputs": outputs,
    "generated_video_sha256": generated["sha256"],
    "allowed_role": "generated_appearance_pseudoview_completion",
    "geometry_status": "not_yet_lifted",
    "review_status": "pending_visual_and_identity_screen",
    "not_proven": [
        "metric geometry completion",
        "object identity outside observed evidence",
        "held-out real-view quality",
        "physics collision geometry",
        "single-Radeon reproducibility",
    ],
}
Path(manifest_path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

(cd "$run_dir" && find . -type f ! -name hashes.sha256 ! -name DONE ! -name FAILED \
  -print0 | sort -z | xargs -0 sha256sum >hashes.sha256)
"$python_bin" - "$run_dir/DONE" "$run_dir/manifest.json" <<'PY'
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

done_path, manifest_path = map(Path, sys.argv[1:])
value = {
    "schema_version": "radeon_oneloop.vista4d_object_completion_done.v1",
    "stage": "MI300X_Vista4D_object_appearance_completion",
    "status": "done_candidate_pending_visual_review",
    "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
}
done_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

trap - EXIT INT TERM
printf 'MI300X Vista4D object completion proposal complete: %s\n' "$run_dir"
