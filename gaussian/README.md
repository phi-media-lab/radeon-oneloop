# Gaussian workspace workstream

> Competition status: the **static workspace reconstruction remains deferred /
> non-formal** because no calibrated workspace capture passed its input gate.
> The separate object-centric observed Gaussian is accepted for the demo and
> now has passing continuous-orbit and Genesis runtime bindings. The pinned
> formal asset remains the anchor-view baseline; the geometry-frozen live-demo
> candidate requires an explicit nonformal switch. Do not promote MI300X
> generator outputs or workspace experiments into formal result tables.

Build a static, calibrated Gaussian representation of the real handover
workspace using VkSplat/Vulkan RADV. The competition deliverable is a visual
twin and synchronized trajectory replay, not a policy observation, collision
model, dynamic 4DGS system, or online correction pipeline.

The implementation is pinned to VkSplat commit
`e26c254938c81ff85998cd357a9e005e255d9b03`. Bootstrap the Vulkan extension,
then train from an immutable COLMAP dataset:

```bash
bash ops/bootstrap_vksplat.sh
python -m gaussian.vksplat_train \
  --source /root/radeon-oneloop-deps/vksplat-e26c254938c81ff85998cd357a9e005e255d9b03 \
  --dataset /root/radeon-oneloop-data/gaussian/workspace_v1 \
  --output /root/radeon-oneloop-runs/gaussian/workspace_v1 \
  --steps 30000 --evaluate
```

`workspace_capture.schema.json` freezes the capture identity, camera model,
image count, and metric scale anchor. The runner validates the COLMAP model,
hashes every source image and calibration file, disables the viewer, and emits
the splat hash, quality metrics, training time, and Vulkan VRAM evidence.

## Historical HIL Real2Sim capture

The historical-data path keeps raw robot data outside the repository. Export a
timestamp-linked capture from one or more reviewed LeRobot v3 episodes, then
run CPU COLMAP on the moving wrist camera:

```bash
python -m gaussian.hil_capture \
  --dataset /path/to/lerobot-v3-dataset \
  --output /path/to/real2sim/workspace-p0 \
  --episodes 0,2,5 --require-success \
  --camera hand_cam --sample-hz 2 --front-anchors 3

python -m gaussian.colmap_workspace \
  --workspace /path/to/real2sim/workspace-p0 \
  --camera-model OPENCV --matcher sequential
```

The capture stores selected RGB frames, per-episode trajectory archives,
relative source names, hashes, and timestamp correspondences. It deliberately
does not claim metric calibration: `coordinate_alignment.status` remains
`pending_hand_eye_similarity` until the COLMAP camera poses have been aligned
to synchronized SO-101 forward kinematics. Fixed front-camera frames are kept
as registration anchors and are not mixed into the moving-camera model.

The handover workspace is strongly planar. If incremental COLMAP rejects the
sequence as a planar degeneracy, recover the actual tabletop directly from the
yellow target square instead of weakening the SfM quality gate:

```bash
python -m gaussian.planar_workspace \
  --workspace /path/to/real2sim/workspace-p0
```

This produces a tracked unit-square camera trajectory, a pinhole
self-calibration, a temporal-median orthomosaic, and a coverage map. The plane
is still unitless at this stage; only the later SO-101 kinematic alignment may
promote it to a metric Genesis asset.

For the fixed front camera, export all reviewed success episodes with
`--camera front_cam`, then recover the static background and three target
regions by temporal median:

```bash
python -m gaussian.fixed_workspace \
  --workspace /path/to/front-camera-capture
```

The variation image records where moving arms and the object made the median
less certain; fixed camera letterboxing is detected and removed explicitly.
It also writes `genesis_table_texture.png` and a quality-gated
`front_camera_calibration.json`. The latter is a P0 view alignment derived
from three coplanar square contours. Its scale is provisional, and it must not
be described as a surveyed or metrology-grade calibration.

After an offline Genesis replay, validate the rendered target registration
against the fixed-camera median image. Robot-occluded targets are omitted only
when their full quadrilateral is not detectable:

```bash
python -m gaussian.front_alignment \
  --reference /path/to/front_background_median.png \
  --simulation /path/to/replay_front_cam.png \
  --output /path/to/front-alignment
```

Historical hand-camera hand-eye alignment remains quality-gated. The present
LeRobot joint values do not agree closely enough with the pinned MJCF forward
kinematics to promote the planar wrist-camera result to metric alignment. Keep
the rejected alignment artifacts and collect a surveyed marker sequence before
using the wrist stream as a metric 3D camera.

## Graffiti Mickey object asset

The static-workspace deferral above does not block the handover object's
appearance asset. Existing four-side product views, supplemental exact-variant
views, historical HIL frames, and the 95 mm product specification are enough
to start an object-centric reconstruction without another capture prerequisite.

The approved technical plan is
[`GRAFFITI_MICKEY_ASSET_PIPELINE.md`](GRAFFITI_MICKEY_ASSET_PIPELINE.md). It
keeps the real-photo Gaussian core, generated completion, and Genesis physics
proxy as separate provenance layers. Generated geometry may fill the underside
and other poorly observed regions, but it is excluded from held-out real-view
metrics and can be disabled without breaking the object or the competition's
single-Radeon execution path.

Implemented object-centric stages include:

- deterministic manual-ring and MI300X VGGT-Omega metric pose initialization;
- COLMAP-text export and real-only VkSplat training/visual probes;
- MI300X SHARP-to-canonical alignment and cross-view generated-fill fusion;
- MI300X UniSHARP per-anchor Gaussian generation and immutable visual auditing
  that accepts appearance proposals independently from metric geometry;
- direct VkSplat/RADV rendering of an arbitrary generated 3DGS PLY;
- immutable visual-audit manifests that explicitly distinguish duplicate
  training-view QA from held-out-real evaluation;
- exact inversion of VkSplat's dataparser similarity so the real-only observed
  PLY, Gaussian scales, and quaternions return to the 95 mm canonical frame.

The SHARP fusion defaults to visible-surface layers with cross-source spatial
support. All-layer and one-per-voxel reductions remain available as ablations;
they are not promoted when the RADV render exposes interior leakage or sparse
coverage. Generated PLYs remain separate from the observed appearance core and
are always marked `formal: false` and ineligible for held-out-real metrics.
The measured UniSHARP candidate is intentionally retained as a high-quality
local pseudo-view source after failing its VGGT metric-geometry gate; it is not
silently promoted to the canonical object frame.

The current MI300X run exports 80 lossless pseudo-views with exact local camera
metadata. A follow-up SHARP-geometry/UniSHARP-appearance PLY passed numeric
alignment but failed the RADV visual gate because color and opacity donation
did not repair missing exterior geometry. The immutable negative result keeps
the pseudo-view branch alive while preventing promotion of the direct hybrid.

Allowing low-confidence single-source surfaces restores a much more continuous
shell. The current depth-pruned candidate keeps 229,576 Gaussians: all
cross-source points plus single-source points that agree with source VGGT depth
and do not conflict in front of non-source real surfaces. It is a refinement
initializer only; observed-visibility masking and real-photo optimization are
still required before release.

## Formal observed-object lineage

The default 30,000-Gaussian core no longer depends on the earlier `radeon-f`
candidate. Its registered formal lineage is:

1. Four masked real views and a deterministic CPU visual hull produce dataset
   SHA-256
   `682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e`.
   Learned depth, generated views, generated geometry, and secondary
   accelerator artifacts are excluded.
2. `20260804T095251Z_gaussian_train_84d468b_20260804` runs 2,000 VkSplat
   steps on `radeon-c` GPU0/gfx1100 and emits PLY SHA-256
   `d95edcb66edd5fd3f6fe3fda4686dfe718ce867507a9c4db0fa6dde88a2cfcc5`.
3. `20260804T095859Z_gaussian_render_c8bd111_20260804` applies the metric
   inverse similarity and emits canonical PLY SHA-256
   `0e26b6c4f993a7052fb471ad84a1a98180b262c868a4b179ce19b294b288bd1a`
   plus provenance SHA-256
   `80efa4f5a98070395844205afa663ee8ca2975eda21e720c3cf785dcfd52bd02`.
4. The same job renders front/right/rear/left at 1024×1024 through pinned
   VkSplat commit `e26c254938c81ff85998cd357a9e005e255d9b03`.

These same-view renders prove lineage, canonical orientation, identity, and
renderer compatibility. They do not prove held-out or novel-view quality.
Identically seeded preflights were numerically and visually stable but not
byte-identical because Vulkan floating-point atomic ordering can vary; bitwise
checkpoint determinism is therefore not claimed. Generated branches remain
disableable, nonformal, and subordinate to this real-only asset.

## Continuous-view candidate and audit

A 360-degree audit exposed an important distinction: the registered formal
asset passes its declared four-anchor checks but produces doubled surfaces and
tearing between anchors. It remains the default evidence asset, not the final
continuous-view appearance.

The current live-demo candidate starts from the same deterministic 30,000-point
95 mm visual hull and four real Mercari views of the same doll, then freezes
the Gaussian centers, scales, and quaternions while fitting DC color and
opacity. Higher SH and refinement stay disabled. Run this nonformal preflight
on the single declared Radeon host with external data paths supplied through
the environment:

```bash
ONELOOP_OBJECT_DATASET=/path/to/observed_only_dataset \
ONELOOP_CAMERA_STAGE=/path/to/canonical_camera_stage \
./ops/run_radeon_c_object_geometry_frozen_preflight.sh
```

The accepted candidate PLY is then audited on the development APU over 72
distinct angles plus a repeated 360-degree endpoint:

```bash
ONELOOP_OBSERVED_CORE_ROOT=/path/to/nonformal_candidate \
ONELOOP_ORBIT_CANDIDATE_NONFORMAL=1 \
./ops/run_amd_gaussian_orbit_audit.sh
```

Numeric acceptance requires a closed cycle, non-empty support, and no border
contact. The generated MP4 and 12-angle contact sheet still require human
review. The current result passes and is materially more coherent, but sparse
four-view side-angle blur remains and no held-out-real quality metric is
claimed.

## Accepted nonformal runtime binding

The object Gaussian is not a collision body or a policy input. Genesis owns the
rigid proxy, object pose, link segmentation, and depth. A separate VkSplat
renderer receives only a pose snapshot and renders the observed-only PLY. The
camera transform is frozen as:

```text
T_camera_object_opencv =
  inverse(T_world_camera_opengl · diag(1,-1,-1,1)) · T_world_object_canonical
```

The pinned-default binding validates all three content hashes and the upstream
`formal=true` flag before renderer initialization. The explicit candidate path
self-binds its three hashes and instead requires `formal=false` plus
`eligible_for_heldout_real_metrics=false`. The `amd` execution remains
`formal=false`; formal input provenance and formal execution provenance are
separate claims. The following latest gates pass with generated fill disabled:

- capability and integrity probe:
  `20260804T101510Z_167855_amd_gaussian_appearance_probe`, which binds the
  canonical PLY, cameras, and provenance hashes before one 1024×1024 render;
- static registration: `20260804T101807Z_169784_amd_gaussian_static_gate`,
  eight of eight renders, zero fallback, and 95.269 mm lightly trimmed center
  height versus the 95 mm anchor. The 0.01% tail trim removes three splats per
  side; the acceptance tolerance remains 2.85 mm;
- gripper/tabletop depth compositing:
  `20260804T101848Z_171506_amd_gaussian_occlusion_gate`, with foreground RGB
  preservation error exactly zero;
- decoupled 10-second live run:
  `20260804T101926Z_173198_amd_decoupled_gaussian_live_gate`, 1,200 control
  steps at 119.999 Hz while the non-authoritative renderer produced 48
  dual-camera frames and 96 successful appearance renders with zero fallback;
- hard renderer crash:
  `20260804T102041Z_176664_amd_decoupled_gaussian_live_gate`, where the
  renderer exits with code 86 after three frames and control still completes
  1,200 steps with no watchdog event or physical output.

The initial coupled live renderer achieved only 79.087 Hz and is retained as a
rejected architecture result. Earlier candidate-asset gates and static/
occlusion runs that reused cached Genesis masks are superseded; only the run
IDs above close the formal-asset integration gates. These are development
results on the AMD APU, not formal Radeon measurements.

For an operator-facing read-only demo, the explicit candidate flag keeps the
pinned default intact. The presenter binds only to loopback; the leader process
uses haptic monitor mode and never writes physical output:

```bash
ONELOOP_OBSERVED_CORE_ROOT=/path/to/nonformal_candidate \
ONELOOP_LIVE_CANDIDATE_NONFORMAL=1 \
./ops/run_amd_real2sim_live_demo.sh \
  LEFT_PORT RIGHT_PORT LEFT_CALIBRATION_ID RIGHT_CALIBRATION_ID
```

The launcher defaults to an 8 Hz dual-camera Gaussian view for 30 minutes and
opens the loopback presenter. Override `ONELOOP_LIVE_DURATION_S` for a bounded
recording and set `ONELOOP_RECORD_VIDEO=1` when an MP4 evidence artifact is
needed. Genesis remains authoritative at 120 Hz even if the renderer exits.

The dated hand-off state, exact run IDs, hashes, acceptance decisions, HIL
status, and remaining dynamic-integration gates are frozen in
[`../reports/progress_snapshot_2026-08-04.md`](../reports/progress_snapshot_2026-08-04.md)
and its machine-readable
[`../reports/real2sim_artifact_inventory_2026-08-04.yaml`](../reports/real2sim_artifact_inventory_2026-08-04.yaml).
