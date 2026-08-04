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
- Vista4D-compatible 49-frame object conditioning, MI300X appearance-video
  completion, and provenance-bound temporal/identity visual auditing;
- a preserved rejected 95 mm procedural surface-carrier ablation with bounded
  AMD-ROCm silhouette fitting, real-photo projection, mask-alignment sweep,
  and portable GLB conversion failure evidence;
- a geometry-free four-reviewed-view contract for SEVA camera-controlled
  orbit generation and Hunyuan3D-2mv complete-mesh generation;
- an observed-mask-only 30,000-point visual-hull initializer shared by the
  primary SEVA distillation path and independent fallback A/Bs;
- a content-addressed quarantine registry that rejects visually invalid mesh,
  Vista4D, hybrid-dataset, and VkSplat descendants before a new dataset build
  or GPU training job;
- independent learned-mesh orbit reviews that bind the exact contact sheet,
  source video, generated mesh, four-view input, and private HIL identity
  exemplars before an orbit can become Vista4D conditioning;
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

Vista4D is connected as a downstream video-reshooting stage, not as a four-
still-image or point generator. `prepare_vista4d_object_input.py` renders the
observed Gaussian into the exact 49-frame, 672 x 384
source/point/mask/camera contract;
`run_phi_vista4d_object_completion.sh` executes the nonformal MI300X run; and
`audit_vista4d_completion.py` records preservation, background, temporal, and
closed-loop diagnostics plus immutable contact sheets.  The same-source
baseline proves the executable interface but not object completion. A
controlled four-real-keyframe branch was rejected because it preserved the
photos locally but did not propagate their improvement and raised temporal
p95 residual from 0.03697 to 0.09882.

The procedural complete-source experiment is frozen as an invalid-geometry-
prior negative control. `surface_carrier.py` produced a 12,078-vertex /
24,044-triangle 95 mm carrier whose mean four-view silhouette IoU is 0.75497,
but real-photo projection cannot correct its distorted depth, proportions, or
topology. Its two Vista4D runs also overexpose and blur the face. Neither the
carrier nor any derivative is eligible for completion, depth lifting,
Gaussian distillation, or Genesis installation.

The corrected mainline starts with `prepare_four_view_generation.py`. It
hash-verifies the four reviewed same-instance observed photographs and emits
no inherited geometry: a four-input/49-target SEVA scene, exact target cameras,
and Hunyuan3D-2mv `front/left/back/right` RGBA inputs. See
`FOUR_VIEW_GENERATIVE_REAL2SIM.md` and
`../reports/vista4d_completion_experiment_2026-08-04.md`.

### Post-review lineage correction

The first Hunyuan3D-2mv seed (`10027`) was later rejected by visual review:
coarse silhouette agreement did not prevent material identity/topology
distortion. Its aligned/real-projected mesh, Vista4D proposal, hybrid COLMAP
dataset, and three Radeon-f VkSplat descendants are now listed in
`real2sim_quarantine.json`. `provenance_quarantine.py` is called before hybrid
dataset construction, Vista4D inference, and VkSplat training; matching any
ancestor hash or dataset hash is a hard error. The artifacts are preserved
only as negative controls.

Three fresh Hunyuan candidates were generated directly from the four observed
images with seeds `10028`, `10029`, and `10030`; none inherits the rejected
mesh. All three pass the explicit topology checklist, and seeds `10028` and
`10029` are retained as valid but dominated alternatives. Seed `10030` has the
best measured coarse four-view fit (mean silhouette IoU `0.77664`, minimum
`0.75134`) and is accepted only as generated conditioning. Its review binds
the continuous orbit and two private rear/top identity exemplars; it remains
ineligible for observed geometry, collision geometry, held-out metrics, or
formal lineage. Four-view texture projection still visibly blurs
inter-cardinal appearance and leaves the unseen underside neutral.

SEVA remains the preferred geometry-free multi-image novel-view front end,
but the phi host currently reports an invalid stored Hugging Face credential
and has no local v1.1 weights. This is an access gate, not a reason to reuse a
rejected mesh. Hunyuan seed `10030` is therefore a controlled nonformal
fallback while SEVA remains paused.

The seed-`10030` fallback is now fully separated from the observed core. Its
Vista4D reshoot is rejected: mean silhouette IoU is `0.43319`, the minimum is
`0.18367`, observed RGB MAE is `0.3543`, and visual review shows the front face
persisting around the orbit while rear/ear identity drifts. The direct
frozen-geometry VkSplat A/B starts from a 30,000-point initializer derived only
from the four real masks. Visibility pruning retains only 1,224 generated
Gaussians seen in at most one real anchor and rejects 28,776 that would overlap
at least two observed views. The resulting 31,224-Gaussian preview preserves
the 30,000-Gaussian observed core bit-for-bit.

A 49-view comparison accepts this layer only as an optional nonformal toggle,
default off: mean RGB change from the observed core is `0.001655`, worst real-
anchor change is `0.002220`, and minimum anchor silhouette IoU is `0.992553`.
Completion effectiveness remains inconclusive without held-out real views.

## Formal observed-object lineage

The default 30,000-Gaussian core no longer depends on the earlier `radeon-f`
candidate. The original registered baseline below is retained as prior formal
evidence, while the geometry-frozen successor is now the default:

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

The registered geometry-frozen successor is training job
`20260804T204531Z_gaussian_train_14053af_20260804` followed by render job
`20260804T205059Z_gaussian_render_e149f01_20260804`. It binds the same real-only
dataset, holds Gaussian centers and shape fixed, disables refinement and
generated fill, and emits trained PLY SHA-256
`e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378`
and canonical PLY SHA-256
`7f01c1e6d8253d7f15162e2cb51e18845676fa1015983266b7d356d9b21aa706`.
Its four-anchor render takes 1.176 s and reports 34,884,748 bytes peak Vulkan
memory. This is formal anchor-view evidence only, not held-out or continuous-
orbit quality evidence.

## Continuous-view candidate and audit

A 360-degree audit exposed an important distinction: the registered formal
asset passes its declared four-anchor checks but produces doubled surfaces and
tearing between anchors. It remains the default evidence asset, not the final
continuous-view appearance.

The previously audited live-demo candidate starts from the same deterministic 30,000-point
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
claimed. The exact registered successor now has its own 72-angle development
audit, `20260804T205804Z_217937_amd_gaussian_orbit_audit`: zero cycle-closure
error, zero border contacts, 25.108 ms mean render time, and a human-accepted
shell with the same side-angle blur/stretch limitation. It is therefore the
current continuous live-demo binding. Read-only runtime gate
`20260804T205940Z_218151_amd_decoupled_gaussian_live_gate` also passes with
zero renderer fallbacks, zero watchdog events, and zero physical-output
commands. Its leader inputs were stationary, so final task execution remains
unrecorded.

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

The generated-fill runtime path is a separate `layered-preview` asset class;
it cannot pass through the observed-only loader. Static Genesis gate
`layered_fusion_genesis_static_run_20260804T203108Z_215958` binds PLY SHA-256
`30ea567c52c4942a80f3e7e999ab4e0854681684143ad4e830f1e177bb860b0c`,
renders all eight views on the AMD APU with zero fallback, and reports mean
VkSplat time `23.319 ms`. Its metric gate uses the anisotropic Gaussian
two-sigma support envelope—not center-only point-cloud bounds—and measures
`94.873 mm`, only `0.127 mm` from the 95 mm anchor. Collision, mass, friction,
and contact continue to use the independent procedural proxy.

For an operator-facing read-only demo, the explicit candidate flag keeps the
pinned default intact. The presenter binds only to loopback; the leader process
uses haptic monitor mode and never writes physical output:

```bash
ONELOOP_OBSERVED_CORE_ROOT=/path/to/nonformal_candidate \
ONELOOP_LIVE_CANDIDATE_NONFORMAL=1 \
./ops/run_amd_real2sim_live_demo.sh \
  LEFT_PORT RIGHT_PORT LEFT_CALIBRATION_ID RIGHT_CALIBRATION_ID
```

To show the reviewed generated-fill ablation, point the same root variable at
the layered runtime bundle and replace the candidate flag with
`ONELOOP_GENERATED_FILL_ENABLED=1`. It is mutually exclusive with
`ONELOOP_LIVE_CANDIDATE_NONFORMAL` and remains off by default.

The launcher defaults to an 8 Hz dual-camera Gaussian view for 30 minutes and
opens the loopback presenter. Override `ONELOOP_LIVE_DURATION_S` for a bounded
recording and set `ONELOOP_RECORD_VIDEO=1` when an MP4 evidence artifact is
needed. Genesis remains authoritative at 120 Hz even if the renderer exits.

The dated hand-off state, exact run IDs, hashes, acceptance decisions, HIL
status, and remaining dynamic-integration gates are frozen in
[`../reports/progress_snapshot_2026-08-04.md`](../reports/progress_snapshot_2026-08-04.md)
and its machine-readable
[`../reports/real2sim_artifact_inventory_2026-08-04.yaml`](../reports/real2sim_artifact_inventory_2026-08-04.yaml).
