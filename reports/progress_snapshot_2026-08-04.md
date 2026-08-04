# Radeon OneLoop progress snapshot — 2026-08-04

This document freezes the current engineering state of the dual-arm HIL,
Genesis, and Graffiti Mickey Real2Sim work. It contains both registered formal
Radeon object-asset evidence and explicitly nonformal integration evidence;
each claim is labelled at its own boundary. The machine-readable companion is
[`real2sim_artifact_inventory_2026-08-04.yaml`](real2sim_artifact_inventory_2026-08-04.yaml).

## Current outcome

The project now has a measured end-to-end Real2Sim demo path. The Gaussian
appearance process is pose-synchronous and isolated from the 120 Hz
authoritative simulator. Physical force feedback remains a useful extension,
but it is deferred and is no longer on the submission critical path:

1. Two USB SO-101 leaders can drive two virtual SO-101 arms through the live
   Genesis bridge on the `amd` APU host.
2. A bounded single-joint haptic output path has completed two 10-second,
   low-torque hardware trials. A subsequent millimetre-scale Genesis sweep
   explains why those `wrist_roll` trials felt weak and selects
   `left/elbow_flex` plus a `0.6727447` effort scale for the next guarded trial.
3. The handover object is correctly identified as the 95 mm Graffiti Mickey
   pendant. Its procedural Genesis representation is a stable physics/debug
   proxy, not a reconstructed appearance asset.
4. A real-photo-only, canonical 30,000-Gaussian object appearance has now been
   trained and rendered in registered jobs on `radeon-c` GPU0/gfx1100. It
   remains the content-addressed formal anchor-view baseline and makes no
   held-out-view claim. A new continuous-orbit audit rejects it as the final
   live-demo asset because intermediate angles show doubled surfaces and edge
   tearing.
5. MI300X SHARP and UniSHARP experiments produced useful pose, depth,
   appearance, and completion hypotheses. None of the generated branches has
   passed the release gate as metric truth.
6. The formal asset has been promoted to `amd`; static registration,
   gripper/table depth occlusion, a decoupled 10-second live gate, and a
   deliberate hard renderer-crash gate all pass against its exact hashes.
   Genesis Nyx is absent from the installed 1.3.1 package; pinned VkSplat/RADV
   is therefore the accepted nonformal renderer with a debug-mesh fallback.
7. A second single-Radeon run freezes the deterministic 95 mm visual-hull
   geometry and fits only observed-photo color/opacity. Its explicit nonformal
   candidate passes a 72-frame continuous 360-degree audit and a 12-second
   read-only dual-leader live gate with no Gaussian fallback.
8. The Vista4D-specific ablation includes a procedural 95 mm carrier,
   real-photo projection, measured mask alignment, and two fixed-seed MI300X
   runs. The carrier is now explicitly rejected as a complete-geometry prior:
   color projection did not correct its distorted shape, and both videos also
   lose identity. The branch is preserved only as failure evidence.
9. The independent Hunyuan seed-`10030` fallback now completes the full
   observed-core/generated-fill separation and Genesis static-binding path.
   It passes the no-harm gate only as a default-off nonformal toggle; it does
   not replace the SEVA primary branch, whose 49-frame orbit is now generated
   and audited but still blocked on explicit human identity/temporal review.
10. The final read-only handover recorder now evaluates task semantics rather
    than runtime health alone. A preserved stationary-input trial proves that
    it rejects missing arm/gripper motion and a missing left-grasp → dual-contact
    → right-hold sequence even when control and rendering remain healthy.

The next release milestone is a reproducible package of the real-photo input,
geometry-frozen single-Radeon build, continuous-orbit review, and dual-leader
live recording. Additional real held-out photographs are still required before
novel-view quality metrics or formal promotion. The force-feedback branch is
preserved but stays disabled and does not block this package.

## What is accepted now

### Dual-leader HIL and Genesis

- The serial publisher and Genesis consumer are separated into two processes,
  with a versioned UDP contract, sequence checks, joint clamping, watchdogs,
  and no physical follower commands.
- The checked-in scene contract now places the two followers side by side,
  parallel, and facing the same direction. Leader-to-follower left/right
  mapping is explicit. The earlier parallel-layout smoke exposed the swapped
  mapping and therefore does not close the post-fix interactive gate.
- A monitor-only bridge run completed 960 Genesis steps at a requested 120 Hz,
  accepted 425 packets with no rejects, and triggered no watchdog event.
- One-button virtual reset and the physical-output safety shutdown path are
  implemented.
- Two guarded `left/wrist_roll` haptic bench runs completed at a maximum
  `30/1000` torque limit, one-degree offset, and ten-second output duration.
  The second run accepted 302 feedback packets with no rejects, peaked at
  33 °C and raw current magnitude 1, and ended with output disabled.

The two physical haptic results are safety/transport gates only. The operator
reported weak feedback because those runs targeted `wrist_roll`. Accepted
simulation calibration run
`20260804T085549Z_165620_amd_haptic_contact_calibration` instead sweeps the
object from 2 mm clearance to 3 mm penetration against the left gripper. The
negative-clearance force is exactly zero; stable 1–3 mm contact produces about
5.72–11.05 N; the right arm remains below the 0.5 N isolation threshold; and
no gripper solver limit is hit. The p95 simulated reaction is 0.13455 on
`left_elbow_flex`, 0.06421 on `left_shoulder_pan`, and only 0.00066 on
`left_wrist_roll`. Mapping the strongest p95 to the existing 0.20 normalized
cap gives a candidate effort full scale of `0.6727447137236594`. This candidate
is configurable but is not the default and has not yet authorized motor output.

Hardware read-only run
`20260804T103609Z_180178_amd_haptic_readonly_preflight` opened both calibrated
leader buses through the monitor connection, confirmed healthy electrical
state, and exercised the pure command envelope with zero register writes. It
was then superseded because it did not enforce distance from joint limits.
Corrected run `20260804T104004Z_180237_amd_haptic_readonly_preflight` adds a
five-degree bidirectional margin and fails closed: the current
`left/elbow_flex` value is 93.538°, outside the allowed -94° through 84° range.
All other checks pass: torque disabled, position mode, current 0, temperature
34 °C, voltage 7.3 V, status 0, 0.2-degree candidate envelope at 30/1000, and
101 ms watchdog fail-zero. It still issued zero register writes and no torque
command. Failed-run metrics SHA-256:
`1a55238d339b7d5f039c064e7f6cac9c7f91698025a66eaf8a8f41aa859eee9a`.
Latest repeat `20260804T112411Z_180658_amd_haptic_readonly_preflight`
reconfirms the unchanged 93.538° position and the same single failed check.
Its metrics SHA-256 is
`11d61a61dde86b3e1e8025d0971f0f0613ab83152b600e0795ea37f9de99e590`
and its hash-index SHA-256 is
`bb6beeccdb14cb6ab3719ffd69946baaba63f5bac336d0712c90d3ac02e2b005`;
both serial devices were free after exit.
A torque-free live watcher subsequently observed the calibrated elbow moving
continuously from 96.791° to 61.187°, inside the accepted range, with no register
writes or torque command. After the watcher disconnected and the operator
released the light arm, gravity returned it to 96.615°; standalone run
`20260804T113715Z_180829_amd_haptic_readonly_preflight` therefore failed only
the same margin check. Its metrics SHA-256 is
`825f54fc721440d80fedad977fb4976899e72f9637d0dbf07b01f022d811fb8f`.
This proves that the calibration is valid and isolates the remaining failure to
the process handoff between manual positioning and motor arming.

The current single-joint physical bench runner now removes that handoff by
reusing the existing HIL intervention semantics in the already-open publisher.
After fresh estop and clear-workspace attestations, it waits torque-free until
the selected `elbow_flex` remains inside its bidirectional margin for 0.4
seconds with no more than 2 degrees of span. In that same serial process it then
reads the motor's seven health/mode registers, freezes a zero-write preflight
boundary, and only starts the synthetic feedback sender after a ready marker.
The first feedback packet causes a final current-pose check and immediate
low-torque arming; no Python process restart occurs for gravity to exploit. The
final bench gate now requires this same-process intervention evidence in
addition to the prior envelope, health, transport, watchdog, and fail-zero
evidence. The same state machine is also wired to the later five-joint runner,
where it requires all five selected joints and all 35 registers.

Deployment probe `runs/haptic_intervention_dryrun/zero_output_v2_ldTD4d` exercised
the refactored calibrated dual-leader read path on `amd` for 60 samples at
29.898 Hz with zero send errors, zero output commands, and
`physical_output_commands=false`; both
serial devices were free after exit. Metrics SHA-256:
`18ccabeb9e403258c3e6ef2cd6e7a2c72cc70617ca6770b864dab34ad8367c1f`.
The intervention implementation is software-tested and deployed but has not
yet issued a physical command.

Post-run operator perception is also fail-closed: a separate content-addressed
receipt must bind the accepted gate, source hash index, `DONE` marker, a
useful/comfortable verdict, and free leader motion after shutdown. That receipt
can unlock only single-arm monitor mode, never physical single-arm output.
The subsequent single-arm monitor stage is now executable but has not been run:
it requires the receipt before opening either bus, records per-channel motion
ranges, enforces one exercised arm plus one quiet arm, and accepts only a
100-Hz-or-faster unclamped Genesis run with zero watchdog and zero physical
output. Its own operator mapping receipt can unlock only a five-joint read-only
preflight. That preflight checks all non-gripper motors at a candidate 20/1000
torque and 0.5-degree envelope; the physical runner repeats the checks at the
same-process intervention boundary after a fresh estop/workspace attestation.
The corresponding five-second single-arm physical runner, machine gate, and
post-run operator receipt are implemented and software-tested but deliberately
unexecuted. They use reliable per-motor writes and readback for arm/release/
restore, batched bounded writes only in the 30 Hz loop, and allow the operator
receipt to unlock only dual-arm monitor mode. No multi-motor physical claim is
made until that staged real-hardware run is completed.
The dual-arm zero-output continuation is also implemented and software-tested:
a READY-bounded monitor run requires motion coverage on all twelve leader
channels and the fixed same-side parallel layout, a separate operator receipt
binds the visual mapping judgment, and a ten-motor read-only preflight checks
both buses. The dual physical adapter is intentionally not implemented before
the single-arm empirical receipt determines a safe and useful scale.

### Physics and debug object

- The corrected Graffiti Mickey procedural asset uses the standard asymmetric
  ears, pink face, white plush body, white hands, and yellow shoes.
- The rigid convexified proxy is the real-time collision representation.
- A separate closed plush-body PBD proxy built as 1,303 particles on the AMD
  backend. It demonstrates Genesis/Taichi feasibility but is not a calibrated
  plush-material model.
- Gaussian density, generated meshes, and photogrammetry surfaces are not used
  directly as collision geometry.

### Formal real-photo Gaussian baseline

The default appearance is now the registered, real-only VkSplat lineage on the
declared Radeon host:

- input: four masked real anchor photographs plus a deterministic CPU visual
  hull initializer; no learned depth, generated view, generated geometry, or
  secondary-accelerator artifact enters the formal dataset;
- dataset SHA-256:
  `682b65e97653ffe08e469496bb0554f349aeff103ddf8e57f1e4857f8c04534e`;
- formal training job:
  `20260804T095251Z_gaussian_train_84d468b_20260804`, 2,000 steps and 30,000
  splats in 18.203 s, with 123,231,612 bytes reported peak Vulkan memory;
- trained PLY SHA-256:
  `d95edcb66edd5fd3f6fe3fda4686dfe718ce867507a9c4db0fa6dde88a2cfcc5`;
- formal render/canonicalization job:
  `20260804T095859Z_gaussian_render_c8bd111_20260804`, four 1024×1024 renders
  in 1.201 s, with 35,649,308 bytes reported peak Vulkan memory;
- canonical PLY SHA-256:
  `0e26b6c4f993a7052fb471ad84a1a98180b262c868a4b179ce19b294b288bd1a`;
- cameras SHA-256:
  `050891df1cfc5ef33070f7ab6becdd168267e5951143523519601f38963cbc26`;
- canonical provenance SHA-256:
  `80efa4f5a98070395844205afa663ee8ca2975eda21e720c3cf785dcfd52bd02`.

Both formal jobs are registered in `ops/formal_run_registry.yaml`, use GPU UID
`0x153f7d55778ab659`, and bind the exact config, dataset, parent checkpoint,
commit, and VkSplat commit. The render audit reuses the four optimization
views. It verifies lineage, metric canonicalization, orientation, identity,
and renderer compatibility; it is **not** PSNR/SSIM/LPIPS evidence for held-out
or novel views. Two identically seeded preflights were numerically stable but
not byte-identical because Vulkan floating-point atomic accumulation order is
not deterministic, so bitwise checkpoint determinism is explicitly excluded.

Continuous-orbit run
`20260804T125613Z_184390_amd_gaussian_orbit_audit` is preserved as a negative
result. It closes exactly at 360 degrees, but 49 of 73 frames touch the image
boundary and visual review shows doubled faces/ears, tearing, and ghosted
surfaces between the four observed anchors. The registered asset therefore
remains valid for its declared anchor-view lineage and registration claims, but
is rejected as the final continuous-view demo appearance.

### Geometry-frozen formal successor and continuous-view candidate

The four Mercari photographs are views of the same Graffiti Mickey variant as
the physical handover object. To prevent four-view optimization from moving
the deterministic visual-hull shell into view-specific duplicate surfaces,
run `20260804T130751Z_1956764_radeon_c_object_geometry_frozen_preflight` keeps
Gaussian scales and quaternions fixed, holds center learning rates at a
numerically safe `1e-12`, disables refinement and higher SH, and optimizes only
observed-photo DC color and opacity. It uses the same real-only dataset hash
`682b65e9…`, performs 2,000 steps in 10.355 s on `radeon-c` GPU0/gfx1100,
peaks at 122,502,412 bytes reported Vulkan memory, and emits canonical PLY
SHA-256 `dc4de9a0a3f4dadf62a4c03c2be939a9b178d284254c9e833a2e19941e41793b`.

That preflight remains explicitly `formal=false`, and its two earlier failed
attempts remain preserved. A clean, registered successor now executes the same
geometry-frozen contract as formal training job
`20260804T204531Z_gaussian_train_14053af_20260804`: 2,000 steps in 10.389 s,
30,000 splats, 122,502,412 bytes reported peak Vulkan memory, and trained PLY
SHA-256 `e9be3a2df4c1ca7fcfddc86deee4c366a2f941f66a881e41d13367c329aff378`.
Formal render job `20260804T205059Z_gaussian_render_e149f01_20260804` binds
that exact checkpoint, emits canonical PLY SHA-256
`7f01c1e6d8253d7f15162e2cb51e18845676fa1015983266b7d356d9b21aa706`,
and renders the four 1024-square observed anchors in 1.176 s with 34,884,748
bytes reported peak Vulkan memory. Both jobs exclude generated fill, learned
secondary-accelerator artifacts, collision geometry, and held-out-quality
claims.

On `amd`, continuous-orbit run
`20260804T131459Z_193927_amd_gaussian_orbit_audit` renders 72 distinct 512-square
angles plus the repeated endpoint. It has zero cycle-closure RGB MAE, zero
border-contact frames, alpha support from 0.4390 to 0.6280, and mean per-frame
VkSplat time 25.127 ms. Visual review accepts shell continuity and the major
front/rear/ear features as materially better than the formal baseline, while
retaining visible side-angle blur as a four-view limitation. It is ineligible
for held-out-real metrics.

A private-data audit sampled 96 images from both cameras across eight reviewed
successful HIL episodes. It confirms the same doll and supplies task-domain
rear/top/tag/workspace evidence. It does not supply calibrated camera poses or
dense front-to-side coverage, so it is identity/domain evidence rather than
metric training geometry or held-out evaluation.

The exact registered successor has now also passed its own development audit,
so it no longer inherits the old checkpoint's orbit result by assumption.
Run `20260804T205804Z_217937_amd_gaussian_orbit_audit` binds canonical PLY
`7f01c1e6…`, closes with zero RGB error, has zero border-contact frames, and
renders in 25.108 ms/frame on average. Human review accepts the shell and
front/rear identity while retaining side-angle blur and stretch as an explicit
four-view limitation. The preceding failed run
`20260804T205629Z_217870_amd_gaussian_orbit_audit` is preserved: it exposed
that the AMD runtime still pinned the superseded formal hashes, which was then
corrected rather than bypassed.

Read-only dual-leader runtime gate
`20260804T205940Z_218151_amd_decoupled_gaussian_live_gate` binds the same exact
formal asset and passes at 119.999 Hz control and 7.667 Hz composed rendering,
with 186 Gaussian successes, zero fallback frames, zero watchdog events, and
zero physical-output commands. All 12 leader channels were stationary, so the
run proves integration only; it does not count as an approach, grasp, handover,
release, or task-success recording.

Final-task fail-closed run
`20260804T213311Z_228906_amd_decoupled_gaussian_live_gate` exercises the new
task recorder with stationary leaders. As expected it is marked `FAILED`:
control remains at 119.999 Hz, rendering at 7.666 Hz, all 184 Gaussian renders
succeed, and watchdog, stream, fallback, and physical-output counts remain
zero, but neither arm/gripper motion coverage nor the ordered contact sequence
passes. The task trace contains 360 no-contact samples, never reaches the
handover target, and is not relabelled as a successful demo. Contacts are
computed only between each gripper/finger pair and the target object; table,
other-arm, and upstream-link contacts cannot satisfy the task sequence. Gate
SHA-256 is
`62253651ccc4fe2a283c8d1732e5ec82c9bd333b9e96e2fb881fd39d425c8b29`;
the failed-run hash-index SHA-256 is
`ed8031bbc5d586ace2b7c337730b2bcbadca35c9124e2930c95a0b3f31626b3d`.

### AMD runtime appearance capability

Run `20260804T101510Z_167855_amd_gaussian_appearance_probe` verifies the three
formal asset hashes and checks `asset.formal=true` before renderer
initialization. The APU integration run itself remains `formal=false`. It
rendered the 1024×1024 front camera through VkSplat/RADV in 81.17 ms on the
first probe, with 35,649,312 bytes reported Vulkan memory. Its metrics SHA-256
is `dbd6b1cf2e7f84d16246827dad5091c632f7998b87fb13765ee7c1e5c941428a`.

The runtime transform contract is now explicit:

```text
T_camera_object_opencv =
  inverse(T_world_camera_opengl · diag(1,-1,-1,1)) · T_world_object_canonical
```

This value is passed to VkSplat as its object/world-to-camera matrix. The
default appearance is the formal real-only 30k PLY and generated fill remains
off. The `amd` runtime is development evidence: consuming a formal upstream
asset does not make the APU execution formal.

### Accepted Genesis-to-Gaussian integration gates

Static gate `20260804T101807Z_169784_amd_gaussian_static_gate` validates the
formal asset hashes, object-pose transform, and eight pose/camera renders with
no fallback. The metric check trims only 0.01% of Gaussian centers per tail
(three splats), rather than the old 0.5% that discarded boundary support. It
measures 95.269 mm against the 95 mm anchor: 0.269 mm error inside the unchanged
2.85 mm tolerance. Mean per-view VkSplat time is 22.48 ms and p95 is 24.36 ms.
Metrics SHA-256:
`101ebe541a109c18f35eb9625b6605f9dc72e42401cd2f306c69bd2d86c4fa46`.

Occlusion gate `20260804T101848Z_171506_amd_gaussian_occlusion_gate` proves
conservative Genesis depth compositing against the formal asset: gripper and
tabletop pixels replace Gaussian color with maximum RGB error zero, and valid
proxy depth fraction is 1.0. Its metrics SHA-256 is
`bcede577ec84453d5b8648b1cffcfdd1b5a8949aee4f249eee66bf8ba129b79d`.

The normal decoupled gate
`20260804T101926Z_173198_amd_decoupled_gaussian_live_gate` runs the
authoritative physics/control process for 1,200 steps in 10.000 s at
119.999 Hz with zero watchdog events and no physical output. The separate,
non-authoritative renderer consumes 300 state snapshots and produces 48
dual-camera composites at 4.800 Hz (96/96 VkSplat renders, no fallback). Gate
metrics SHA-256:
`fe0f272245ad42023937acc120cb4d58c99f1e622243fe460b7654de2b1432d8`.

The hard-failure gate
`20260804T102041Z_176664_amd_decoupled_gaussian_live_gate` terminates the
renderer with `os._exit(86)` after three combined frames. The authoritative
process still completes 1,200 steps at 119.999 Hz with zero watchdog events,
300 successful state sends, and no physical output. Gate metrics SHA-256:
`abc0536382b6b5fed3067d61044eda7ad4bd6b4400c74bc68c0fccb9014c106f`.

The geometry-frozen candidate is connected only through an explicit
`candidate_nonformal` switch; the pinned formal default is unchanged. Run
`20260804T131754Z_194185_amd_decoupled_gaussian_live_gate` reads both physical
leaders without motor output, executes authoritative Genesis control at
119.9994 Hz, sends 360 ordered visual snapshots with no watchdog or send
error, and renders 92 dual-camera frames at 7.6665 Hz. All 184 VkSplat camera
renders succeed with zero fallback. A loopback-only browser presenter publishes
all 92 frames and records 220 frame requests; the short MP4 is content-hashed.
Both the consumer and leader publisher report `physical_output=false`.

Earlier runs that rendered cached Genesis masks or coupled rendering into the
control loop are retained as superseded negative evidence and must not be cited
as accepted registration or performance results.

### Radeon-c formal lineage status

The readiness stage is closed. Clean upstream VkSplat commit `e26c2549…` was
used for both registered jobs on RADV NAVI31/gfx1100, under the GPU0 lock and
the declared host/GPU UID checks. Formal training manifest and metrics hashes
are `d27440b7…` and `6c8cd274…`; formal render manifest and metrics hashes are
`a74191f3…` and `7bfe11f6…`. Only their declared training/runtime and
same-view visual-QA claims may enter formal tables. The workspace Gaussian,
generated-fill branches, APU integration timings, held-out quality, policy
success, and physical HIL remain outside this formal object-asset claim.

## Generator findings on MI300X

### VGGT-Omega pose and depth

VGGT-Omega provided the accepted metric pose/depth initialization and the
surface evidence used to reject or prune generated geometry. The accepted
spatial audit is bound by manifest hash
`851e8b5a60b4d695d1513cc381ff7f3be255a5f1d44a4b1dc233ec90fccfb461`.

### Apple SHARP

Per-anchor Apple SHARP outputs contained 1,179,648 Gaussians each. Their
pixel-corresponded alignment to VGGT was substantially better than UniSHARP's
independent geometry: median residuals were approximately 0.97–2.39 mm with
approximately 0.28–4.49 degree rotation correction.

The two-source-supported SHARP geometry plus UniSHARP SH0/opacity hybrid
contained 124,420 Gaussians and passed its numeric metric gate, but its RADV
render retained exterior holes. Direct appearance-field donation is therefore
a preserved negative result: it cannot create missing surfaces.

### UniSHARP

The enhanced UniSHARP run completed in 43 seconds on MI300X and retained 80
lossless pseudo-views with exact local camera metadata. Its images preserve
local identity well, but its independent geometry required 10.7–16.2 degree
rotation correction and had 5.3–6.8 mm median residual against the accepted
metric frame. On a 95 mm object this error is material.

UniSHARP is accepted only for generated appearance proposals and pseudo-view
research. It is rejected for metric geometry, observed truth, and held-out
evaluation.

### Confidence-pruned completion

Relaxing SHARP support to a single source restored a recognizable shell with
260,942 Gaussians but also introduced floaters and duplicated ear surfaces.
VGGT depth/front-conflict pruning reduced it to 229,576 Gaussians:

- all 124,420 cross-source points were retained;
- 105,156 depth-consistent single-source points were retained;
- 27,892 single-source front conflicts were rejected;
- source-depth absolute error was 1.87 mm at p50 and 6.84 mm at p90.

The result is useful as an optional real-photo refinement initializer. It is
not the default asset and must remain independently disableable. Its PLY hash
is `9a2dac7e7c5b6380e96c016ee9d8d6893e0923007c7a4a62a64cfe749d8f2a63`.

### Vista4D appearance-completion A/B

The MI300X Vista4D interface experiment uses an exact object
adapter now exports 49 source/point frames, alpha and zero-motion masks,
target cameras, hashes, and a terminal marker.  The first fixed-seed run used
the observed Gaussian orbit for both source and point condition and completed
in 746 seconds.  It is visually stable and mildly smoother, but does not by
itself create metric geometry.

A second 726-second run held the seed and all model parameters fixed while
injecting the four reviewed SAM3 Mercari views at frames 0/12/24/37.  Those
four identities were preserved, but detail did not propagate to intermediate
views.  Temporal-delta residual p95 rose from 0.03697 to 0.09882, so sparse
keyframe injection is rejected as the next source-video construction and is
retained as a negative control.

The then-proposed procedural carrier stage was executed in AMD run
`20260804T153528Z_201147_amd_surface_carrier` constructs a complete
12,078-vertex / 24,044-triangle carrier at exact 95 mm height, fits it to the
four reviewed silhouettes (mean IoU 0.75497), and projects real-photo color to
82.12% of its vertices. These numbers characterize only the fitted procedural
carrier; they do not validate the photographed object's geometry. A measured
point-mask sweep selects alpha 0.50 as the
best tested alignment, raising mean point/carrier IoU from 0.68555 to 0.70619.

Two carrier-conditioned Vista4D runs then hold seed and prompt fixed.  The
default-mask branch reaches source silhouette IoU 0.82811 but is fuzzy and
overexposed.  The alpha-0.50 / CFG-3 branch improves source IoU to 0.83728,
background MAE to 0.02610, temporal residual to 0.10198, and closure to
0.05709, but worsens observed RGB MAE to 0.29321 and further blurs the face.
Both are rejected for depth lifting. Genesis conversion experiments are also
preserved as negative visual evidence: PLY is unsupported, vertex colors are
ignored, and texture/material GLBs show unacceptable breakup. The carrier is
not an offline candidate; it is an invalid complete-geometry prior and may not
enter the corrected generation mainline. The collision proxy and observed-
Gaussian runtime layers remain unchanged.

### Corrected four-view generation mainline

The reviewed four-view M1 artifact is now the sole identity authority. The
new geometry-free input contract emits four observed SEVA inputs, 49 ordered
target cameras, and Hunyuan3D-2mv `front/left/back/right` RGBA inputs. It
explicitly sets `geometry_input: null` and rejects procedural carrier use.
Stable Virtual Camera supplies the video-first novel-view prior;
Hunyuan3D-2mv supplies a learned complete mesh; real-view differentiable
alignment must pass before optional Vista4D reshooting or Gaussian
distillation.

### 2026-08-05 generated-lineage correction

Subsequent human review rejected Hunyuan seed `10027` despite its coarse
four-view silhouette fit. The mesh has material identity/topology distortion;
therefore its aligned/real-projected versions, the derived Vista4D video, the
hybrid COLMAP dataset, and all three Radeon-f VkSplat descendants are preserved
negative controls only. `gaussian/real2sim_quarantine.json` records exact
run IDs and hashes. Dataset construction, Vista4D inference, and VkSplat
training now run `gaussian.provenance_quarantine` before taking the GPU lock.
The old hybrid dataset was tested on both phi and Radeon-f and exits with code
65 before training.

A replacement Hunyuan sweep generated seeds `10028`, `10029`, and `10030`
directly from the four reviewed images. All have a single front face, two
correctly asymmetric ears, one rear strap, and a continuous oval body over the
reviewed 360-degree contact sheets. Seed `10030` has the best coarse alignment
(mean silhouette IoU `0.77664`, minimum `0.75134`) and is accepted only as a
generated conditioning proposal. Seeds `10028` and `10029` are valid but
superseded. The accepted review binds the orbit, aligned mesh, four-image
manifest, and the two private HIL rear/top identity exemplars; generated hidden
geometry remains non-metric and ineligible for collision or formal evidence.

SEVA remains the preferred four-image-to-orbit stage. Its fixed-revision
authorized installer, four-view generation, numeric audit, mandatory human
review boundary, and accepted-review-only pseudo-view builder are now deployed
on `phi-amd-work`. The installer accepts no token argument or environment
token and uses only a credential entered interactively with the Hugging Face
CLI. Authorized install run `seva_model_install_20260804T215409Z_1434707`
binds model revision `e538e251c1009e9a41cf8b7fee5f21332a1960de` to account
`fbsh96`. The upstream empty `config.yaml` is accepted only when its fixed-
revision size matches repository metadata; the nonempty safetensors checkpoint
remains mandatory. The upstream deleted SD2.1 VAE locator is replaced by a
reviewed patch pinned to the official byte-identical Stability AI VAE revision,
and its files are hash-checked before each run.

Run `seva_primary_20260804T222216Z_1436441` completed all 300 diffusion steps
and produced 49 frames in `4349.76 s`; its inference is preserved even though
the old recorder rejected SEVA's valid `3x4` camera matrices. Recovery pipeline
`seva_primary_recovered_20260804T233837Z_1438301` canonicalized those matrices
to homogeneous `4x4`, reran no inference, and completed the numeric orbit
audit. Real-anchor silhouette IoU is `0.981943` mean / `0.976890` minimum;
adjacent-frame foreground IoU is `0.943569` mean / `0.904279` minimum; and the
first/last cyclic seam foreground IoU is `0.961873`. The camera round-trip
maximum error is `5.24e-08`. Automatic promotion remains disabled: the video,
all-frame contact sheet, four real/generated anchor comparison, and private HIL
rear/top exemplars still require explicit human review before any pseudo-view
dataset is built. Prior invalid-credential and recorder failures remain
preserved negative engineering evidence. Every run declares
`credential_material_recorded=false`; no token is recorded in the repository
or command logs for the account-bound install and pipeline. Hunyuan seed
`10030` remains a controlled fallback, not a replacement for the preferred
geometry-free SEVA path.

### 2026-08-05 layered fallback and Genesis gate

The fallback experiment no longer uses a generated mesh as Gaussian geometry.
`observed_visual_hull_initialization_v1` exports 30,000 centers from only the
four reviewed real masks; its PLY SHA-256 begins `1138a85221bf`. Hunyuan seed
`10030` supplies appearance pseudo-views only, with centers, scales, rotations,
refinement, and higher SH frozen. The 5k and 15k candidates are visually
equivalent, so the 15k result is the terminal ablation rather than an argument
for more optimization.

The seed-`10030` Vista4D reshoot is rejected and quarantined as a descendant:
mean silhouette IoU is `0.433189`, minimum IoU is `0.183667`, mean observed RGB
MAE is `0.3543`, and visual review shows the front face persisting around the
orbit while ear/rear identity drifts. It is not a training source.

Observed-visibility pruning partitions the terminal VkSplat candidate without
mixing provenance. Of 30,000 generated-appearance Gaussians, 151 are invisible
from all four real anchors, 1,073 are visible from one, and 28,776 are visible
from at least two. The optional fill therefore contains 1,224 Gaussians; the
30,000-Gaussian geometry-frozen observed core remains byte-identical. The fused
preview PLY SHA-256 is
`30ea567c52c4942a80f3e7e999ab4e0854681684143ad4e830f1e177bb860b0c`.

The same-camera 49-view comparison passes the observed-anchor safety gate:
mean RGB MAE versus the observed core is `0.0016553`, worst anchor RGB MAE is
`0.0022200`, minimum anchor foreground IoU is `0.992553`, and mean orbit IoU is
`0.995814`. It is accepted only as
`optional_nonformal_toggle_default_off`; completion effectiveness is explicitly
inconclusive without held-out real views.

On `amd`, static run
`layered_fusion_genesis_static_run_20260804T203108Z_215958` loads the generated
layer through a dedicated `layered-preview` schema rather than weakening the
observed-only loader. Eight of eight VkSplat renders succeed with zero fallback
at mean `23.319 ms`. The metric gate now measures the anisotropic two-sigma
Gaussian support envelope rather than point-center bounds: robust height is
`94.873 mm`, only `0.127 mm` from the 95 mm anchor. The independent procedural
collision proxy remains active; no USB bus or physical output is involved.

## Decision matrix

| Component | Current decision | Allowed role |
| --- | --- | --- |
| 30k canonical real-only VkSplat core | Accepted baseline | Default appearance and dynamic-binding input |
| 30k geometry-frozen real-only candidate | Accepted nonformal demo candidate | Continuous orbit and live Real2Sim appearance; explicit opt-in only |
| Procedural rigid proxy | Accepted | Genesis collision/debug rendering |
| PBD plush-body proxy | Feasibility accepted, uncalibrated | Qualitative soft-body experiments only |
| VGGT-Omega | Accepted nonformal initializer | Pose, depth and generated-geometry audit |
| Apple SHARP | Accepted nonformal geometry proposal | Completion candidates and ablations |
| UniSHARP | Appearance accepted, geometry rejected | Pseudo-views and masked appearance proposals |
| 124,420 direct SHARP/UniSHARP hybrid | Rejected | Preserved negative control |
| 229,576 confidence-pruned fill | Conditional | Optional refinement initializer, default off |
| Vista4D same-source video | Accepted nonformal interface baseline | Reshooting execution proof; no completion claim |
| Vista4D sparse-real-keyframe video | Rejected source construction | Preserved negative control; identity impulses do not propagate |
| 95 mm real-textured surface carrier | Rejected geometry prior | Preserved failure control; never enters learned completion mainline |
| Vista4D surface-carrier videos | Rejected for depth lift | Preserved continuity/identity ablation; never promoted to Gaussian or mesh truth |
| Surface-carrier Genesis GLBs | Rejected visual fallback | Portable conversion evidence only; procedural debug proxy remains active |
| Four-view generator input | Accepted nonformal contract | Sole identity source for SEVA and Hunyuan branches |
| Stable Virtual Camera v1.1 | Generated and numerically audited; human review open | 49-frame orbit prior only; no pseudo-view promotion before explicit acceptance |
| Hunyuan3D-2mv seed 10027 | Rejected and hash-quarantined | Preserved negative control; all descendants blocked |
| Hunyuan3D-2mv seeds 10028/10029 | Valid but superseded candidates | Candidate diversity evidence only |
| Hunyuan3D-2mv seed 10030 | Accepted nonformal conditioning only | Vista4D/direct appearance A/B; never observed/collision/formal truth |
| Hunyuan seed-10030 Vista4D reshoot | Rejected and quarantined | Preserved identity-drift negative control |
| 30k observed-mask visual-hull initializer | Accepted | Geometry initialization for frozen-geometry appearance fits only |
| 1,224-Gaussian low-visibility fill | Conditional | Default-off nonformal appearance toggle; no metric/physics claim |
| 31,224-Gaussian layered preview | Static gate accepted | Genesis visual A/B only; observed core stays authoritative |

## Four-machine allocation and competition boundary

| Host | Current role | Evidence status |
| --- | --- | --- |
| `amd` | USB leaders, Genesis, HIL, haptics, physics integration | Development, `formal: false` |
| `phi-amd-work` / MI300X | VGGT, SHARP, UniSHARP, SEVA, Hunyuan3D-2mv, Vista4D generation and pruning ablations | Development, `formal: false` |
| `radeon-f` | VkSplat/RADV training and render integration | Development, `formal: false` |
| `radeon-c` / GPU0 / `gfx1100` | Final pinned optimization, rendering and declared measurements | Only formal evidence host |

MI300X is deliberately used to shorten research iteration, but its checkpoints
and timing numbers are not silently promoted into the formal single-Radeon
lineage. The competition submission's formal evidence path must still run with
generated fill disabled. Only jobs registered in `ops/formal_run_registry.yaml` and executed
on `radeon-c` GPU0 may populate formal result tables.

## Remaining critical path

1. Review the completed SEVA v1.1 video, all-frame contact sheet, real/generated
   anchor comparison, and private HIL rear/top exemplars. Record an explicit
   hash-bound accepted or rejected decision. Build the observed-initialized,
   frozen-geometry pseudo-view dataset only if that review accepts it.
2. Capture additional real views at inter-anchor angles and freeze them as a
   held-out set before the next optimization run. Use them for silhouette and
   photometric evaluation, not as hidden training views.
3. Record the final short approach–left grasp–dual contact–right hold/release
   demo with the exact formal asset. The implemented gate also requires motion
   on at least three joints per arm, both grippers exercised, 12 cm object
   transfer, target tolerance, no table-envelope drop, hashes, timing, safety
   state, and a terminal marker. Leader access remains read-only.
4. Package three evidence layers separately: registered single-Radeon build,
   nonformal continuous-orbit/live integration, and private HIL identity/domain
   audit. Do not relabel `amd` timings as formal Radeon measurements.
5. Improve presenter framing with an evidence-labelled object crop only if the
   unchanged full camera views remain alongside it; do not replace the real
   task-camera evidence with a beauty shot.
6. Resume the calibrated haptic branch only after the visual submission path is
   sealed. Any future motor-output run still requires a fresh estop/workspace
   attestation and its existing bounded stages.

## Verification status

The fresh full scaffold runs 264 tests and passes 251, with 13 dependency-
specific tests skipped in the local environment. Shell syntax, YAML/JSON
parsing, and `git diff --check` also pass. These source-tree checks are not a
substitute for held-out capture or the final recorded handover demo.
