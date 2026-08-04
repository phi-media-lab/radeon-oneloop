# Radeon OneLoop progress snapshot — 2026-08-04

This document freezes the current engineering state of the dual-arm HIL,
Genesis, and Graffiti Mickey Real2Sim work. It contains both registered formal
Radeon object-asset evidence and explicitly nonformal integration evidence;
each claim is labelled at its own boundary. The machine-readable companion is
[`real2sim_artifact_inventory_2026-08-04.yaml`](real2sim_artifact_inventory_2026-08-04.yaml).

## Current outcome

The project now has the components required for a credible end-to-end demo.
The Gaussian appearance path is pose-synchronous and isolated from the 120 Hz
authoritative simulator; the remaining critical gate is calibrated physical
haptic bring-up followed by the recorded demo:

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
   trained and rendered in registered jobs on `radeon-c` GPU0/gfx1100. This is
   the content-addressed default visual asset; it makes no held-out-view claim.
5. MI300X SHARP and UniSHARP experiments produced useful pose, depth,
   appearance, and completion hypotheses. None of the generated branches has
   passed the release gate as metric truth.
6. The formal asset has been promoted to `amd`; static registration,
   gripper/table depth occlusion, a decoupled 10-second live gate, and a
   deliberate hard renderer-crash gate all pass against its exact hashes.
   Genesis Nyx is absent from the installed 1.3.1 package; pinned VkSplat/RADV
   is therefore the accepted nonformal renderer with a debug-mesh fallback.

The next release milestone is therefore **not another generator or renderer
experiment**. It is the calibrated `elbow_flex` single-joint haptic gate, then
single-arm and dual-arm low-torque expansion, followed by a short handover
recording. Physical output remains disabled until the operator re-attests that
a reachable power cut/emergency stop is present.

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

## Decision matrix

| Component | Current decision | Allowed role |
| --- | --- | --- |
| 30k canonical real-only VkSplat core | Accepted baseline | Default appearance and dynamic-binding input |
| Procedural rigid proxy | Accepted | Genesis collision/debug rendering |
| PBD plush-body proxy | Feasibility accepted, uncalibrated | Qualitative soft-body experiments only |
| VGGT-Omega | Accepted nonformal initializer | Pose, depth and generated-geometry audit |
| Apple SHARP | Accepted nonformal geometry proposal | Completion candidates and ablations |
| UniSHARP | Appearance accepted, geometry rejected | Pseudo-views and masked appearance proposals |
| 124,420 direct SHARP/UniSHARP hybrid | Rejected | Preserved negative control |
| 229,576 confidence-pruned fill | Conditional | Optional refinement initializer, default off |
| Hunyuan3D-2mv | Deferred | Non-blocking future comparison |

## Four-machine allocation and competition boundary

| Host | Current role | Evidence status |
| --- | --- | --- |
| `amd` | USB leaders, Genesis, HIL, haptics, physics integration | Development, `formal: false` |
| `phi-amd-work` / MI300X | VGGT, SHARP, UniSHARP, generation and pruning ablations | Development, `formal: false` |
| `radeon-f` | VkSplat/RADV training and render integration | Development, `formal: false` |
| `radeon-c` / GPU0 / `gfx1100` | Final pinned optimization, rendering and declared measurements | Only formal evidence host |

MI300X is deliberately used to shorten research iteration, but its checkpoints
and timing numbers are not silently promoted into the formal single-Radeon
lineage. The competition submission must still run with generated fill
disabled. Only jobs registered in `ops/formal_run_registry.yaml` and executed
on `radeon-c` GPU0 may populate formal result tables.

## Remaining critical path

1. Manually rotate `left/elbow_flex` from 93.538° into the read-only gate's
   -94° through 84° range, preferably around 60° rather than on the boundary,
   and rerun the corrected
   read-only preflight until it passes.
2. Re-attest the reachable physical emergency stop, clear the left elbow, and
   run one 10-second `left/elbow_flex` test at the unchanged 30/1000 torque and
   one-degree limits using the calibrated 0.6727447 effort scale.
3. Accept that gate only if watchdog/reject counts remain zero, output disables
   on exit, current/temperature/voltage remain inside bounds, and the operator
   reports a useful but comfortable resistance.
4. Generalize the already guarded renderer from one joint to one left arm, then
   to both arms; repeat monitor-only and time-bounded low-torque gates at each
   expansion. Do not increase the 30/1000 first-trial torque ceiling.
5. Record a short approach–grasp–handover–release demo with manifests, hashes,
   timing, safety state, and a terminal marker.
6. Package the formal object lineage, nonformal HIL safety evidence, and demo
   video as separate evidence layers; do not relabel the APU integration runs
   as formal Radeon measurements.
7. Capture additional real held-out views before claiming PSNR, SSIM, LPIPS,
   silhouette IoU, or novel-view quality.

## Verification status

The fresh full scaffold collects 182 tests and passes all available tests, with
5 OpenCV-dependent tests skipped in the local environment;
shell syntax, YAML/JSON parsing, and `git diff --check` also pass. These
source-tree checks are not a substitute for the physical haptic gate or the
recorded dual-arm handover demo.
