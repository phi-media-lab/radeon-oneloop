# Haptic feedback design

The live bridge carries a 30 Hz return stream from Genesis to the leader host.
Its default is **monitor-only**: no torque is enabled and no goal is written to
either physical leader.

## Signal path

1. Genesis advances the two SO-101 entities at 120 Hz.
2. Once every four steps, one scene-wide contact query identifies external
   contacts for each arm. Self-contact is excluded.
3. When an arm is in contact, its position controller reaction effort is
   sampled and negated to obtain the operator-facing reaction direction.
4. A versioned UDP packet carries the two contact-force totals and frozen
   12-DoF reaction-effort vector back at 30 Hz.
5. The leader process validates ordering and finite bounds, but currently only
   records the signal.

Genesis exposes resolved contact forces in newtons through `get_contacts()` and
per-DoF control effort through `get_dofs_control_force()`. The SO-101 leader's
STS3215 has position, velocity, PWM, and step modes plus a writable SRAM
`Torque_Limit`; it does not expose the Dynamixel-style `Goal_Current` register.
LeRobot's `SO101Leader.send_feedback()` remains unimplemented. Therefore the
proposed renderer is bounded position impedance, not direct torque control.

## Physical renderer — disabled by default

For each leader joint, the hardware adapter would set a goal a few degrees from
the measured position in the simulated reaction direction while applying a low
`Torque_Limit`. The checked-in pure safety kernel currently caps:

- feedback age at 100 ms before fail-zero and disarm;
- normalized reaction at 0.20;
- position offset at 3 degrees before normalization, with 0.025 slew per update;
- STS3215 torque-limit register at 80/1000;
- contact dead band at 0.5 N.

The original simulated-effort full scale of 3.35 remains the fail-safe default.
It is now known to be too large for the measured handover contact, and must not
be confused with a physical force calibration. A first-gate hardware adapter
is implemented for exactly one
non-gripper joint. It further caps torque at 30/1000, offset at one degree and
physical output at ten seconds. It monitors `Present_Current`, temperature,
voltage and status, fails torque to zero on a 100 ms feedback timeout or any
exception, disables the selected motor on exit, and restores the prior SRAM
torque limit only after torque is disabled.

The command-line estop confirmation is an operator attestation, not an
electrically monitored emergency-stop input. Do not run the hardware gate
unless a reachable physical power cut is present and the selected joint is
mechanically safe to resist motion.

## Bring-up gates

1. **Passed:** synthetic safety-kernel tests and AMD-GPU contact signal smoke.
2. **Passed:** dual-leader monitor-only round trip; 240/240 feedback packets,
   no rejects, and no physical output.
3. **Passed twice:** the left leader `wrist_roll` completed two ten-second
   physical bench runs at 30/1000 torque and one degree maximum offset. Both
   runs ended with torque disabled; the repeat run accepted 302 feedback
   packets with no rejects, peaked at raw current magnitude 1 and 33 °C.
   This validates the guarded output path, not useful force magnitude.
4. **Passed, simulation only:** run
   `20260804T085549Z_165620_amd_haptic_contact_calibration` held the object at
   nine controlled poses from 2 mm clearance to 3 mm penetration against the
   left gripper. Clearance force was 0 N; stable 1–3 mm contact was
   5.72–11.05 N; the right arm stayed isolated; and no gripper solver limit was
   hit. P95 reaction effort ranked `elbow_flex` (0.13455), `shoulder_pan`
   (0.06421), then the remaining joints. `wrist_roll` was only 0.00066, which
   explains the weak earlier trials.
5. **Pending, physical:** use `left/elbow_flex` and an explicit candidate
   `simulated_effort_full_scale=0.6727447137236594`. This is
   `p95_effort / max_normalized_effort`, so p95 contact reaches the existing
   0.20 normalized ceiling. It does not increase the 30/1000 torque limit,
   one-degree pre-normalization offset, or ten-second duration. The default
   remains 3.35 until this test passes current, thermal, watchdog, shutdown,
   and subjective-resistance gates.
6. Expand only in this order: one calibrated joint, one arm, both arms. Add a
   monitor-only gate before physical output at each expansion. Increase only
   after a measured force/current calibration. The current
   software ceiling of 80/1000 must not be raised during the first dual-arm
   trial.

## Why the physical gate is last

The dependency order is intentional:

1. Freeze the formal observed-object asset and its metric coordinate frame.
2. Bind that exact asset to the Genesis collision proxy and pass static and
   foreground-occlusion gates.
3. Run the 120 Hz authoritative control loop with the renderer in a separate,
   non-authoritative process, then force that renderer to hard-exit.
4. Calibrate simulated contact force to joint reaction effort without enabling
   any motor.
5. Only then enable one bounded motor for ten seconds, followed by one arm and
   finally both arms.

Steps 1–4 are now complete for canonical PLY SHA-256
`0e26b6c4f993a7052fb471ad84a1a98180b262c868a4b179ce19b294b288bd1a`.
The latest normal and fault-injected integration gates are
`20260804T101926Z_173198_amd_decoupled_gaussian_live_gate` and
`20260804T102041Z_176664_amd_decoupled_gaussian_live_gate`; both retain
`physical_output=false`, approximately 120 Hz control, and zero watchdog
events. This ordering prevents an asset/coordinate defect or a renderer crash
from first being discovered while an operator-facing motor is energized.

Step 5 remains pending. A previous command-line confirmation is not reusable:
the operator must freshly attest that the physical power cut/emergency stop is
immediately reachable and that the left `elbow_flex` sweep region is clear.

The simulation-only calibration command is:

```bash
./ops/run_amd_haptic_contact_calibration.sh
```

The prepared single-joint bench parameters are deliberately explicit rather
than new defaults:

```bash
ONELOOP_PHYSICAL_ESTOP_CONFIRMED=1 \
ONELOOP_HAPTIC_SIMULATED_EFFORT_FULL_SCALE=0.6727447137236594 \
ONELOOP_HAPTIC_BENCH_REACTION_EFFORT=0.1345489427447319 \
./ops/run_amd_haptic_bench.sh LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID \
  left elbow_flex
```

Do not execute this command from automation without a fresh operator
attestation that the power cut is reachable and the selected elbow joint is
clear to resist motion.

The runner writes separate publisher/sender metrics and evaluates them with
`haptic_bench_gate.py`. A `DONE` marker now requires at least 250 accepted
feedback packets, zero rejects and send errors, at least 250 bounded commands,
valid motor health, a verified torque-disable/zero-limit readback, and verified
restoration of the pre-run SRAM torque limit. Process exit alone is not a pass.
Operator-perceived resistance remains a separate human attestation and cannot
be inferred from these machine checks.

Primary references:

- [Genesis rigid contacts and forces](https://genesis-world.readthedocs.io/en/latest/user_guide/theory/rigid_collision/collision_contacts_forces.html)
- [LeRobot SO-101 hardware documentation](https://huggingface.co/docs/lerobot/so101)
- [Feetech STS3215 product specification](https://www.feetech.cn/Data/feetechrc/upload/file/20200611/6372749961523760249976542.pdf)
