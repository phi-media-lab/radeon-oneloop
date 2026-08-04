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
5. **Pending manual reposition, hardware read-only:** the first read-only run
   `20260804T103609Z_180178_amd_haptic_readonly_preflight` passed electrical
   health and command-envelope checks but did not enforce a joint-limit margin;
   it is superseded. Corrected run
   `20260804T104004Z_180237_amd_haptic_readonly_preflight` added a fail-closed
   five-degree bidirectional model margin and rejected the current
   `left/elbow_flex` position of 93.538°; the permitted preflight range is
   -94° through 84°. Both runs performed zero register writes or torque-enable
   commands. Electrical state remains healthy: torque disabled, position mode,
   current 0, 34 °C, 7.3 V, and status 0. The pure command kernel reaches the
   expected 0.2-degree offset at 30/1000 and fails zero after a 101 ms
   synthetic timeout. Manually place the elbow at or below 84° and repeat this
   gate before physical output.
6. **Pending, physical:** use `left/elbow_flex` and an explicit candidate
   `simulated_effort_full_scale=0.6727447137236594`. This is
   `p95_effort / max_normalized_effort`, so p95 contact reaches the existing
   0.20 normalized ceiling. It does not increase the 30/1000 torque limit,
   one-degree pre-normalization offset, or ten-second duration. The default
   remains 3.35 until this test passes current, thermal, watchdog, shutdown,
   and subjective-resistance gates.
7. Expand only in this order: one calibrated joint, one arm, both arms. Add a
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

Steps 1–4 are complete for canonical PLY SHA-256
`0e26b6c4f993a7052fb471ad84a1a98180b262c868a4b179ce19b294b288bd1a`.
The latest normal and fault-injected integration gates are
`20260804T101926Z_173198_amd_decoupled_gaussian_live_gate` and
`20260804T102041Z_176664_amd_decoupled_gaussian_live_gate`; both retain
`physical_output=false`, approximately 120 Hz control, and zero watchdog
events. This ordering prevents an asset/coordinate defect or a renderer crash
from first being discovered while an operator-facing motor is energized.

The hardware read-only electrical checks pass, but physical step 6 remains
blocked by the failed joint-margin check and then requires a fresh operator
attestation. A previous command-line confirmation is not reusable:
the operator must freshly attest that the physical power cut/emergency stop is
immediately reachable and that the left `elbow_flex` sweep region is clear.

The simulation-only calibration command is:

```bash
./ops/run_amd_haptic_contact_calibration.sh
```

The hardware read-only preflight can be repeated without estop attestation
because it contains no motor write path:

```bash
./ops/run_amd_haptic_readonly_preflight.sh \
  LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID left elbow_flex
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

The bench wrapper cannot jump directly to motor output: it first runs the
read-only electrical, command-envelope, and bidirectional joint-margin gate in
the same evidence directory. The physical publisher is not started unless
that preflight exits successfully, and `haptic_bench_gate.py` independently
requires the exact preflight selection and calibration in the final `DONE`
decision.

After a machine-accepted physical run, progression still remains locked until
the operator reports both useful/comfortable resistance and free leader motion
after shutdown. `ops/seal_haptic_stage_receipt.sh` creates a separate immutable
receipt that binds the source `gate.json`, source hash index, and `DONE` marker;
it never mutates the physical run. A weak, strong, or uncomfortable verdict is
preserved as negative evidence and authorizes no next stage. Only an accepted
single-joint receipt authorizes `single_arm_monitor_only`; it does not authorize
single-arm physical output.

```bash
ONELOOP_LEADER_MOVES_FREELY_CONFIRMED=1 \
./ops/seal_haptic_stage_receipt.sh SOURCE_RUN_DIR useful_comfortable
```

The next stage is a 30-second, strictly no-output mapping exercise. It requires
the sealed single-joint receipt before either serial bus is opened, records the
range of every leader channel, and accepts only when all six channels on the
selected arm are deliberately exercised while the other arm stays quiet. The
Genesis consumer must also run at at least 100 Hz with zero input clamping,
zero watchdog events, bounded tracking error, and complete haptic-monitor
transport. Appearance rendering is intentionally absent from this safety gate;
the already-tested decoupled renderer remains the demo path.

```bash
./ops/run_amd_haptic_monitor_stage.sh \
  LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID left SINGLE_JOINT_RECEIPT_RUN_DIR
```

Even a machine-accepted monitor run cannot enable motors. The operator must
confirm that the selected real leader drove the same-side virtual follower in
the same direction and remained freely movable. A separate receipt binds that
judgment and authorizes only the five-joint read-only preflight:

```bash
ONELOOP_LEADER_MOVES_FREELY_CONFIRMED=1 \
./ops/seal_haptic_monitor_receipt.sh \
  MONITOR_RUN_DIR correct_same_side_same_direction

./ops/run_amd_haptic_arm_readonly_preflight.sh \
  LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID left MONITOR_RECEIPT_RUN_DIR
```

The arm preflight reads seven registers from each of the five non-gripper
motors and checks torque-disabled state, position mode, electrical health, and
bidirectional model-limit margin. It writes no register and uses a pure safety
kernel to prove the candidate 20/1000 torque, 0.5-degree offset, and 100 ms
watchdog envelope. A future physical single-arm runner must repeat this exact
preflight in the same run after a fresh estop/workspace attestation; a standalone
preflight result never authorizes physical output by itself.

The first five-joint physical stage remains a synthetic, deterministic bench
gate rather than a full live-contact demo. This isolates simultaneous bus
writes, cross-joint behavior, shutdown, and operator comfort before Genesis
contact dynamics are combined with two physical arms. It is limited to the five
non-gripper motors on one selected arm, 20/1000 torque, a 0.5-degree configured
offset (0.1 degree at the calibrated test effort), five seconds of output, and
the 100 ms feedback watchdog. The runner requires fresh estop and clear-arm
workspace environment attestations and repeats the five-joint read-only
preflight in the same run:

```bash
ONELOOP_PHYSICAL_ESTOP_CONFIRMED=1 \
ONELOOP_SELECTED_ARM_WORKSPACE_CLEAR_CONFIRMED=1 \
./ops/run_amd_haptic_arm_bench.sh \
  LEFT_PORT RIGHT_PORT LEFT_ID RIGHT_ID left MONITOR_RECEIPT_RUN_DIR
```

Machine acceptance still does not unlock dual-arm operation. The operator must
report useful/comfortable resistance, no cross-joint instability, and free
motion after shutdown in a separately hashed receipt. Only that receipt can
authorize `dual_arm_monitor_only`:

```bash
ONELOOP_LEADER_MOVES_FREELY_CONFIRMED=1 \
ONELOOP_NO_CROSS_JOINT_INSTABILITY_CONFIRMED=1 \
./ops/seal_haptic_arm_stage_receipt.sh \
  SINGLE_ARM_PHYSICAL_RUN_DIR useful_comfortable
```

Do not execute the physical bench command, or any future physical-arm command,
from automation without a fresh operator attestation that the power cut is
reachable and the selected workspace is clear to resist motion. The monitor
and read-only preflight commands above never satisfy that attestation.

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
