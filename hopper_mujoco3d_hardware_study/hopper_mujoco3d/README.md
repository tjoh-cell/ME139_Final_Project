# Twin-pogo hopper — Stage 2 MuJoCo 3D research prototype

For teammate hardware studies, open **Hopper Setup.pyw** or run **python view.py --setup**. Change components and goals, start another run and compare results in the same settings window. See the [hardware selection guide and parameter list](HARDWARE_SETUP.md).


## Crosswind control

For a 60 m destination and 7 m requested apex, run `python view.py --target 60 --apex-target 7 --wind-y 0.3`. Use `--gust-y 0.3` in place of steady wind for a zero-mean 0.5 Hz sinusoid, and add `--planner predictive` to watch the predictive run. The same settings are available in the hardware setup panel.

The ascent controller accounts for the flywheel's angular momentum when steering roll and yaw. When differential nozzle forces would exceed the elevation range, it requests extra common upward thrust within the actuator envelope. Predictive coasting reduces travel assistance while retaining lateral and attitude corrections. Predictive landing scores include lateral position, lateral speed and out-of-plane tilt.

Both controllers reached qualifying touchdowns in steady and sinusoidal winds of 0.3, 0.65 and 1.0 m/s at the default 2 ms timestep. At 1 m/s steady wind their peak lateral excursions were 0.47 m (feedback) and 0.52 m (predictive). The arrival tolerance applies at touchdown. See the [before/after mission measurements](validation/crosswind/comparison.csv) for individual results. These trials used the reference robot, constant wind or a single gust frequency and phase, and eight predictive candidates.

## Watch the stance swing and thrust

Use `python view.py --target 15 --planner predictive` to watch a predictive mission. The terminal reports each hop during preparation. The window then plays the recorded MuJoCo states at the requested `--speed`, including the applied nozzle states. For a live run where you can perturb the robot and see the planner respond, add `--live-planning`. In that mode the simulation waits while the planner evaluates candidate hops at touchdown. Feedback mode always runs live.

Run `python view.py --target 15` for the live viewer, or `python run.py --target 15 --gif --output results_3d/stance_swing_15` to export the same feedback gait. The overlay shows elapsed seconds, trunk x, target x, toe contact, and the applied force and body-frame gimbal angles for each nozzle. Elevation is measured from forward toward up. Positive lateral angle points thrust toward body +y. The gray cylinders point along the exhaust and the orange jets extend opposite thrust. Plume length exaggerates weak firing for visibility. Use `--no-thrusters` for a comparison.

The default feedback profile now requests a 3 m apex, uses a 0.55 foot-placement gain, and uses a forward thrust feedback gain of 0.25. Shallower placement lets the trunk rotate past the feet during stance. In the 15 m calm-air trial, the first landing and subsequent liftoff had pitches of -6.5 and +4.6 degrees. All six takeoffs had forward speed. The 15 m mission used 9.76 N s. A zero-thrust trial reached a furthest touchdown of 2.59 m before losing the gait.

Run `python benchmark.py --profiles 3 --output validation/stance_swing` to try the three new feedback profiles. To view the verified 120 m and 240 m trials individually, use `--apex-target 2.2` with `--foot-placement-gain 0.44` or `--foot-placement-gain 0.415`, respectively.

The [previous validation report](VALIDATION_STATUS.md) records results with the earlier feedback settings. The exact source and tests preceding this gait change are in `../hopper_mujoco3d_before_stance_guidance.zip`. The remaining notes below describe the original release.

## Read this before interpreting results

**This is a fresh 3D MuJoCo plant and a hardware-bounded 3D feedback controller; it is NOT a validated port of the previously successful Stage 1.2 joint MPC.** The 2D optimized action sequences cannot be assumed to work with two separate pogo feet, three-dimensional body dynamics, and MuJoCo contacts. The 15, 30, 60, 120, and 240 m distances must be re-tested with the benchmark runner. No outcomes have been pre-filled, replayed, or fabricated.

**Authoring environment limitation:** MuJoCo could not be downloaded into the artifact-building environment, so the native engine was not executed here. The source was syntax-checked and 9 pure/static tests passed; 2 MuJoCo integration tests were skipped. The benchmarks are provided for you to run. If `mjModel.from_xml_string` rejects the MJCF for your installed version, send the error before relying on any numerical results.

## Model

- World coordinates **x forward**, **y out of the old page**, **z up** (MuJoCo convention). This changes the old Stage 1 label, where vertical was sometimes `z` but lateral was not explicitly modeled.
- Free-floating 3D rectangular body with body-ground collision geometry and a quaternion orientation.
- **Two separate vertical pogo sliders and feet**, spaced on body y-axis. Each is passive: **225 N/m and 0.5 N·s/m**. In symmetric compression the combined rate is **450 N/m**, matching the 2D example. The two feet have **0.035 kg total mass**.
- Reaction wheel on a real hinge joint about body **y** (pitch axis), with an illustrative flywheel inertia of **0.003 kg·m²** and finite rotor mass of 0.03 kg. The rotor mass and 3D transverse inertia are new approximations, not exact equivalence to the previous planar Lagrangian model.
- The body has a 1 kg mass and pitch inertia of 0.015 kg·m². Spring starts pre-compressed by 0.5 m; no automatic preload/reset is inserted after landing.
- Coulomb friction and contacts use **MuJoCo's contact solver**. No manual velocity reset or foot pinning.
- Two independently commanded thrust vector forces with mounts at **y=±0.16 m** on the body. Each nozzle can command a magnitude, elevation angle (in body x–z plane) and lateral deflection angle (toward body y). These virtual gimbal forces act at the mount sites; **no massive/kinematically actuated gimbal CAD is included yet**. The housings are visual proxies.
- Wind is a world-relative airspeed: steady vector plus independent deterministic sinusoidal gusts. It produces quadratic drag force at a position offset above the body center, so crosswind can create body moments.

## Motor / thruster limits

The DC flywheel motor uses the **same assumed** 48 V, 0.16 Ω, 0.05 Nm/A, 0.05 V·s/rad, 5:1 gearbox, 90% efficiency and 50 A current limit as Stage 1.2. A torque request is reduced by back EMF and voltage/current limits; `190 rad/s` remains a **warning**, not a motor's actual mechanical hard limit. Thermal/inductance/gearbox limits remain unmodeled.

The provisional independent nozzles each have up to **8 N**, total impulse budget **100 N·s** (change via CLI), slew **120 N/s**, gimbal slew **9 rad/s**, elevation 0–160°, lateral deflection ±45°. These are simulation design assumptions, not selected hardware. No extra power is granted at long range.

Thrusters are **strictly off** on ground, in descent and at touchdown. They can apply thrust only when both toes are airborne, the body is ascending with positive vertical speed, the foot clearance is more than 3 cm, and predicted ballistic apex lies below the 8 m ceiling (with 0.12 m buffer). Any actual height-ceiling violation produces a failure status, never clamps vertical position or resets energy. A 7 m thrust cutoff is separate from the 8 m hard ceiling.

## Controller — what it does and doesn't do

The included baseline 3D controller is **state-feedback / gain-scheduled guidance**, not a new global 3D trajectory optimization. It requests forward velocity based on goal distance and a braking envelope; changes flywheel pitch target during stance; brings pitch upright during flight; and allocates ascent-only thrust between both nozzles to track forward/lateral motion, roll, yaw and a desired ballistic apex. It estimates the next touchdown position to begin braking earlier. Roll uses *differential vertical thrust* and yaw uses *differential forward thrust*. It can naturally command reverse thrust when projected to overshoot.

There are no unbounded idealized horizontal/vertical forces. Commands go through physical motor, two independent gimbal/throttle envelopes, energy/impulse and contact constraints. There is **no thruster correction on descent**, so strong crosswind may make the landing unrecoverable. This is an intended limitation of your requirement, not a silently enabled stabilizer.

A simple outer benchmark can try 1–4 **different controller settings** with unchanged hardware. It must re-simulate every run and labels failures as failures. This is **not** a port of the optimized 2D multi-hop action-search engine. Before claiming that the 3D plant reaches 240 m as well as the 2D version, a roll/contact-aware multi-hop shooting MPC should be built and measured against these runs.

## Install

Python 3.11+ recommended. On Windows:

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

To inspect the MJCF in the MuJoCo interactive viewer:

```bash
python view.py --target 15
```

Alternatively load `robot.xml` in MuJoCo Studio. The interactive viewer runs the same guidance policy as `run.py`.

## Simulation runs

```bash
# Initial 3D physics smoke test
python run.py --target 15 --max-time 35 --max-hops 30 --gif

# Same hardware, five missions, calm weather
python benchmark.py --targets 15 30 60 120 240 --profiles 1 --max-hops 120

# Broader CONTROLLER search; same plant/motor/thrusters for every goal
python benchmark.py --targets 15 30 60 120 240 --profiles 4 --max-hops 150

# Compare propulsion OFF versus ON
python benchmark.py --targets 15 30 60 120 240 --profiles 1 --no-thrusters --output baseline_no_thrust
python benchmark.py --targets 15 30 60 120 240 --profiles 1 --output assisted_thrust

# Crosswind: +y = out of former 2D plane (steady 5 m/s, gust 3 m/s)
python run.py --target 30 --wind-y 5 --gust-y 3 --gust-frequency .4 --gif

# Test stronger gusts and longer mission; no change to per-thruster power
python run.py --target 120 --wind-x -4 --wind-y 8 --gust-y 5 --max-hops 150 --gif

# Different mission impulse budget (not different thruster force ceiling)
python run.py --target 240 --mission-impulse-budget 100 --max-hops 160 --gif

# Generate/check MJCF without native MuJoCo
python run.py --target 15 --dry-run
```

If running headless on Linux with compatible drivers, `MUJOCO_GL=egl` may help for GIF capture. A rendering failure writes `gif_error.txt`; numerical CSVs, summary and plots remain saved.

## Output and acceptance criteria

`run.py` creates `results_3d/goal_<target>m` by default. Files: `robot.xml`, `summary.json`, `trajectory.csv`, `landings.csv`, `diagnostics.png`, `trajectory_3d.png`, and optional `hopper.gif`. `final_foot_x` and `position_error` use touchdown toe position; `final_x` is the root body x position. On failure it still writes everything possible and identifies `body_collision`, `unsafe_touchdown_attitude`, `height_ceiling_exceeded`, `lateral_excursion_limit`, `safety_hop_limit`, or `mission_time_limit`.

A benchmark result is **PASS only if**:

1. The **foot** lands within the 5 cm goal tolerance, in the lateral plane corridor, not merely if COM x momentarily crosses the target.
2. No body-ground collision occurs along the way, and modeled touchdown roll/pitch and angular rates pass their limits.
3. It stays under the configured 8 m body-origin ceiling and 100 N·s impulse budget, and motor/force/gimbal limits remain active.

Successful *arrival* does not imply zero forward velocity or a stable resting state. The model checks a 20° roll bound and 30° pitch bound at touchdown. The engine's contact solver is approximate and should be checked under timesteps 0.002, 0.001 and solver convergence variations before CAD parameter selection.

## Current verification status

| Test | Status when packaged |
|---|---|
| MJCF XML well-formed, representative 2-leg structure | PASS (static XML parse) |
| Roll/yaw differential thrust cross-products | PASS |
| Independent throttle/gimbal force projection and impulse | PASS |
| Flight-only thrust activation and height gate | PASS |
| Motor V/I/T static limit equations | PASS |
| Wind and reverse-guidance math | PASS |
| Native MuJoCo load/step | **NOT RUN** — package unavailable in authoring environment |
| 15 / 30 / 60 / 120 / 240 m 3D engine mission | **NOT RUN** — use `benchmark.py` |

Do not substitute earlier 2D landing results for these missing 3D tests. The first likely tuning task on your machine is restoring safe, repeatable multi-hop contact through the two-leg 3D plant, then validating lateral/roll recovery under wind. This is a representative first 3D prototype, **not final CAD/hardware certification**.
