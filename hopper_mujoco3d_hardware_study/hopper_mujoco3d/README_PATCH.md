# MuJoCo 3D Hopper — attitude + predictive-landing patch

## Installation

Make a copy of your current `hopper_mujoco3d` folder. Copy the files from this patch **over the matching paths in that copied folder**:

- `control.py` — continuous full-range planar pitch, SO(3) out-of-plane correction, predictive touchdown-pitch reference
- `simulator.py` — updated feedback, MuJoCo state branching for predictive rollouts, new diagnostics, multi-frame GIF renderer, and corrected `t_yaw` variable
- `predictive.py` **(new)** — bounded physically simulated candidate hopping search and second-hop safety probe
- `run.py` — new CLI flags
- `benchmark.py`, `view.py` — optional predictive-mode controls
- `tests/test_attitude_predictive.py`, `tests/test_predictive_mujoco.py` **(new)** — orientation and native-branch tests

**DO NOT replace `model.py` or `robot.xml`.** This patch does not change robot mass, foot geometry, spring stiffness, motor constants, friction, or thrust capacity.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v

# Fast diagnostic with corrected attitude feedback
python run.py --target 15 --no-thrusters --max-hops 8 \
  --gif --output results_3d/v2_feedback_unpowered

# New rollouts use SAME MuJoCo model/contact solver. More expensive.
python run.py --target 15 --planner predictive --planner-budget 10 \
  --next-hop-actions 3 --max-hops 12 --gif \
  --output results_3d/v2_predictive_15m

# Test wind, roll and independently vectored ascent thrust
python run.py --target 15 --planner predictive --planner-budget 8 \
  --wind-y 3 --gust-y 1 --max-hops 12 --gif \
  --output results_3d/v2_predictive_wind

# Run five actual MuJoCo missions, preserving physical limits
python benchmark.py --targets 15 30 60 120 240 --profiles 1 \
  --planner predictive --planner-budget 8 --max-hops 150 \
  --output benchmarks_3d_v2
```

Windows PowerShell: place each command on one line, or use PowerShell backticks instead of Bash `\\` line continuations.

Start with 15 m and only proceed to longer-distance benchmarks once contact/attitude behavior is validated. Predictive rollouts can be computationally expensive because each candidate actually steps MuJoCo over the next hop; the second-hop probe is bounded, not a proof of feasibility. The controller is **sample-based receding-horizon reference planning**, not the full continuous-torque CasADi MPC or a guarantee of 2D-equivalent distances.

## Changes and interpretation

- `pitch` in CSV is now the **full signed planar pitch** measured from body +z projected into the x–z plane; it handles >90° planar pitch correctly. `roll`/`yaw` are local out-of-plane *tilt errors*, not XYZ Euler angles; `outplane_tilt` is their physically meaningful combined magnitude. All angles in CSV are radians.
- Motor commands still pass through the original voltage, current, gear and back-EMF limits. Roll/yaw thrust commands still pass through the per-thruster force, gimbal-angle, slew and impulse constraints. Thrusters remain **off during stance and descent**.
- During flight the flywheel aims for a conservative attitude that anticipates angular drift before touchdown. This is a heuristic: the model can still crash if the hardware or geometry cannot follow it.
- `--planner predictive` tests up to `--planner-budget` stance/flight/thrust combinations in actual MuJoCo, then applies the safest low-cost feasible reference it finds. Selected candidates are tested for a further safe hop, where affordable. If none is found, it reverts to the motor-limited feedback controller; `planner_diagnostics.json` records the lack of found candidates (not a physical impossibility claim).
- New `attitude_and_legs.png` separately shows out-of-plane tilt, body angular rate, flywheel rate and both leg lengths. The original `diagnostics.png` shows continuous pitch and thrust. `planner_diagnostics.json` is written on every run.
- GIF encoder now uses distinct saved body states with explicit Pillow multi-frame output and a timestamp overlay, and verifies that the saved file has >1 frame. If unavailable due to the local OpenGL renderer, `gif_error.txt` will contain the issue; CSV and plots are still saved.

## Validation performed in the authoring environment

- 15 non-MuJoCo tests passed. They include tests for a planar 120° rotation generating **zero out-of-plane tilt**, SO(3) correction, motor saturation, force allocation, thrust budget and orientation setpoint bounds.
- Three native MuJoCo tests are provided but **not run** in the authoring environment: MuJoCo was not installed. Only your local MuJoCo runs can establish true landing/goal success and contact behavior.
- No 15 / 30 / 60 / 120 / 240 m successes are claimed for this patch until its real MuJoCo benchmark is executed.
