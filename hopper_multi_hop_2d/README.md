# 2D Passive-Spring Hopper: Multiple Hops and Landing Dynamics

This is a **standalone follow-up to the earlier single-hop CasADi/IPOPT demo**. It does not modify, overwrite, or import the original `hopper_2d.py` or `run_mpc.py`. You can keep both folders side by side. It uses the same **2D physical state convention** but introduces repeated contact and spring compression dynamics.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m pip install -r requirements.txt
python run_multi_hop.py --target 0.45 --hops 5
python -m unittest test_hybrid.py -v
```

For a quicker run without animation:

```bash
python run_multi_hop.py --target 0.45 --hops 5 --no-gif
```

Adjust the task:

```bash
# Move left. Target is the desired final world x-position of the contact foot.
python run_multi_hop.py --target -0.45 --hops 5

# Less initial spring preload / more damping, to test limited hop energy.
python run_multi_hop.py --target 0.45 --compression 0.10 --damping 2.0

# Larger target, with no promise that the passive robot can reach it:
python run_multi_hop.py --target 1.00 --hops 8
```

The example creates `results_multihop/multi_hop.gif`, `multi_hop_plots.png`, `trajectory.csv`, and `summary.txt`.

## What runs and how

This program simulates the full repeated event sequence:

```
Initially precompressed spring
         |
         v
   STANCE (release)
         |  leg length reaches free length while extending
         v
   FLIGHT (ballistic COM, reaction-wheel attitude control)
         |  foot height crosses zero downward
         v
   MASSLESS POINT-FOOT TOUCHDOWN / IMPACT PROJECTION
         |  choose new world foot anchor
         v
   STANCE (compression, minimum length, rebound)
         |  return to free length while extending
         v
   FLIGHT -> TOUCHDOWN -> ...
```

The foot changes its world x-position after each landing, which is what makes cumulative hopping possible. There are no thrusters or active spring-compression actuators; the first hop begins with a precompressed passive spring. Subsequent rebounds come from landing energy and the flywheel motor may perform work during stance. Because damping and impacts remove energy, continued hopping is not guaranteed.

### Dynamics and state variables

Planar convention: alpha>0 tilts the leg/body toward +x. The flywheel angle phi is **absolute**, not body-relative.

- **Stance state:** `[alpha, phi, l, alpha_dot, phi_dot, l_dot]`.
- **Flight state:** `[xCOM, zCOM, vxCOM, vzCOM, alpha, phi, alpha_dot, phi_dot]`.
- `l0` = unloaded spring-leg length; `c` = axial damper coefficient.
- Leg and foot are massless. The body translates as a single mass `m`, with pitch inertia `Ib`, and the wheel has inertial spin inertia `Iw`.

The compressive spring–damper force is

```
Fs = max(0, k*max(l0-l, 0) - c*l_dot)  (while l < l0 or compressing)
```

It acts only in compression, never pulls the body down. In contact, the modeled equations are

```
(Ib + m*l²)*alpha_ddot + 2*m*l*l_dot*alpha_dot - m*g*l*sin(alpha) = -tau
m*l_ddot = m*l*alpha_dot² - m*g*cos(alpha) + Fs
Iw*phi_ddot = tau
```

The COM is at `[foot_x+l*sin(alpha), l*cos(alpha)]`. During flight, `xCOM_ddot=0`, `zCOM_ddot=-g`, and the wheel torque gives `alpha_ddot=-tau/Ib`, `phi_ddot=tau/Iw`.

### Touchdown / massless point-foot idealization

Touchdown is detected when the predicted **foot** height crosses downward through zero:

```
zCOM - l0*cos(alpha) = 0
```

The code places a new planted point at `foot_x=xCOM-l0*sin(alpha)`. Rather than adding a foot mass or arbitrarily resetting spring compression, it applies a **dissipative velocity projection** to the no-slip stance kinematics:

```
l_dot_plus = sin(alpha)*vx_minus + cos(alpha)*vz_minus
alpha_dot_plus = [m*l0*(cos(alpha)*vx_minus - sin(alpha)*vz_minus)
                  + Ib*alpha_dot_minus] / [m*l0² + Ib]
phi_dot_plus = phi_dot_minus
```

The radial velocity is retained for compliant spring compression, while the tangential/body rate is projected with respect to the mass/inertia matrix. The projection cannot add kinetic energy. If `l_dot_plus<0`, the spring compresses automatically and rebounds according to the same stance ODE.

**Major idealization:** This instantaneous sticking projection does **not** enforce the *impulsive* Coulomb friction cone. In the sample result, the simulated contact impulse would require more tangential than friction-supported impulse. The console prints a warning and the per-hop diagnostics include the deficit. Therefore the demonstration is a useful **reduced-order algorithm prototype**, **not a validated physically feasible landing**. For hardware-grade results, replace touchdown with a finite-duration compliant/slipping contact law or add realistic foot/leg mechanics and check impact impulses. The stance controller *does* project torque to satisfy nonnegative normal force and a friction-cone inequality at the model level.

### Takeoff

Release is approximated when the spring-leg reaches the free length `l0` while extending. The stance solver also computes ground normal and tangential reactions, enforces feasible torques, and reports failures. **This is not yet an exact complementarity-based normal-force separation model:** it can miss some early lift-off modes. The simplified geometry assumes the leg becomes rigid at `l0` in flight and does not need independent leg deployment control.

### Controller (important distinction from the original MPC)

The included `plan_tilt_one_hop()` is a **one-hop-ahead shooting controller**. Each time a new stance begins, it:

1. Chooses the next horizontal waypoint no farther than `step_limit` (default 0.17 m) toward the global target.
2. Evaluates several candidate stance **body-angle setpoints** (`--alpha-grid 9` default) by simulating that stance and its following flight.
3. Selects the candidate with lowest next-landing error, plus a small attitude/effort proxy penalty.
4. Uses stance PD torque control and flight PD attitude control, with torque and wheel-speed limits.
5. Repeats after touchdown with the updated global foot position and compressed spring state.

This is **not high-frequency receding-horizon CasADi/IPOPT NMPC**. Unlike the original single-hop MPC, it re-plans only at landing. A full multi-phase NMPC would make the motor torque history a decision variable and impose the stance/flight/impact dynamics in one hybrid NLP (or use mode schedules/complementarity), using this simulator as a validation plant. The original CasADi files are preserved independently for that future step.

### What energy can change without thrusters?

The initial spring preload supplies the first hop; subsequent descending motion compresses it again. Mechanical energy falls due to damping and inelastic impact. **The reaction wheel's electric motor is an external electrical energy source** to this mechanical system: changing wheel and body speeds can add mechanical energy. The plotted total mechanical energy includes wheel spin, so it may rise from motor work even without thrusters. Turn off actuation (`Robot(tau_max=0)` in a Python script) to test passive loss-only hopping.

### Limitations / next physics improvements

- The spring is a massless telescoping leg; no leg rotational mass, linkage clearances, or finite-size foot.
- Body pitch is aligned with the leg in stance; the pitch constraint reappears through the idealized impact reset.
- No accurate frictional touchdown/sliding transition, no unilateral contact complementarity, and no ground compliance.
- The foot is assumed planted throughout stance; stance motor torque is clipped to the modeled Coulomb cone, but touchdown impulse feasibility is not enforced.
- No passive leg retraction/deployment dynamics during flight; leg length stays `l0`.
- No thrusters, air drag, 3D dynamics, electrical motor model, or robust state estimation.
- The controller is deliberately simple, and grid search can miss good feasible control laws. Long-distance goals may be unreachable without thrust and sustained energy input.

## Source code map

| File | Purpose |
| --- | --- |
| `hybrid_hopper.py` | Parameters, stance/flight ODEs, reaction-wheel controls, spring damping, impact projection, hybrid events, one-hop lookahead controller |
| `run_multi_hop.py` | CLI, trajectories, CSV, diagnostic plots, GIF animation |
| `test_hybrid.py` | Tests for touchdown reset, passive-energy loss, and repeated hopping |
| `requirements.txt` | Python dependencies |

Run `python run_multi_hop.py --help` for options.
