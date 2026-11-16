# Replacement patch: safe touchdown and stance + flight trajectory optimization

## Use with

Your existing `hopper_multihop_trajectory_optimization` project. **Keep your existing** `legacy_run_multi_hop.py`, `animate_csv.py`, `test_hybrid.py`, and `requirements.txt`. These files are not replaced by this patch.

## Installation

1. Back up the original project folder (or use git to save your current revision).
2. Copy the four Python files in this patch directly over the matching four files **in the root of** `hopper_multihop_trajectory_optimization`:
   - `hybrid_hopper.py`
   - `trajectory_planner.py`
   - `run_optimized_hops.py`
   - `test_optimized.py` (updated tests)
3. In that folder run:

   ```bash
   python -m pip install -r requirements.txt
   python -m unittest test_optimized.py test_hybrid.py -v
   ```

4. Try the same scenarios as before:

   ```bash
   python run_optimized_hops.py --scenario aggressive --target 15 --compression 0.5 --lookahead 4 --maxfev 35 --stop-after-crossings 2 --tolerance 0.05 --safety-max-hops 10
   python run_optimized_hops.py --scenario aggressive --target 20 --compression 0.5 --lookahead 4 --maxfev 35 --stop-after-crossings 2 --tolerance 0.05 --safety-max-hops 10
   ```

   To skip GIF rendering while testing, add `--no-gif`.

## What changed

- **Per-hop flight angle optimization.** Each horizon `N` now jointly optimizes `2N` scalar references: `N` stance angles and `N` flight angles. They are tracked by the original torque-limited reaction-wheel PD feedback; this is still SciPy/Powell hybrid shooting, **not** a CasADi or IPOPT direct-torque MPC.
- **Touchdown safety is mandatory for accepted hops.** The simulator checks pitch measured relative to the upright configuration and pitch rate immediately before applying the sticking-foot impact reset. Defaults: `abs(pitch) <= 30 degrees`, `abs(pitch_rate) <= 2 rad/s`. Invalid landings cannot count as arrival or be selected as feasible plan candidates.
- **Degenerate and stalled hop detection.** Reject simulated flights shorter than 0.06 s or with a COM rise less than 0.025 m above takeoff height. Terminate a candidate if two consecutive hops make less than 0.01 m of horizontal progress. The live simulation reports no safe plan / stall instead of printing repeated zero-motion hops.
- **Reports.** CLI prints stance and flight references as well as touchdown pitch and rate for each valid hop. `results_optimized/optimization.json` contains the references, touchdown diagnostics and per-horizon feasibility reasons.
- **Faster planning integration.** Approximate rollouts use a coarser numerical timestep. Final executed hops still use the original tighter integration tolerances, and can be rejected at execution time if the accurate simulation violates any safety limit.
- **Previous behavior retained.** Each `N` is optimized separately; automatic hop count selection, backward hopping after overshoot, target tolerance, and goal-crossing stop rules remain. No thrusters added, no change to the spring/impact model.

## New CLI options

```
--max-touchdown-angle 30    # degrees
--max-touchdown-rate 2      # rad/s
--max-flight-ref 35         # degrees (bound on optimizer's flight setpoints)
--flight-kp 1.3             # flight PD proportional gain
--flight-kd 0.24            # flight PD derivative gain
--min-flight-time 0.06      # seconds
--min-flight-rise 0.025     # meters
--min-progress 0.01         # meters
--max-stalled-hops 2
--maxfev 35                 # default; can raise for better candidate search
```

For example, to demand a more upright touchdown:

```bash
python run_optimized_hops.py --scenario aggressive --target 15 --compression 0.5 --lookahead 4 --max-touchdown-angle 15 --max-touchdown-rate 1.0 --maxfev 60
```

A more restrictive attitude limit can make a previously achievable target infeasible; the solver will report that rather than accepting an unsafe touchdown.

## Validation performed on this patch

- `python -m unittest test_optimized.py test_hybrid.py -v`: 12 tests passed.
- In the included idealized model, the 15 m scenario above landed at ~14.987 m in 3 accepted hops, the 20 m scenario landed at ~20.002 m in 4 accepted hops, with touchdown pitch and rate within the configured limits. These are not guaranteed exact results on other Python/SciPy versions and do not demonstrate hardware feasibility.

## Critical limitations STILL PRESENT

- **No realistic body-ground collision geometry.** The attitude check is only at foot touchdown. The model does not yet detect an actual robot body striking the floor *before* the foot. Do not equate the new checks with collision-safe landing.
- **The ideal sticking-foot impact reset can violate Coulomb friction.** This remains a soft impact impulse penalty and a printed warning, not a physically feasible sliding/impact model.
- **Motor speed limits still use the wheel's inertial speed rather than motor-relative speed**, and there is no torque-speed curve or power limit. Logged relative wheel speed can exceed the nominal 160 rad/s limit. Do not size hardware from these runs.
- **Not guaranteed globally optimal.** Bounded Powell with a limited evaluation budget may fail to find good or safe solutions; increasing `--maxfev` and testing multiple starting conditions is recommended for difficult targets.
- **A safe touchdown with nonzero energy is a visualization stopping condition**, not a command that physically holds the robot upright forever after landing.
