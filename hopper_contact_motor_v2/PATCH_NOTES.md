# Hopper contact/motor V2 — active recovery and full-hop viability patch

This is a **drop-in patch for `hopper_contact_motor_v2` with the prior recovery patch installed**. It intentionally does **not** modify `physics.py`, the spring/motor parameters, or contact model. Back up the project, then overwrite **only**:

- `planner.py`
- `simulator.py`
- `run.py`
- `test_sim.py`

Keep `physics.py`, `visualize.py`, `requirements.txt`, your output directory, etc. intact.

## What's different

1. **Active emergency control:** if no safe goal-oriented plan is found, the planner runs a secondary bounded stance/flight reference optimization using the *same real motor and contact model*. It chooses the least unsafe predicted next hop, prioritizing collision avoidance, pitch/rate, and remaining horizontal momentum. The motor controller remains ON. Previously, the fallback disabled all gains. An emergency hop may still crash or fail to take off: there is no guaranteed recovery.
2. **Full next-hop assessment:** shortlisted normal trajectory candidates are evaluated from their post-impact state through a **complete additional stance, takeoff, flight, and touchdown** under several candidate reference pairs. The previous short probe stopped when the next takeoff occurred. A demonstrated safe next hop is a positive witness only; failing these samples does *not* prove that no feasible continuation exists.
3. **Recovery-aware momentum:** penalizes the center-of-mass speed relative to the goal after **intermediate as well as final** landings, plus post-impact spin. This helps discourage aggressive landings with no control margin.
4. **Computational strategy:** full next-hop probes only run on `--continuation-candidates` top-ranked candidates per horizon, not on every derivative-free optimizer evaluation. Defaults are 1 candidate and 3 complete-hop reference trials; larger values are slower and may find more options.
5. **Logging:** `summary.json` records planned references, probe viability, active emergency mode and predicted event. Console and `motor.csv` reveal motor usage during emergency recovery. Existing physical failure status remains honest: a crash is still a crash.

## Installation / regression

```bash
python -m unittest -v test_sim.py
python run.py --target 3 --compression 0.4 --lookahead 2 --maxfev 12 --max-hops 8 --gif
```

This patch's **16 unit/regression tests passed** in the creation environment. A 3 m simulation reached 3.018 m in four hops with a 2-hop horizon, 12 Powell evaluations, and 3 full-hop probe references. The result can change with timestep and platform.

## Repeat the longer-target test

```bash
python run.py --target 15 --compression 0.5 --lookahead 4 --maxfev 40 --max-hops 10 --gif
```

For a deeper (more computationally demanding) check:

```bash
python run.py --target 15 --compression 0.5 --lookahead 4 --maxfev 80 \
  --continuation-candidates 2 --continuation-probe-count 5 \
  --emergency-maxfev 20 --max-emergency-hops 2 --gif
```

The long-distance target is **not proven reachable** with these parameters. The controller may still report `active_emergency_failed:body_collision`, `no_feasible_plan_after_recovery_search`, or a safety limit. There is no claim of global optimality or exhaustive reachability testing.

## New switches

| Flag | Default | Meaning |
|---|---:|---|
| `--continuation-candidates` | 1 | Number of shortlisted trajectories tested with complete next-hop rollouts per horizon |
| `--continuation-probe-count` | 3 | Number of next-hop reference pairs tested for each shortlisted state (max 5 currently) |
| `--emergency-maxfev` | 10 | Maximum Powell budget for secondary emergency reference search; it also tests a 3×3 grid of seeds |
| `--intermediate-speed-weight` | 0.45 | Weight applied to arrival-speed penalties on intermediate landings |
| `--max-emergency-hops` | 2 | Maximum emergency attempts |
| `--no-emergency-fallback` | off | Stops when normal optimization finds no feasible plan |
| `--no-full-next-hop` | off | Use old short probe instead of complete next-hop checks |
| `--no-terminal-probe-rejection` | off | Allows candidates even if tested continuation controls fail (less conservative) |

Old `--no-passive-fallback` and `--max-passive-fallbacks` are kept as aliases for compatibility, **but recovery is no longer passive**. `--recovery-probe-time` only matters if full-hop checking is disabled.

## Limitations

- A safe full next hop does not prove the robot can keep hopping indefinitely or eventually reach the final goal.
- Emergency search optimizes **angle references**, not every motor torque. Motor control still uses the original PD + voltage/current-limited model.
- All searches are local/finite-sample and the contact solver is an approximation. Increasing the horizon or search budget does not guarantee a feasible plan.
- Body collision, slip, and finite-step impact errors still require convergence studies before hardware choices.
- This is not an IPOPT or CasADi NLP; it retains the previous bounded Powell reference optimization.
