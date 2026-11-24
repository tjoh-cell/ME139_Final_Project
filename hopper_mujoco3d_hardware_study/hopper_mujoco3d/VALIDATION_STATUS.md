# MuJoCo 3D hopper validation before the stance-swing update

MuJoCo 4.0.1 ran the free-body twin-pogo plant from the original 0.5 m compression in calm air. All 18 pre-existing tests passed before editing. The revised suite passes 19 tests, including a powered multi-hop arrival with ascent-only thrust. The original source, MJCF, tests, and documentation are preserved in `../hopper_mujoco3d_original.zip`.

## Touchdown results

Each row uses the same guidance settings at both timesteps. Arrival requires a foot within 5 cm along x and inside the lateral corridor, with acceptable touchdown attitude and rates. Integration stops at that contact. Forward speed is the trunk speed at touchdown.

| Goal (m) | Feedback profile | Foot x at 2 ms (m) | Hops | Impulse (N s) | Peak z (m) | Arrival vx (m/s) | Foot x at 1 ms (m) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 15 | balanced | 14.9743 | 4 | 29.69 | 7.14 | +0.36 | 14.9629 |
| 30 | balanced | 29.9921 | 6 | 51.33 | 7.10 | +0.18 | 29.9891 |
| 60 | balanced | 59.9675 | 9 | 96.54 | 7.13 | +0.39 | 59.9629 |
| 120 | range | 120.0036 | 10 | 79.57 | 6.86 | -0.11 | 120.0016 |
| 240 | low_apex | 239.9892 | 41 | 91.96 | 5.37 | +3.90 | 239.9911 |

The 240 m low-apex profile uses a 1.8 m requested apex and a 0.44 foot-placement gain. The preloaded first hop can exceed the requested apex. At 1 ms, arrival took 44 hops and 97.35 N s, with a final vx of -0.16 m/s. A nearby gain of 0.45 used all 100 N s at 2 ms and failed the touchdown attitude check at 1 ms. These measurements establish sensitivity to contact integration. Wind and a stable stop after arrival remain untested.

## Controller and contact diagnosis

The original 15 m feedback run landed at 10.410 m with forward speed 4.45 m/s. Its next flight ended in a trunk strike at x = 15.687 m with pitch near -155 degrees and the motor at its 48 V rail. The original predictive planner landed four hops before its fifth search found zero safe candidates and fallback collided. Both runs are in `results_audit`.

The revised controller estimates the downward speed at touchdown and pitches the toes ahead of the trunk in proportion to horizontal approach velocity. The placement gain controls how much momentum carries through stance. Smaller gains and a lower apex can stretch the impulse budget. Touchdown scoring uses the contacting toes, and simultaneous collision or workspace-limit failures take precedence over arrival.

On the revised 15 m feedback run both toes contacted at each scored landing. The first powered landing was at 8.933 m and 3.47 m/s. With identical guidance and thrust disabled it was at 1.638 m and 0.76 m/s, followed by a backward hop and a body strike. The successful 15 m run peaked at 7.14 m, with maximum touchdown pitch of 20.28 degrees, negligible lateral displacement, and peak wheel speed of 190.47 rad/s. The motor reached 48 V and 50 A. The 190 rad/s wheel warning was exceeded. The motor model treats it as a warning threshold.

## Predictive planning

The 3D planner searches stance, flight, foot-placement and thrust references using branched MuJoCo simulations. It now includes full feedback and coasting in small search budgets, ranks landing distance monotonically, prefers a simulated goal immediately, and checks continuations from the stored landing state. It records the breakdown of rejected first-hop outcomes. The Stage 1.2 optimizer also searched separate thrust segments. That decision space has not been ported.

| Goal (m) | Predictive result | Last touchdown x (m) | Hops | Impulse (N s) |
|---:|---|---:|---:|---:|
| 15 | goal_tolerance_reached | 14.9983 | 4 | 24.98 |
| 30 | goal_tolerance_reached | 29.9692 | 4 | 30.55 |
| 60 | goal_tolerance_reached | 59.9532 | 7 | 60.56 |
| 120 | body_collision | 87.8369 | 12 | 100.00 |
| 240 | body_collision | 87.8369 | 12 | 100.00 |

Each predictive mission allowed five first-hop actions and up to three continuation actions per shortlisted landing. At 120 and 240 m, the planner exhausted the impulse budget, failed to find a safe next hop, and fallback physically collided at trunk x = 86.652 m. The successful feedback trajectories establish reachability under the same hardware limits.

## Reproduction

Run these historical commands from the extracted hopper_mujoco3d_before_stance_guidance.zip. The current default guidance and benchmark profile order are documented in README.md.

```powershell
python -m unittest discover -s tests -v
python benchmark.py --profiles 6 --output validation/benchmark_suite
python run.py --target 240 --foot-placement-gain 0.44 --apex-target 1.8 --dt 0.001 --output validation/repeat240
python run.py --target 15 --planner predictive --planner-budget 5 --next-hop-actions 3 --max-time 90 --output validation/repeat_predictive15
```

The benchmark CSV includes every attempted profile, with both final trunk x and last touchdown x. Each mission folder includes the trajectory, landings, planner diagnostics and plots.
