"""Hybrid multiple-hop planner with per-hop stance AND flight references.

One independent bounded Powell optimization is solved for each horizon
N=1,...,lookahead. The optimization variables are
    [stance_ref_1,...,stance_ref_N, flight_ref_1,...,flight_ref_N].
The plant still uses reaction-wheel PD control, not direct-torque NLP/MPC.

Every simulated touchdown must satisfy body attitude and angular-rate limits.
Short/degenerate flights and repeated no-progress hops are rejected; a landing
near the target cannot count as success if its attitude is unsafe.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from hybrid_hopper import (
    Robot, Control, initial_state, simulate_stance, simulate_flight,
    impact_reset, impact_impulse_diagnostics, to_flight,
    foot_position_flight,
)


@dataclass(frozen=True)
class PlanningOptions:
    lookahead: int = 3
    safety_max_hops: int = 10
    max_crossings: int = 2
    arrival_tolerance: float = 0.05
    maxfev_per_horizon: int = 40
    hop_cost: float = 0.015
    tilt_cost: float = 0.005
    friction_cost: float = 0.0003  # still SOFT: friction impact model unchanged
    landing_angle_cost: float = 8.0
    landing_rate_cost: float = 1.0
    max_touchdown_angle_deg: float = 30.0
    max_touchdown_pitch_rate: float = 2.0  # rad/s
    max_flight_reference_deg: float = 35.0
    min_flight_time: float = 0.06  # seconds
    min_flight_rise: float = 0.025  # meters above takeoff COM height
    min_progress: float = 0.01  # meters per hop, for stall detection
    max_stalled_hops: int = 2
    replan_each_hop: bool = True
    verbose: bool = True


def wrap_pitch(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def validate_touchdown(ff, ss, p, options: PlanningOptions):
    """Check attitude and reject degenerate ballistic arcs at foot contact."""
    state = ff['state']
    pitch = wrap_pitch(state[4])
    rate = float(state[6])
    duration = float(ff['end'] - ss['end'])
    rise = float(max(0.0, np.max(ff['y'][:, 1]) - ss['state'][2] * np.cos(ss['state'][0])))
    errors = []
    if duration < options.min_flight_time:
        errors.append('degenerate_flight_time')
    if rise < options.min_flight_rise:
        errors.append('degenerate_flight_rise')
    if abs(pitch) > np.deg2rad(options.max_touchdown_angle_deg):
        errors.append('unsafe_touchdown_pitch')
    if abs(rate) > options.max_touchdown_pitch_rate:
        errors.append('unsafe_touchdown_pitch_rate')
    return dict(safe=not errors, errors=errors,
                touchdown_pitch_rad=pitch, touchdown_pitch_rate=rate,
                flight_duration=duration, flight_rise=rise)


def _one_hop(y0, foot_x, alpha_ref, p, c, start_time=0., accurate=False,
             flight_ref=None, options=None):
    """Predict stance -> flight -> touchdown with landing safety validation.

    Keeps the original positional arguments unchanged; new parameters are
    keyword arguments. Returns unsafe touchdown details without applying the
    impact reset to an invalid landing.
    """
    if options is None:
        options = PlanningOptions()
    if flight_ref is None:
        flight_ref = c.flight_target
    ss = simulate_stance(y0, foot_x, float(alpha_ref), p, c, start_time, accurate)
    if ss['event'] != 'takeoff':
        return dict(ok=False, event=ss['event'], stance=ss, flight=None)
    ff = simulate_flight(to_flight(ss['state'], foot_x, p), p, c,
                         ss['end'], accurate, flight_target=float(flight_ref))
    if ff['event'] != 'touchdown':
        return dict(ok=False, event=ff['event'], stance=ss, flight=ff)
    check = validate_touchdown(ff, ss, p, options)
    x_foot, _ = foot_position_flight(ff['state'], p)
    if not check['safe']:
        return dict(ok=False, event=check['errors'][0], stance=ss, flight=ff,
                    foot_x=float(x_foot), landing=check)
    foot_next, y_next, loss = impact_reset(ff['state'], p)
    jx, jz, shortfall = impact_impulse_diagnostics(ff['state'], y_next, foot_next, p)
    return dict(ok=True, event='touchdown', stance=ss, flight=ff,
                foot_x=float(foot_next), state=y_next, landing=check,
                impact_loss=float(loss),
                impact_friction_shortfall=float(shortfall),
                impact_jx=float(jx), impact_jz=float(jz))


def _split_references(refs, flight_refs=None):
    """Accept old stance-only rollout API or paired (stance,flight) arrays."""
    st = np.asarray(refs, dtype=float)
    if st.ndim == 2:
        if st.shape[1] != 2:
            raise ValueError('Paired references must be N x 2')
        return st[:, 0].copy(), st[:, 1].copy()
    st = st.reshape(-1)
    fl = np.zeros(len(st)) if flight_refs is None else np.asarray(flight_refs, dtype=float).reshape(-1)
    if len(st) != len(fl):
        raise ValueError('Must have one flight reference per stance reference')
    return st, fl


def rollout(refs, y0, foot_x, target, p, c, full=False, accurate=False,
            flight_refs=None, options=None):
    """Simulate a multi-hop candidate, rejecting unsafe landings and stalls."""
    if options is None:
        options = PlanningOptions()
    st_refs, fl_refs = _split_references(refs, flight_refs)
    foot_x = float(foot_x)
    y = np.asarray(y0, dtype=float).copy()
    time = 0.0
    outputs = []
    crossings = 0
    stalled = 0
    for i, (sref, fref) in enumerate(zip(st_refs, fl_refs)):
        one = _one_hop(y, foot_x, sref, p, c, time, accurate,
                       flight_ref=fref, options=options)
        if not one['ok']:
            return dict(ok=False, completed=i, foot_x=foot_x, y=y,
                        time=time, event=one['event'], steps=outputs,
                        crossings=crossings, failure=one)
        time = one['flight']['end']
        foot_prev = foot_x
        foot_x, y = one['foot_x'], one['state']
        if (foot_prev - target) * (foot_x - target) < 0:
            crossings += 1
        stalled = stalled + 1 if abs(foot_x - foot_prev) < options.min_progress else 0
        record = dict(hop=i + 1, alpha_ref=float(sref),
                      flight_ref=float(fref), foot_start=foot_prev,
                      foot_end=foot_x, touchdown_time=float(time),
                      touchdown_pitch_rad=one['landing']['touchdown_pitch_rad'],
                      touchdown_pitch_rate=one['landing']['touchdown_pitch_rate'],
                      flight_duration=one['landing']['flight_duration'],
                      flight_rise=one['landing']['flight_rise'],
                      impact_loss=one['impact_loss'],
                      impact_friction_shortfall=one['impact_friction_shortfall'],
                      impact_jx=one['impact_jx'], impact_jz=one['impact_jz'],
                      takeoff_time=one['stance']['end'],
                      min_stance_normal=one['stance']['normal_min'],
                      max_friction_margin=one['stance']['max_slip'],
                      spring_min_length=one['stance']['minimum_length'])
        if full:
            record['stance'] = one['stance']
            record['flight'] = one['flight']
        outputs.append(record)
        if stalled >= options.max_stalled_hops and abs(foot_x-target) > options.arrival_tolerance:
            return dict(ok=False, completed=i+1, foot_x=foot_x, y=y,
                        time=time, event='stalled_no_progress', steps=outputs,
                        crossings=crossings)
    return dict(ok=True, completed=len(outputs), foot_x=foot_x, y=y,
                time=time, event='touchdown', steps=outputs,
                crossings=crossings)


def _sequence_cost(result, stance_refs, flight_refs, target, options):
    dx = float(result['foot_x'] - target)
    angle_cost = options.landing_angle_cost * sum(
        s['touchdown_pitch_rad']**2 for s in result['steps'])
    rate_cost = options.landing_rate_cost * sum(
        s['touchdown_pitch_rate']**2 for s in result['steps'])
    if not result['ok']:
        # Soft distance-to-feasible shaping while keeping invalid candidates
        # strictly ineligible for selection, even when their foot x matches.
        fail = result.get('failure', {})
        landing = fail.get('landing', {})
        pitch_excess = max(0., abs(landing.get('touchdown_pitch_rad', 0.))
                           - np.deg2rad(options.max_touchdown_angle_deg))
        rate_excess = max(0., abs(landing.get('touchdown_pitch_rate', 0.))
                          - options.max_touchdown_pitch_rate)
        time_excess = max(0., options.min_flight_time
                          - landing.get('flight_duration', options.min_flight_time))
        rise_excess = max(0., options.min_flight_rise
                          - landing.get('flight_rise', options.min_flight_rise))
        estimate_x = fail.get('foot_x', result['foot_x'])
        return float(1e5 + 100*(estimate_x-target)**2
                     + 2e3*(pitch_excess**2 + rate_excess**2)
                     + 2e4*(time_excess**2 + rise_excess**2)
                     + 1e3*(len(stance_refs)-result['completed']))
    return float(100*dx**2 + options.hop_cost*len(stance_refs)
                 + options.tilt_cost*(np.sum(np.asarray(stance_refs)**2)
                                      + np.sum(np.asarray(flight_refs)**2))
                 + angle_cost + rate_cost
                 + options.friction_cost * sum(
                     s['impact_friction_shortfall']**2 for s in result['steps'])
                 + 2.0*sum(abs(s['foot_end']-s['foot_start']) < options.min_progress
                           for s in result['steps']))


def optimize_horizons(y, foot_x, target, p: Robot, c: Control,
                      options: PlanningOptions):
    """Independently optimize N-hop stance+flight references for N=1..H."""
    choices = []
    max_stance_ref = float(c.max_tilt_reference)
    max_flight_ref = float(np.deg2rad(options.max_flight_reference_deg))
    direction = 1. if target >= foot_x else -1.
    for n in range(1, options.lookahead + 1):
        # Interleaved references inside seeds are later stored as [stances, flights].
        seed_pairs = [
            (np.zeros(n), np.zeros(n)),
            (np.full(n, direction * max_stance_ref * .55), np.zeros(n)),
            (np.linspace(direction * max_stance_ref * .75,
                         -direction * max_stance_ref * .50, n), np.zeros(n)),
            (np.linspace(-direction * max_stance_ref * .3,
                          direction * max_stance_ref * .65, n), np.zeros(n)),
            (np.full(n, direction * max_stance_ref * .55),
             np.full(n, direction * max_flight_ref * .25)),
            (np.full(n, direction * max_stance_ref * .55),
             np.full(n, -direction * max_flight_ref * .25)),
        ]
        seeds = [np.r_[st, fl] for st, fl in seed_pairs]
        bounds = [(-max_stance_ref, max_stance_ref)]*n + [(-max_flight_ref, max_flight_ref)]*n
        cache = {}
        best_safe = [np.inf, None, None]
        best_any = [np.inf, None, None]

        def objective(xx):
            xx = np.asarray(xx, dtype=float)
            xx = np.array([np.clip(v, *bounds[j]) for j,v in enumerate(xx)])
            key = tuple(np.round(xx, 7))
            if key not in cache:
                rr = rollout(xx[:n], y, foot_x, target, p, c,
                             flight_refs=xx[n:], options=options)
                cache[key] = (_sequence_cost(rr, xx[:n], xx[n:], target, options), rr)
            cost, rr = cache[key]
            if cost < best_any[0]:
                best_any[:] = [float(cost), xx.copy(), rr]
            if rr['ok'] and cost < best_safe[0]:
                best_safe[:] = [float(cost), xx.copy(), rr]
            return float(cost)

        for seed in seeds:
            objective(seed)
        seed = min(seeds, key=objective)
        minimize(objective, seed, method='Powell', bounds=bounds,
                 options={'maxfev':options.maxfev_per_horizon,
                          'maxiter':max(4, 3*n+2),
                          'xtol':0.025, 'ftol':0.01, 'disp':False})
        _, params, rr = best_safe if best_safe[1] is not None else best_any
        score = best_safe[0] if best_safe[1] is not None else best_any[0]
        choices.append(dict(n=n, score=float(score),
                            refs=params[:n].copy(), flight_refs=params[n:].copy(),
                            rollout=rr, ok=bool(rr['ok']), evaluations=len(cache)))
    valid = [v for v in choices if v['ok']]
    selected = min(valid, key=lambda item: item['score']) if valid else None
    return selected, choices


def run_optimized_hops(p: Robot, c: Control, target, compression,
                       options: PlanningOptions):
    y = initial_state(p, compression)
    foot_x = 0.
    t = 0.
    hops, segments, landings, all_plans = [], [], [], []
    crossings = 0
    previous_error = foot_x - target
    status = 'safety hop limit reached'
    active_plan = []
    stalled_hops = 0
    for i in range(options.safety_max_hops):
        if not active_plan or options.replan_each_hop:
            chosen, choices = optimize_horizons(y, foot_x, target, p, c, options)
            all_plans.append(dict(
                at_hop=i+1, from_foot=foot_x,
                selected_horizon=None if chosen is None else chosen['n'],
                candidates=[dict(horizon=h['n'], score=h['score'],
                                 refs=h['refs'].tolist(),
                                 flight_refs=h['flight_refs'].tolist(),
                                 predicted_terminal_x=h['rollout']['foot_x'],
                                 feasible=h['ok'], evaluations=h['evaluations'],
                                 failure_reason=None if h['ok'] else h['rollout']['event'])
                            for h in choices]))
            if chosen is None:
                status = f'no safe feasible candidate trajectory from hop {i+1}'
                break
            active_plan = list(zip(chosen['refs'].tolist(), chosen['flight_refs'].tolist()))
            if options.verbose:
                opts = ', '.join(f"{j['n']}hop:{j['score']:.2f}" if j['ok']
                                 else f"{j['n']}hop:invalid({j['rollout']['event']})"
                                 for j in choices)
                print(f'Plan at hop {i+1}: choose {chosen["n"]} hops; ' + opts)
        stance_ref, flight_ref = active_plan.pop(0)
        step = _one_hop(y, foot_x, stance_ref, p, c, t, accurate=True,
                        flight_ref=flight_ref, options=options)
        if not step['ok']:
            # Preserve attempted motion for crash diagnosis; do NOT apply a
            # fictitious landing reset or label it a successful arrival.
            if step.get('stance') is not None:
                segments.append(step['stance'])
            if step.get('flight') is not None:
                segments.append(step['flight'])
            status = f'unsafe/invalid hop {i+1}: {step["event"]}'
            break
        ss, ff = step['stance'], step['flight']
        segments.extend([ss, ff])
        foot_old, foot_x, y, t = foot_x, step['foot_x'], step['state'], ff['end']
        error = foot_x - target
        if previous_error*error < 0:
            crossings += 1
        previous_error = error
        progress = abs(foot_x - foot_old)
        stalled_hops = stalled_hops + 1 if progress < options.min_progress else 0
        check = step['landing']
        info = dict(hop=i+1, foot_start=foot_old, foot_end=foot_x,
                    alpha_reference=float(stance_ref),
                    flight_reference=float(flight_ref),
                    touchdown_pitch_rad=check['touchdown_pitch_rad'],
                    touchdown_pitch_deg=float(np.degrees(check['touchdown_pitch_rad'])),
                    touchdown_pitch_rate=check['touchdown_pitch_rate'],
                    flight_duration=check['flight_duration'],
                    flight_rise=check['flight_rise'],
                    touchdown_safe=True, touchdown_time=t,
                    takeoff_time=ss['end'], impact_loss=step['impact_loss'],
                    impact_friction_shortfall=step['impact_friction_shortfall'],
                    spring_min_length=ss['minimum_length'],
                    min_stance_normal=ss['normal_min'],
                    max_friction_margin=ss['max_slip'], goal_crossings=crossings)
        hops.append(info)
        landings.append((t, foot_x))
        if options.verbose:
            print(f"Hop {i+1}: {foot_old:+.3f} → {foot_x:+.3f} m; "
                  f"stance={np.degrees(stance_ref):+.1f}°, "
                  f"flight={np.degrees(flight_ref):+.1f}°; "
                  f"TD pitch={info['touchdown_pitch_deg']:+.1f}°, "
                  f"TD rate={check['touchdown_pitch_rate']:+.2f} rad/s; "
                  f"crossings={crossings}; "
                  f"impact slip deficit={step['impact_friction_shortfall']:.2f} Ns")
        if abs(error) <= options.arrival_tolerance:
            status = 'SAFE touchdown inside goal tolerance (residual energy allowed)'
            break
        if crossings >= options.max_crossings:
            status = f'stopped after {crossings} goal crossings (safe landings; residual energy allowed)'
            break
        if stalled_hops >= options.max_stalled_hops:
            status = f'stalled: {stalled_hops} consecutive near-zero-progress hops'
            break

    relative_speeds, motor_powers = [], []
    for seg in segments:
        yy, uu = seg['y'], seg['u']
        relative = yy[:, 4]-yy[:, 3] if seg['phase']=='stance' else yy[:, 7]-yy[:, 6]
        relative_speeds.extend(np.abs(relative).tolist())
        motor_powers.extend(np.abs(uu*relative).tolist())
    return dict(segments=segments, landings=landings, hops=hops,
                final_foot_x=float(foot_x), target=float(target),
                status=status, last_stance_y=y, total_time=float(t),
                goal_crossings=crossings, plans=all_plans,
                peak_relative_wheel_speed=max(relative_speeds, default=0.),
                peak_motor_mechanical_power=max(motor_powers, default=0.),
                settings=options)
