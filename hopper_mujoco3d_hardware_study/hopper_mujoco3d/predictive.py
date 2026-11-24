"""Bounded sample-based *physical* MuJoCo hop planner.

This is intentionally not called an IPOPT MPC: decisions are angle references
and a flight-thrust scale. Each candidate is integrated in an isolated copy
of the *same* MuJoCo plant. A short additional hop checks viability; no
kinematic resets, ideal impulses, or extra actuation are introduced.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from control import clip, target_vx


@dataclass(frozen=True)
class HopCommand:
    stance_ref: float
    flight_ref: float
    thrust_scale: float = 1.0
    label: str = 'feedback'
    foot_placement_gain: float | None = None


def candidates(sim):
    """Bounded, deterministic candidate references; no desired stride."""
    pos = sim.d.qpos[:3]
    vx = sim.d.qvel[0]
    v_goal = target_vx(sim.mission.goal_x-pos[0], sim.g)
    nominal = clip(sim.g.pitch_gain_v*(v_goal-vx),
                   -sim.g.stance_pitch_limit, sim.g.stance_pitch_limit)
    bound=sim.g.stance_pitch_limit
    # Include neutral/backward recovery even when the goal is forward.
    starts = [nominal, 0.0,
              clip(nominal+math.radians(12),-bound,bound),
              clip(nominal-math.radians(12),-bound,bound),
              -math.copysign(math.radians(15), vx if abs(vx)>0.1 else nominal or 1)]
    # Include full ascent assistance, coasting and a braking posture even
    # when the caller uses a small search budget.
    alternate_placement = 0.44 if math.isclose(sim.g.foot_placement_gain, 0.55) else 0.55
    refs = [HopCommand(nominal, 0., 1., 'velocity_feedback'),
            HopCommand(nominal, 0., 1., 'velocity_feedback', foot_placement_gain=alternate_placement),
            HopCommand(nominal, 0., 0., 'coast'),
            HopCommand(0., 0., 0.65, 'neutral'),
            HopCommand(-nominal, 0., 1., 'brake')]
    for s in starts:
        for f,scale in ((0.0,0.0), (0.0,0.65),
                        (-math.radians(10),0.45), (math.radians(10),0.45)):
            cmd=HopCommand(float(s),float(f),float(scale),'rollout')
            if not any(abs(z.stance_ref-cmd.stance_ref)<1e-5 and
                       abs(z.flight_ref-cmd.flight_ref)<1e-5 and
                       abs(z.thrust_scale-cmd.thrust_scale)<1e-5 for z in refs):
                refs.append(cmd)
    return refs


def rollout_hop(sim, cmd, max_duration=3.5):
    """Return a branch after real contact, or failure reason on the branch."""
    branch=sim.fork_for_prediction(cmd)
    start_hops=branch.hops
    max_steps=math.ceil(max_duration/branch.p.dt)
    for _ in range(max_steps):
        if branch.status!='running' or branch.hops>start_hops:
            break
        branch.forward_step()
    return branch, branch.hops>start_hops and branch.status in ('running','goal_tolerance_reached')


def landing_score(start, branch):
    landing=branch.landings[-1]
    goal=start.mission.goal_x
    dx0=abs(goal-start.d.qpos[0])
    dx1=abs(goal-landing['foot_x'])
    pitch=abs(landing['pitch'])
    tilt=landing['offplane_tilt']
    wx=abs(landing['roll_rate'])
    wy=abs(landing['pitch_rate'])
    vx=landing['vx']
    # Keep the position penalty monotone through the braking region.
    # An approaching high speed adds a viability penalty, not a hard velocity
    # limit; backward hops and stored kinetic energy remain permitted.
    speed_excess=max(0.,abs(vx)-max(1.,math.sqrt(2*start.g.braking_accel*max(dx1,0.01))))
    return (dx1 +
            2.5*pitch**2 + 4.0*tilt**2 +
            3.0*landing['foot_y']**2 + 0.6*landing['vy']**2 +
            0.6*wy**2+0.8*wx**2+0.4*speed_excess**2+
            0.025*(branch.memory.impulse_used-start.memory.impulse_used) +
            (3.0 if dx1 > dx0+0.1 else 0.0))


def choose_hop(sim, budget=12, viable_next=True, next_budget=3):
    """Pick a reference that completes a physically simulated landing.

    Only the shortlist gets a second-hop check. Infeasible next-hop search is
    a bounded search outcome. It is
    reported in the candidate diagnostics.
    """
    trials = []
    failures = {}
    evaluated = 0
    for cmd in candidates(sim)[:max(1, int(budget))]:
        branch, safe = rollout_hop(sim, cmd)
        evaluated += 1
        if branch.status == 'goal_tolerance_reached':
            return cmd, dict(status='goal', evaluated=evaluated, safe=len(trials)+1,
                             cost=0., failures=failures)
        if safe:
            trials.append((landing_score(sim, branch), cmd, branch))
        else:
            reason = 'rollout_time_limit' if branch.status == 'running' else branch.status
            failures[reason] = failures.get(reason, 0) + 1

    diagnostic = dict(evaluated=evaluated, safe=len(trials), failures=failures)
    if not trials:
        return None, dict(diagnostic, status='no_safe_candidate_in_bounded_search', cost=None)
    trials.sort(key=lambda entry: entry[0])
    score, cmd, _ = trials[0]
    if viable_next:
        for candidate_score, candidate, branch in trials[:3]:
            for second in candidates(branch)[:max(1, int(next_budget))]:
                _, good = rollout_hop(branch, second)
                if good:
                    return candidate, dict(diagnostic, status='next_hop_found', cost=candidate_score)
        return cmd, dict(diagnostic, status='next_hop_not_found', cost=score)
    return cmd, dict(diagnostic, status='safe_touchdown', cost=score)
