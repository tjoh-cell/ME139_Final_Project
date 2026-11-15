"""Multi-hop event-driven shooting optimization for the 2-D passive hopper.

Optimizes an *entire sequence* of one stance attitude reference per hop,
solving separate small bounded optimization problems for N=1..lookahead.
The hybrid stance/flight/impact model is the original hybrid_hopper.py.

NOT a full CasADi / IPOPT time-discretized torque MPC; the optimized control
variables are per-hop stance reference angles and the torques follow bounded
PD feedback. Hybrid events are nonsmooth, so derivative-free Powell is used.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from hybrid_hopper import (
    Robot, Control, initial_state, simulate_stance, simulate_flight,
    impact_reset, impact_impulse_diagnostics, to_flight,
    foot_position_flight, com_from_stance,
)

@dataclass(frozen=True)
class PlanningOptions:
    lookahead: int = 3
    safety_max_hops: int = 10
    max_crossings: int = 2
    arrival_tolerance: float = .05
    maxfev_per_horizon: int = 70
    hop_cost: float = .015
    tilt_cost: float = .005
    friction_cost: float = .0003  # soft: impact is not contact-feasible yet
    replan_each_hop: bool = True
    verbose: bool = True


def _one_hop(y0, foot_x, alpha_ref, p, c, start_time=0., accurate=False):
    """Return predicted touchdown (after reset) or a descriptive failure."""
    ss=simulate_stance(y0,foot_x,float(alpha_ref),p,c,start_time,accurate)
    if ss['event']!='takeoff':
        return dict(ok=False,event=ss['event'],stance=ss,flight=None)
    ff=simulate_flight(to_flight(ss['state'],foot_x,p),p,c,ss['end'],accurate)
    if ff['event']!='touchdown':
        return dict(ok=False,event=ff['event'],stance=ss,flight=ff)
    foot_next,y_next,loss=impact_reset(ff['state'],p)
    jx,jz,shortfall=impact_impulse_diagnostics(ff['state'],y_next,foot_next,p)
    return dict(ok=True,event='touchdown',stance=ss,flight=ff,
                foot_x=float(foot_next),state=y_next,impact_loss=float(loss),
                impact_friction_shortfall=float(shortfall),
                impact_jx=float(jx),impact_jz=float(jz))


def rollout(refs,y0,foot_x,target,p,c,full=False,accurate=False):
    """Propagate candidate hop sequence, including every impact reset.

    full=False excludes trajectory arrays from returned data but simulation
    functions still compute samples for contact/actuator diagnostics.
    """
    foot_x=float(foot_x); y=np.asarray(y0,dtype=float).copy()
    time=0.; outputs=[]; reached=0
    for i,ref in enumerate(refs):
        one=_one_hop(y,foot_x,ref,p,c,time,accurate)
        if not one['ok']:
            return dict(ok=False,completed=i,foot_x=foot_x,y=y,
                        time=time,event=one['event'],steps=outputs)
        time=one['flight']['end']
        foot_prev=foot_x;foot_x=one['foot_x'];y=one['state']
        if (foot_prev-target)*(foot_x-target)<0: reached+=1
        record=dict(hop=i+1,alpha_ref=float(ref),foot_start=foot_prev,
                    foot_end=foot_x,touchdown_time=float(time),
                    impact_loss=one['impact_loss'],
                    impact_friction_shortfall=one['impact_friction_shortfall'],
                    impact_jx=one['impact_jx'],impact_jz=one['impact_jz'],
                    takeoff_time=one['stance']['end'],
                    min_stance_normal=one['stance']['normal_min'],
                    max_friction_margin=one['stance']['max_slip'],
                    spring_min_length=one['stance']['minimum_length'])
        if full:
            record['stance']=one['stance'];record['flight']=one['flight']
        outputs.append(record)
    return dict(ok=True,completed=len(outputs),foot_x=foot_x,y=y,
                time=time,event='touchdown',steps=outputs,crossings=reached)


def _sequence_cost(result,refs,target,options):
    if not result['ok']:
        # Strongly discourage plans which stall or violate stance feasibility.
        # Successful shorter horizons are optimized separately.
        return 1e5+1e3*(len(refs)-result['completed'])+(
            100*(result['foot_x']-target)**2)
    dx=result['foot_x']-target
    return float(100*dx*dx+options.hop_cost*len(refs)
                 +options.tilt_cost*np.sum(np.asarray(refs)**2)
                 +options.friction_cost*sum(s['impact_friction_shortfall']**2
                                           for s in result['steps']))


def optimize_horizons(y,foot_x,target,p:Robot,c:Control,
                      options:PlanningOptions):
    """Solve independent shooting problems with 1,2,...lookahead hops.

    The selected plan can change sign after an overshoot. Cost uses the end
    of each candidate sequence, rather than requiring every hop to make
    forward progress. Returns a cross-horizon comparison including failures.
    """
    choices=[]
    max_ref=float(c.max_tilt_reference)
    # Positive alpha => +x tilt. Negative => -x tilt.
    direction=1. if target>=foot_x else -1.
    for n in range(1,options.lookahead+1):
        # Diverse feasible seeds; strongly prioritize sign-reversal trajectories.
        seeds=[np.zeros(n),
               np.full(n,direction*max_ref*0.55),
               np.linspace(direction*max_ref*0.7,-direction*max_ref*0.6,n),
               np.linspace(-direction*max_ref*0.35,direction*max_ref*0.65,n)]
        seeds=[np.asarray(s,dtype=float).reshape(n) for s in seeds]
        cache={}
        best=[np.inf,None,None]
        def objective(xx):
            xx=np.clip(np.asarray(xx,dtype=float),-max_ref,max_ref)
            key=tuple(np.round(xx,7))
            if key not in cache:
                rr=rollout(xx,y,foot_x,target,p,c)
                cache[key]=(_sequence_cost(rr,xx,target,options),rr)
            cost,rr=cache[key]
            if cost<best[0]:best[:]=[float(cost),xx.copy(),rr]
            return float(cost)
        for s in seeds:
            objective(s)
        # The optimization is genuinely joint: all N reference angles are
        # simultaneously adjustable, not independently selected greedily.
        seed=min(seeds,key=objective)
        minimize(objective,seed,method='Powell',
                 bounds=[(-max_ref,max_ref)]*n,
                 options={'maxfev':options.maxfev_per_horizon,
                          'maxiter':max(4,2*n+2),
                          'xtol':0.02,'ftol':0.015,'disp':False})
        score,refs,rr=best
        choices.append(dict(n=n,score=float(score),refs=refs.copy(),
                            rollout=rr,ok=bool(rr['ok']),
                            evaluations=len(cache)))
    ok=[item for item in choices if item['ok']]
    selected=min(ok,key=lambda item:item['score']) if ok else None
    return selected,choices


def run_optimized_hops(p:Robot,c:Control,target,compression,
                       options:PlanningOptions):
    y=initial_state(p,compression)
    foot_x=0.;t=0.
    hops=[];segments=[];landings=[];all_plans=[]
    crossings=0;previous_error=foot_x-target
    status='safety hop limit reached'
    active_plan=[]
    for i in range(options.safety_max_hops):
        if not active_plan or options.replan_each_hop:
            chosen,choices=optimize_horizons(y,foot_x,target,p,c,options)
            all_plans.append(dict(at_hop=i+1,from_foot=foot_x,
                                  selected_horizon=None if chosen is None else chosen['n'],
                                  candidates=[dict(horizon=h['n'],score=h['score'],
                                                   refs=h['refs'].tolist(),
                                                   predicted_terminal_x=h['rollout']['foot_x'],
                                                   feasible=h['ok'],evaluations=h['evaluations'])
                                              for h in choices]))
            if chosen is None:
                status=f'no feasible candidate trajectory from hop {i+1}'
                break
            active_plan=chosen['refs'].tolist()
            if options.verbose:
                opts=', '.join(f"{j['n']}hop:{j['score']:.2f}" if j['ok']
                               else f"{j['n']}hop:invalid" for j in choices)
                print(f'Plan at hop {i+1}: choose {chosen["n"]} hops; '+opts)
        ref=active_plan.pop(0)
        step=_one_hop(y,foot_x,ref,p,c,t,accurate=True)
        if not step['ok']:
            status=f"validation failed at hop {i+1}: {step['event']}"
            break
        ss=step['stance'];ff=step['flight']
        segments.extend([ss,ff])
        foot_old=foot_x;foot_x=step['foot_x'];y=step['state'];t=ff['end']
        error=foot_x-target
        if previous_error*error<0:crossings+=1
        previous_error=error
        info=dict(hop=i+1,foot_start=foot_old,foot_end=foot_x,
                  alpha_reference=ref,touchdown_time=t,
                  takeoff_time=ss['end'],impact_loss=step['impact_loss'],
                  impact_friction_shortfall=step['impact_friction_shortfall'],
                  spring_min_length=ss['minimum_length'],
                  min_stance_normal=ss['normal_min'],
                  max_friction_margin=ss['max_slip'],
                  goal_crossings=crossings)
        hops.append(info);landings.append((t,foot_x))
        if options.verbose:
            print(f"Hop {i+1}: {foot_old:+.3f} → {foot_x:+.3f} m; "
                  f"reference={np.degrees(ref):+.1f}°; "
                  f"crossings={crossings}; impact slip deficit={step['impact_friction_shortfall']:.2f} Ns")
        if abs(error)<=options.arrival_tolerance:
            status='landing inside goal tolerance (residual energy allowed)'
            break
        if crossings>=options.max_crossings:
            status=f'stopped after {crossings} goal crossings (residual energy allowed)'
            break
    relative_speeds=[]; motor_powers=[]
    for seg in segments:
        yy=seg['y']; uu=seg['u']
        relative=yy[:,4]-yy[:,3] if seg['phase']=='stance' else yy[:,7]-yy[:,6]
        relative_speeds.extend(np.abs(relative).tolist())
        motor_powers.extend(np.abs(uu*relative).tolist())
    return dict(segments=segments,landings=landings,hops=hops,
                final_foot_x=float(foot_x),target=float(target),
                status=status,last_stance_y=y,total_time=float(t),
                goal_crossings=crossings,plans=all_plans,
                peak_relative_wheel_speed=max(relative_speeds,default=0.),
                peak_motor_mechanical_power=max(motor_powers,default=0.),
                settings=options)
