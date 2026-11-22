"""Multi-hop trajectory search with active fallback and full-hop recovery checks.

1..lookahead horizon problems are solved separately with bounded Powell.
The optimizer chooses a stance AND flight body-angle reference per hop.
Intermediate hops are simulated with physical contact; the terminal hop is
evaluated with full next-hop trial rollouts for shortlisted candidates.

Importantly: this is a local trajectory search, NOT a globally optimal MPC.
No physical energy is added by the planner. When it cannot find a full hop,
an independent active emergency attitude policy is searched and executed.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import minimize
from physics import Robot,Controller,SimulationConfig,initial_state,foot_kinematics
from simulator import GoalOptions,one_hop,rollout,recovery_probe,full_hop_probe
from dataclasses import dataclass, replace

@dataclass(frozen=True)
class PlannerOptions:
    lookahead: int=3
    maxfev: int=30
    landing_pitch_weight: float=6.0
    landing_rate_weight: float=0.5
    postimpact_rate_weight: float=5.0
    recovery_rate_weight: float=1.0
    slip_weight: float=2.0
    terminal_speed_weight: float=18.0
    moving_away_weight: float=8.0
    recoverable_deceleration: float=2.0 # heuristic m/s^2 for *planning*, not constraint
    near_goal_speed: float=0.8
    recovery_probe_time: float=0.10 # legacy: used for optional immediate-collision diagnostic
    continuation_candidates: int=1 # expensive full-hop checks per horizon
    continuation_probe_count: int=3
    require_full_next_hop: bool=True
    intermediate_speed_weight: float=0.45
    emergency_maxfev: int=10
    emergency_reference_levels: int=3
    require_terminal_recovery: bool=True
    max_planned_postimpact_rate: float=3.5 # rad/s; adjustable conservative planning bound
    passive_fallback: bool=True # legacy flag: now enables ACTIVE emergency fallback
    max_passive_fallbacks: int=2 # legacy limit: number of ACTIVE fallback attempts
    energy_weight: float=0.0
    hop_cost: float=0.08
    reference_cost: float=0.03
    invalid_cost: float=20000.0
    position_weight: float=100.0
    verbose: bool=True


def arrival_speed_cost(x,vx,target,options):
    """Penalize excessive impact momentum as goal approaches.

    The sqrt threshold is a stopping-distance *heuristic*; only actual rollout
    contacts and dynamics decide whether a candidate is physically feasible.
    """
    error=float(x-target)
    allowance=math.sqrt(options.near_goal_speed**2 +
                        2*options.recoverable_deceleration*abs(error))
    excess=max(0.0,abs(vx)-allowance)
    away=max(0.0,math.copysign(1.0,error)*vx) if abs(error)>1e-6 else 0.0
    return options.terminal_speed_weight*excess**2 + options.moving_away_weight*away**2


def trajectory_cost(out,refs,goal:GoalOptions,options:PlannerOptions):
    x_end=float(out['foot_x'])
    if not out['ok'] and out['steps']:
        x_end=float(out['steps'][-1].get('foot_x',x_end))
    err=x_end-goal.target_x
    cost=options.position_weight*err**2+options.hop_cost*len(refs)
    for i,hop in enumerate(out['steps']):
        cost+=options.landing_pitch_weight*float(hop['touchdown_pitch'])**2
        cost+=options.landing_rate_weight*float(hop['touchdown_rate'])**2
        cost+=options.postimpact_rate_weight*float(hop.get('postimpact_rate',0.))**2
        cost+=options.slip_weight*float(hop['slip_duration'])**2
        # Penalize dangerous momentum at EVERY landing, not only the last.
        # Position-only planning rewarded long hops that stranded the robot.
        wx=options.intermediate_speed_weight if i<len(out['steps'])-1 else 1.0
        cost+=wx*arrival_speed_cost(hop.get('foot_x',x_end),hop.get('postimpact_vx',0.),
                                    goal.target_x,options)
        # Detect high post-impact angular speed even if pre-impact rate was tiny.
        overspin=max(0.,abs(hop.get('postimpact_rate',0.))-
                     options.max_planned_postimpact_rate)
        cost+=1000*overspin**2
    cost+=options.reference_cost*float(np.sum(np.asarray(refs)**2))
    probe=out.get('terminal_recovery')
    if probe and probe['viable'] and probe.get('best') is not None:
        good=probe['best']
        if 'score' in good:
            cost+=0.10*good['score']
        else:  # backwards-compatible short-probe diagnostic
            cost+=0.25*options.postimpact_rate_weight*good.get('ending_pitch_rate',0.)**2
    if not out['ok']:
        cost+=options.invalid_cost+1500*max(0,len(refs)-len(out['steps']))
    return float(cost)


def _full_reference_candidates(state,p,c,count):
    direction=1. if state[5]>=0 else -1.
    refs=[(0.,0.),
          (-direction*.30*c.max_stance_ref,0.),
          (-direction*.70*c.max_stance_ref,-direction*.30*c.max_flight_ref),
          (direction*.30*c.max_stance_ref,0.),
          (0.,-direction*.5*c.max_flight_ref)]
    return refs[:max(1,count)]


def _rollout_with_terminal_recovery(state,refs,p,c,cfg,goal,options):
    """Expensive acceptance check, only called on shortlisted candidates.

    A full next-hop witness is required unless the terminal landing already
    meets the arrival tolerance. This does not certify global viability.
    """
    result=rollout(state,refs,p,c,cfg,goal)
    if result['ok'] and any(abs(h.get('postimpact_rate',0.))>
                            options.max_planned_postimpact_rate for h in result['steps']):
        result['ok']=False
        result['event']='excess_postimpact_rotation'
    if not result['ok'] or not result['steps']:
        return result
    # The stated objective allows residual energy at destination; avoid
    # requiring an extra hop after a successful, safely oriented touchdown.
    if abs(result['foot_x']-goal.target_x)<=goal.tolerance:
        result['terminal_recovery']=dict(viable=True,best=None,
                                         arrival_exempt=True,attempts=[])
        return result
    if options.require_full_next_hop:
        probe=full_hop_probe(result['final_state'],p,c,cfg,
            references=_full_reference_candidates(result['final_state'],p,c,
                                                     options.continuation_probe_count),
            max_postimpact_rate=options.max_planned_postimpact_rate)
        result['terminal_recovery']=probe
        if not probe['viable'] and options.require_terminal_recovery:
            result['ok']=False
            result['event']='no_safe_complete_next_hop_found'
    elif options.recovery_probe_time>0:
        probe=recovery_probe(result['final_state'],p,c,cfg,
                             options.recovery_probe_time)
        result['terminal_recovery']=probe
        if not probe['viable'] and options.require_terminal_recovery:
            result['ok']=False
            result['event']='postimpact_body_collision_risk'
    return result


def _emergency_score(h,goal):
    """Safety is primary. Reaching the goal is irrelevant if body collides."""
    pitch=abs(h['touchdown_pitch']);rate=abs(h['touchdown_rate'])
    post=abs(h['postimpact_rate']);v=abs(h['postimpact_vx'])
    safe=h['ok'] and abs(h['postimpact_rate'])<3.5
    if safe:
        """return float(80*pitch*pitch+25*rate*rate+16*post*post+1.5*v*v +
                     2*(h['foot_x']-goal.target_x)**2+20*h['slip_duration'])"""
        return float(
        120 * pitch**2
        + 40 * rate**2
        + 30 * post**2
        + 5 * v**2
        + 20 * h['slip_duration']
        + 0.02 * (h['foot_x'] - goal.target_x)**2
    )
    if h['event']=='body_collision':
        # More time before collision is less bad, but never beats safe touchdown.
        return float(30000-120*min(h['total_duration'],4.) + 100*pitch)
    if h['event']=='no_takeoff':
        return float(8000+100*abs(h['postimpact_rate']))
    if h['event']=='touchdown':
        return float(12000+50*pitch+30*rate+30*post)
    return float(18000+100*abs(h['postimpact_rate']))


def optimize_emergency(state,goal,p,c,cfg,options):
    """Independent bounded two-variable search for safest attainable next hop.

    Real motor gains remain ACTIVE in both stance and flight. Search over
    emergency stance/flight references; do not turn the motor off when the
    distance objective becomes infeasible. This is a local finite search.
    """
    levels=max(2,int(options.emergency_reference_levels))
    sr_grid=np.linspace(-c.max_stance_ref,c.max_stance_ref,levels)
    fr_grid=np.linspace(-c.max_flight_ref,c.max_flight_ref,levels)
    seeds=[(0.,0.)]+[(float(sr),float(fr)) for sr in sr_grid for fr in fr_grid]
    cache={}; best=[float('inf'),None,None]
    def objective(z):
        sr=float(np.clip(z[0],-c.max_stance_ref,c.max_stance_ref))
        fr=float(np.clip(z[1],-c.max_flight_ref,c.max_flight_ref))
        key=(round(sr,5),round(fr,5))
        if key not in cache:
            h=one_hop(state,sr,fr,p,c,cfg,record=False)
            cache[key]=(_emergency_score(h,goal),h)
        score,h=cache[key]
        if score<best[0]:best[:]=[score,(sr,fr),h]
        return score
    for seed in seeds: objective(seed)
    if options.emergency_maxfev>len(cache):
        minimize(objective,best[1],method='Powell',
                 bounds=[(-c.max_stance_ref,c.max_stance_ref),
                         (-c.max_flight_ref,c.max_flight_ref)],
                 options={'maxfev':options.emergency_maxfev,
                          'maxiter':8,'xtol':.05,'ftol':.04})
    _,ref,h=best
    return dict(ref=ref,score=best[0],predicted_event=h['event'],
                predicted_safe=bool(h['ok']),evaluations=len(cache),
                predicted_pitch=float(h['touchdown_pitch']),
                predicted_postimpact_rate=float(h['postimpact_rate']))


def optimize_horizons(state,goal:GoalOptions,p:Robot,c:Controller,
                      cfg:SimulationConfig,options:PlannerOptions):
    if options.lookahead<1:raise ValueError('lookahead must be positive')
    direction=1. if goal.target_x>=foot_kinematics(state[:5])[0][0] else -1.
    choices=[];viable_plans=[]
    for n in range(1,options.lookahead+1):
        maxs=c.max_stance_ref;maxf=c.max_flight_ref
        bounds=[(-maxs,maxs)]*n+[(-maxf,maxf)]*n
        seeds=[np.r_[np.zeros(n),np.zeros(n)],
               np.r_[np.full(n,direction*.7*maxs),np.zeros(n)],
               np.r_[np.full(n,direction*.35*maxs),np.zeros(n)],
               np.r_[np.linspace(direction*.8*maxs,-direction*.2*maxs,n),np.zeros(n)],
               np.r_[np.full(n,-direction*.4*maxs),np.zeros(n)],
               np.r_[np.full(n,direction*.75*maxs),np.full(n,-direction*.2*maxf)],
               np.r_[np.full(n,direction*.75*maxs),np.full(n,direction*.2*maxf)],
               np.r_[np.full(n,-direction*.7*maxs),np.full(n,direction*.3*maxf)],
               np.r_[np.full(n,direction*.15*maxs),np.full(n,direction*.6*maxf)]]
        cache={}
        def objective(v):
            v=np.asarray(v,float)
            v=np.array([np.clip(v[j],lo,hi) for j,(lo,hi) in enumerate(bounds)])
            key=tuple(np.round(v,5))
            if key not in cache:
                refs=list(zip(v[:n],v[n:]))
                # Cheap screening: no additional full-hop probe per Powell call.
                out=rollout(state,refs,p,c,cfg,goal)
                if out['ok'] and any(abs(h.get('postimpact_rate',0.))>
                                      options.max_planned_postimpact_rate for h in out['steps']):
                    out['ok']=False;out['event']='excess_postimpact_rotation'
                score=trajectory_cost(out,refs,goal,options)
                cache[key]=(score,v.copy(),out)
            return cache[key][0]
        for seed in seeds:objective(seed)
        best_seed=min(seeds,key=objective)
        if options.maxfev>len(seeds):
            minimize(objective,best_seed,method='Powell',bounds=bounds,
                     options={'maxfev':options.maxfev,'maxiter':max(4,n*3),
                              'xtol':.04,'ftol':.03,'disp':False})

        ranked=sorted(cache.values(),key=lambda entry:entry[0])
        admissible=[r for r in ranked if r[2]['ok']]
        checked=[]
        for _,params,_ in admissible[:max(1,options.continuation_candidates)]:
            refs=list(zip(params[:n],params[n:]))
            verified=_rollout_with_terminal_recovery(state,refs,p,c,cfg,goal,options)
            verified_score=trajectory_cost(verified,refs,goal,options)
            checked.append((verified_score,params,verified))
            if verified['ok']:
                viable_plans.append(dict(n=n,feasible=True,score=float(verified_score),
                    refs=[(float(a),float(b)) for a,b in refs],
                    terminal_x=float(verified['foot_x']),evaluations=len(cache),
                    failure=None,terminal_recovery_viable=True))
        if checked:
            best=min(checked,key=lambda item:item[0])
        else:
            best=ranked[0]
        score,params,rr=best
        choices.append(dict(n=n,feasible=bool(rr['ok']),score=float(score),
            refs=list(zip(params[:n].tolist(),params[n:].tolist())),
            terminal_x=float(rr['foot_x']),evaluations=len(cache),
            failure=None if rr['ok'] else rr['event'],
            terminal_recovery_viable=(rr.get('terminal_recovery') or {}).get('viable'),
            candidates_full_checked=len(checked)))
    selected=min(viable_plans,key=lambda a:a['score']) if viable_plans else None
    return selected,choices


def run_controller(target=3.,compression=.5,p:Robot|None=None,
                   c:Controller|None=None,cfg:SimulationConfig|None=None,
                   goal:GoalOptions|None=None,options:PlannerOptions|None=None):
    p=p or Robot();c=c or Controller()
    cfg=cfg or SimulationConfig(dt=.003)
    goal=goal or GoalOptions(target_x=target)
    options=options or PlannerOptions()
    y=initial_state(p,compression)
    time=0.;hops=[];history=[];motor_history=[];contact_history=[]
    plans=[];crossings=0;stalls=0;fallbacks=0;recovery_events=[]
    old_error=float(foot_kinematics(y[:5])[0][0]-goal.target_x)
    status='safety_hop_limit'
    for h in range(goal.safety_max_hops):
        # selected,choices=optimize_horizons(y,goal,p,c,cfg,options)
        
        # Stage 1: fast search
        selected, choices = optimize_horizons(
            y, goal, p, c, cfg, options
        )

        # Stage 2: expand search only if needed
        if selected is None:
            expanded = replace(
                options,
                maxfev=max(options.maxfev, 80),
                continuation_candidates=max(
                    options.continuation_candidates, 2
                ),
                continuation_probe_count=max(
                    options.continuation_probe_count, 5
                ),
            )

            if options.verbose:
                print("Fast search failed; expanding search...")

            selected, choices = optimize_horizons(
                y, goal, p, c, cfg, expanded
            )

        # Existing emergency fallback remains below this point.

        plans.append(dict(hop=h+1,chosen=None if selected is None else selected['n'],
                          candidates=choices))
        old_foot_x=float(foot_kinematics(y[:5])[0][0])
        fallback=False
        if selected is None:
            if not options.passive_fallback or fallbacks>=options.max_passive_fallbacks:
                status='no_feasible_plan_after_recovery_search';break
            # Run a separate safety-first search and execute it with the
            # REAL ACTIVE MOTOR, not a zero-gain controller.
            fallback=True;fallbacks+=1
            emergency=optimize_emergency(y,goal,p,c,cfg,options)
            sr,fr=emergency['ref']
            if options.verbose:
                fails=', '.join(f'{a["n"]}:{a["failure"]}' for a in choices)
                print(f'No goal-directed feasible plan at hop {h+1}; '
                      f'ACTIVE emergency search ({emergency["evaluations"]} trials), '
                      f'predicted={emergency["predicted_event"]}. '
                      f'Candidate failures: {fails}')
            used_controller=c
        else:
            sr,fr=selected['refs'][0]
            used_controller=c
        hop=one_hop(y,sr,fr,p,used_controller,cfg,record=True,start_time=time)
        history.extend(hop['trajectory'])
        motor_history.extend(hop['controls'])
        contact_history.extend(hop['contacts'])
        details={key:value for key,value in hop.items() if key not in ('state','trajectory','controls','contacts')}
        details.update(hop=h+1,from_x=old_foot_x,passive_fallback=False,
                       active_emergency=fallback,
                       chosen_horizon=0 if fallback else selected['n'])
        if fallback:
            details['emergency_plan']=emergency
        # In particular, preserve the final state and trace even on a crash.
        y=hop['state'];time=hop['time']
        if fallback:
            recovery_events.append(dict(hop=h+1,control_mode='active_emergency',
                                        chosen_refs=(float(sr),float(fr)),
                                        predicted_event=emergency['predicted_event'],
                                        predicted_safe=emergency['predicted_safe'],
                                        motor_work=hop['motor_mechanical_work'],
                                        event=hop['event'],
                                        errors=hop['errors'],duration=hop['total_duration'],
                                        start_x=old_foot_x,end_x=hop['foot_x']))
        if not hop['ok']:
            hops.append(details)
            status=('active_emergency_failed:' if fallback else 'unsafe_actual_hop:')+hop['errors'][0]
            break
        xnew=float(hop['foot_x']);err=xnew-goal.target_x
        if err*old_error<0:crossings+=1
        old_error=err
        stalls=stalls+1 if abs(xnew-old_foot_x)<goal.min_progress else 0
        details['goal_crossings']=crossings
        hops.append(details)
        if options.verbose:
            print(f'Hop {h+1}{" (ACTIVE emergency)" if fallback else ""}: '
                  f'{old_foot_x:+.3f} -> {xnew:+.3f} m; '
                  f'stance={math.degrees(sr):+.1f}°,flight={math.degrees(fr):+.1f}°; '
                  f'pre-impact rate={hop["touchdown_rate"]:+.2f}, '
                  f'post-impact rate={hop["postimpact_rate"]:+.2f} rad/s')
        if abs(err)<=goal.tolerance:
            status='goal_tolerance_reached';break
        if crossings>=goal.max_crossings:
            status='goal_crossing_limit';break
        if stalls>=goal.stall_hops:
            status='stalled_no_progress';break
    return dict(status=status,target=goal.target_x,final_x=float(foot_kinematics(y[:5])[0][0]),
                final_state=y,hop_info=hops,history=history,
                motor_history=motor_history,contact_history=contact_history,
                plan_history=plans,recovery_events=recovery_events,
                crossings=crossings,total_time=time,
                robot=p,controller=c,sim_config=cfg,goal_config=goal,
                planner_config=options)
