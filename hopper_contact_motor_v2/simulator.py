"""One or many hops using the same rigid-contact plant for prediction and execution."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from physics import (Robot,Controller,SimulationConfig,initial_state,foot_kinematics,
                     wrap_angle,step,mechanical_energy,body_corners)

@dataclass(frozen=True)
class GoalOptions:
    target_x: float = 3.0
    tolerance: float = 0.05
    max_crossings: int = 2
    safety_max_hops: int = 10
    min_progress: float = 0.015
    stall_hops: int = 2


def one_hop(y0, stance_ref, flight_ref, p:Robot,c:Controller,config:SimulationConfig,
            record=False, start_time=0.):
    """Simulate release -> takeoff -> ballistic motion -> foot contact.

    Body-corner contacts are treated as a collision and flagged, not silently
    accepted as valid landings. Only the FIRST foot touchdown terminates a hop;
    the spring compression / rebound is simulated in the NEXT stance segment.
    """
    y=np.asarray(y0,dtype=float).copy()
    phase='stance'
    stage_time=0.
    time=float(start_time)
    launch_z=None
    apex_z=-np.inf
    max_wrel=0.;max_motor_power=0.;max_electrical_power=0.
    slip_steps=0;max_penetration_correction=0.
    touchdown_impulse_loss=0.;motor_mechanical_work=0.;electrical_energy=0.;damper_loss=0.
    energy_initial=mechanical_energy(y,p)
    trajectory=[(time,y.copy(),phase)] if record else []
    control_trace=[]
    contact_trace=[]
    n_steps=0
    while True:
        prev=y.copy()
        y,diag=step(y,float(stance_ref),float(flight_ref),phase=='stance',p,c,config)
        n_steps+=1
        time+=config.dt
        stage_time+=config.dt
        max_wrel=max(max_wrel,abs(diag['motor']['omega_relative']))
        max_motor_power=max(max_motor_power,abs(diag['motor']['p_mechanical']))
        max_electrical_power=max(max_electrical_power,abs(diag['motor']['p_electrical']))
        slip_steps+=int(diag['foot_slip'])
        max_penetration_correction=max(max_penetration_correction,diag['positional_correction'])
        motor_mechanical_work+=diag['motor']['p_mechanical']*config.dt
        electrical_energy+=diag['motor']['p_electrical']*config.dt
        if prev[3]<p.leg_free_length:
            damper_loss+=p.spring_c*prev[8]**2*config.dt
        if record and n_steps%max(1,config.record_stride)==0:
            trajectory.append((time,y.copy(),phase))
            control_trace.append((time,diag['motor']['torque'],diag['motor']['current'],
                                  diag['motor']['voltage'],diag['motor']['p_electrical'],
                                  diag['motor']['p_mechanical'],diag['motor']['omega_relative']))
            fi=next((e for e in diag['impulses'] if e['name']=='foot'),None)
            contact_trace.append((time,fi['normal_impulse'] if fi else 0.,
                                  fi['tangent_impulse'] if fi else 0.,
                                  fi['tangential_speed'] if fi else 0.))
        if diag['body_contact'] and not config.allow_body_contact:
            event='body_collision';break
        if phase=='stance':
            pos,_=foot_kinematics(y[:5]);gap=float(pos[1]-p.foot_radius)
            if (not diag['foot_contact'] and gap > 0.008 and
                y[6]>0.05 and stage_time>0.015):
                phase='flight'
                launch_z=float(y[1]);apex_z=float(y[1]);stage_time=0.
            elif stage_time>config.max_stance_time:
                event='no_takeoff';break
        else:
            apex_z=max(apex_z,float(y[1]))
            if diag['foot_contact']:
                touchdown_impulse_loss=max(0.,-diag['kinetic_impulse_energy_change'])
                event='touchdown';break
            if stage_time>config.max_flight_time:
                event='no_touchdown';break
        if time-start_time>config.max_stance_time+config.max_flight_time+0.2:
            event='timeout';break

    if record:
        trajectory.append((time,y.copy(),phase))
    angle=wrap_angle(prev[2])
    touchdown_rate=float(prev[7])
    flight_rise=0. if launch_z is None else max(0.,apex_z-launch_z)
    # The previous version only checked pre-impact pitch and rate. The ground
    # impulse can generate a large post-impact angular velocity even when the
    # arriving body is upright. Carry both sides of the impact explicitly.
    post_pitch=wrap_angle(float(y[2]))
    post_rate=float(y[7])
    corner_clearance=min(float(pt[1][1]) for pt in body_corners(y[:5],p))
    diagnostics=dict(
        event=event,phase=phase,total_duration=float(time-start_time),
        time=float(time),stance_ref=float(stance_ref),flight_ref=float(flight_ref),
        flight_duration=float(stage_time) if phase=='flight' else 0.,
        flight_rise=float(flight_rise),
        touchdown_pitch=float(angle),touchdown_rate=touchdown_rate,
        postimpact_pitch=float(post_pitch),postimpact_rate=post_rate,
        postimpact_vx=float(y[5]),postimpact_vz=float(y[6]),
        postimpact_spring_speed=float(y[8]),
        postimpact_corner_clearance=float(corner_clearance),
        foot_x=float(foot_kinematics(y[:5])[0][0]),
        com_x=float(y[0]),body_collision=(event=='body_collision'),
        max_wheel_speed=max_wrel,max_mechanical_power=max_motor_power,
        max_electrical_power=max_electrical_power,
        slip_duration=float(slip_steps*config.dt),
        max_position_correction=float(max_penetration_correction),
        touchdown_impulse_loss=float(touchdown_impulse_loss),
        motor_mechanical_work=float(motor_mechanical_work),
        electrical_energy=float(electrical_energy),
        spring_damper_loss=float(damper_loss),
        energy_initial=float(energy_initial),energy_final=mechanical_energy(y,p),
        n_steps=n_steps,
    )
    errors=[]
    if event!='touchdown':errors.append(event)
    else:
        if stage_time<config.min_flight_duration:errors.append('degenerate_flight_time')
        if flight_rise<config.min_flight_rise:errors.append('degenerate_flight_rise')
        if abs(angle)>config.max_touchdown_pitch:errors.append('unsafe_touchdown_pitch')
        if abs(touchdown_rate)>config.max_touchdown_rate:errors.append('unsafe_touchdown_pitch_rate')
    diagnostics['ok']=not errors
    diagnostics['errors']=errors
    return dict(**diagnostics,state=y,trajectory=trajectory,controls=control_trace,
                contacts=contact_trace)


def rollout(y0,refs,p:Robot,c:Controller,cfg:SimulationConfig,
            goal:GoalOptions,record=False,start_time=0.):
    """A candidate sequence of (stance reference, flight reference) per hop."""
    state=np.asarray(y0,dtype=float).copy()
    steps=[];time=float(start_time)
    prev_foot_x=float(foot_kinematics(state[:5])[0][0]);crossings=0;stalls=0
    prev_error=prev_foot_x-goal.target_x
    for h,(sr,fr) in enumerate(refs):
        hop=one_hop(state,sr,fr,p,c,cfg,record=record,start_time=time)
        step_summary={key:val for key,val in hop.items() if key not in ('state','trajectory','controls','contacts')}
        step_summary['number']=h+1
        step_summary['start_foot_x']=prev_foot_x
        steps.append(step_summary)
        if not hop['ok']:
            return dict(ok=False,event=hop['errors'][0],steps=steps,
                        final_state=hop['state'],foot_x=prev_foot_x,
                        time=hop['time'],crossings=crossings)
        new_foot_x=hop['foot_x']
        error=new_foot_x-goal.target_x
        if error*prev_error < 0:crossings+=1
        prev_error=error
        stalls=stalls+1 if abs(new_foot_x-prev_foot_x)<goal.min_progress else 0
        prev_foot_x=new_foot_x
        state=hop['state'];time=hop['time']
        if stalls>=goal.stall_hops:
            return dict(ok=False,event='repeated_zero_progress',steps=steps,
                        final_state=state,foot_x=prev_foot_x,
                        time=time,crossings=crossings)
    return dict(ok=True,event='horizon_complete',steps=steps,final_state=state,
                foot_x=prev_foot_x,time=time,crossings=crossings)


def recovery_probe(state, p:Robot, c:Controller, cfg:SimulationConfig,
                   duration:float=0.10, references=None):
    """Short *physical* stance continuations from an already resolved touchdown.

    These are not a guarantee of global recoverability; they answer whether at
    least one modest emergency pitch setpoint avoids immediate body collision.
    The probe uses the same actual contact+motor step() as the forward plant.
    It does NOT reset velocities or recharge the spring.
    """
    if references is None:
        references=(0.0, -0.5*c.max_stance_ref, 0.5*c.max_stance_ref)
    attempts=[]
    n=max(1,int(math.ceil(duration/cfg.dt)))
    for sr in references:
        y=np.asarray(state,float).copy()
        min_clearance=float('inf'); min_z=float('inf')
        event='survived_probe'; takeoff=False
        for i in range(n):
            y,diag=step(y,float(sr),0.,True,p,c,cfg)
            min_clearance=min(min_clearance,min(pt[1][1] for pt in body_corners(y[:5],p)))
            min_z=min(min_z,float(y[1]))
            if diag['body_contact'] or min_clearance < -cfg.contact_margin*2:
                event='body_collision';break
            fpos,_=foot_kinematics(y[:5]);gap=float(fpos[1]-p.foot_radius)
            if not diag['foot_contact'] and gap>0.008 and y[6]>0.05 and i*cfg.dt>0.015:
                event='takeoff_during_probe'; takeoff=True;break
        attempts.append(dict(ref=float(sr),event=event,survived=event!='body_collision',
                             min_corner_clearance=float(min_clearance),
                             ending_pitch=wrap_angle(float(y[2])),ending_pitch_rate=float(y[7]),
                             ending_vx=float(y[5]),elapsed=(i+1)*cfg.dt,
                             takeoff=takeoff))
    surviving=[z for z in attempts if z['survived']]
    best=max(surviving,key=lambda z:(z['min_corner_clearance'],-abs(z['ending_pitch_rate']))) if surviving else None
    return dict(viable=bool(surviving),best=best,attempts=attempts,
                horizon=duration)


def full_hop_probe(state, p:Robot, c:Controller, cfg:SimulationConfig,
                   references=None, max_postimpact_rate=3.5):
    """Simulate a COMPLETE subsequent stance/flight/touchdown from impact state.

    A finite control sample is a sufficient witness of one feasible continuation,
    not a mathematical proof of viability or infeasibility. Each trial starts
    from the unmodified post-impact state and uses the actual motor/contact model.
    Unlike recovery_probe, it does not stop at the next takeoff.
    """
    y0=np.asarray(state,dtype=float)
    if references is None:
        vx=float(y0[5]); direction=1. if vx>=0 else -1.
        references=[
            (0.,0.),
            (-direction*.30*c.max_stance_ref,0.),
            (-direction*.70*c.max_stance_ref,-direction*.30*c.max_flight_ref),
            (direction*.30*c.max_stance_ref,0.),
            (0.,-direction*.50*c.max_flight_ref),
        ]
    attempts=[]
    for sr,fr in references:
        outcome=one_hop(y0,sr,fr,p,c,cfg,record=False)
        safe=(outcome['ok'] and
              abs(outcome['postimpact_rate'])<=max_postimpact_rate)
        penalty=(40*abs(outcome['touchdown_pitch'])**2+
                 5*abs(outcome['touchdown_rate'])**2+
                 2*abs(outcome['postimpact_rate'])**2+
                 .12*abs(outcome['postimpact_vx'])**2)
        if not safe: penalty+=10000
        attempts.append(dict(stance_ref=float(sr),flight_ref=float(fr),
            safe=bool(safe), event=outcome['event'],errors=outcome['errors'],
            time=float(outcome['total_duration']),
            end_x=float(outcome['foot_x']),
            postimpact_vx=float(outcome['postimpact_vx']),
            postimpact_rate=float(outcome['postimpact_rate']),
            score=float(penalty)))
    safe_attempts=[a for a in attempts if a['safe']]
    best=min(safe_attempts,key=lambda a:a['score']) if safe_attempts else None
    return dict(viable=bool(safe_attempts),best=best,attempts=attempts,
                tested_full_hops=len(attempts))
