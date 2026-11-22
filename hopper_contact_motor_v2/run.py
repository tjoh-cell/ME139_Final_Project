"""Run planar hopper with finite-foot Coulomb contact and a DC motor model."""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
from physics import Robot,Controller,SimulationConfig
from simulator import GoalOptions
from planner import PlannerOptions,run_controller
from visualize import save_plots,save_actuator_contact_plots,make_gif


def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--target',type=float,default=3.0)
    ap.add_argument('--compression',type=float,default=.4)
    ap.add_argument('--lookahead',type=int,default=2)
    ap.add_argument('--maxfev',type=int,default=12)
    ap.add_argument('--max-hops',type=int,default=8)
    ap.add_argument('--tolerance',type=float,default=.05)
    ap.add_argument('--max-crossings',type=int,default=2)
    ap.add_argument('--dt',type=float,default=.004)
    ap.add_argument('--contact-iterations',type=int,default=16)
    ap.add_argument('--mass',type=float,default=1.0)
    ap.add_argument('--spring-k',type=float,default=450.)
    ap.add_argument('--damping',type=float,default=1.0)
    ap.add_argument('--mu',type=float,default=.85)
    ap.add_argument('--voltage',type=float,default=48.)
    ap.add_argument('--current-limit',type=float,default=50.)
    ap.add_argument('--gear-ratio',type=float,default=5.)
    ap.add_argument('--max-landing-angle',type=float,default=30.)
    ap.add_argument('--max-landing-rate',type=float,default=2.)
    ap.add_argument('--recovery-probe-time',type=float,default=.10,
                    help='Legacy short probe duration if full-hop continuation check is disabled')
    ap.add_argument('--max-planned-postimpact-rate',type=float,default=3.5,
                    help='Conservative planned touchdown post-impact pitch-rate limit [rad/s]')
    ap.add_argument('--postimpact-rate-weight',type=float,default=5.,
                    help='Penalize impulse-induced body angular speed')
    ap.add_argument('--terminal-speed-weight',type=float,default=18.,
                    help='Penalize residual horizontal speed near target')
    ap.add_argument('--recoverable-deceleration',type=float,default=2.,
                    help='Heuristic arrival speed envelope [m/s^2], not a physical constraint')
    ap.add_argument('--continuation-probe-count',type=int,default=3,
                    help='Number of stance/flight samples for next COMPLETE hop viability')
    ap.add_argument('--continuation-candidates',type=int,default=1,
                    help='Shortlisted candidates checked per planning horizon')
    ap.add_argument('--no-full-next-hop',action='store_true',
                    help='Disable complete next-hop check (not recommended)')
    ap.add_argument('--emergency-maxfev',type=int,default=10,
                    help='Budget for independent active attitude-recovery search')
    ap.add_argument('--intermediate-speed-weight',type=float,default=.45,
                    help='Multiplier for momentum penalties at intermediate landings')
    ap.add_argument('--no-terminal-probe-rejection',action='store_true',
                    help='Do not reject plans when full continuation search finds no safe hop')
    ap.add_argument('--no-emergency-fallback','--no-passive-fallback',
                    action='store_true',dest='no_emergency_fallback',
                    help='Do not execute the active safety-oriented fallback when planning fails')
    ap.add_argument('--max-emergency-hops','--max-passive-fallbacks',
                    type=int,default=2,dest='max_emergency_hops',
                    help='Maximum active emergency hops before reporting planner failure')
    ap.add_argument('--output',default='results_contact_motor')
    ap.add_argument('--gif',action='store_true',help='Render animation GIF (slower)')
    args=ap.parse_args()
    p=Robot(body_mass=args.mass,spring_k=args.spring_k,spring_c=args.damping,
            friction_mu=args.mu,supply_voltage=args.voltage,
            current_limit=args.current_limit,gear_ratio=args.gear_ratio)
    c=Controller()
    cfg=SimulationConfig(dt=args.dt,contact_iterations=args.contact_iterations,
                         max_touchdown_pitch=np.deg2rad(args.max_landing_angle),
                         max_touchdown_rate=args.max_landing_rate)
    goal=GoalOptions(target_x=args.target,tolerance=args.tolerance,
                     max_crossings=args.max_crossings,safety_max_hops=args.max_hops)
    plan=PlannerOptions(lookahead=args.lookahead,maxfev=args.maxfev,
        recovery_probe_time=args.recovery_probe_time,
        postimpact_rate_weight=args.postimpact_rate_weight,
        max_planned_postimpact_rate=args.max_planned_postimpact_rate,
        terminal_speed_weight=args.terminal_speed_weight,
        recoverable_deceleration=args.recoverable_deceleration,
        require_terminal_recovery=not args.no_terminal_probe_rejection,
        passive_fallback=not args.no_emergency_fallback,
        max_passive_fallbacks=args.max_emergency_hops,
        continuation_probe_count=args.continuation_probe_count,
        continuation_candidates=args.continuation_candidates,
        require_full_next_hop=not args.no_full_next_hop,
        emergency_maxfev=args.emergency_maxfev,
        intermediate_speed_weight=args.intermediate_speed_weight)
    print('Robot',p)
    print('Motor max static output torque (at zero speed)',
          round(p.gear_ratio*p.gear_efficiency*p.torque_constant*p.current_limit,3),'Nm')
    print('Controller',c)
    print('Planner: independent horizons 1..',plan.lookahead)
    result=run_controller(target=args.target,compression=args.compression,
                          p=p,c=c,cfg=cfg,goal=goal,options=plan)
    print('\nRESULT:',result['status'])
    print(f'Final foot x={result["final_x"]:+.4f} m; goal={args.target:+.4f} m; '
          f'hops={len(result["hop_info"])}')
    for h in result['hop_info']:
        print(f'  Hop {h["hop"]}: touchdown pitch={np.degrees(h["touchdown_pitch"]):+.1f}°, '
              f'pitch rate={h["touchdown_rate"]:+.2f}rad/s, '
              f'max wheel relative speed={h["max_wheel_speed"]:.1f}rad/s, '
              f'peak output power={h["max_mechanical_power"]:.1f}W, '
              f'peak electrical power={h["max_electrical_power"]:.1f}W, '
              f'foot slip duration={h["slip_duration"]:.3f}s, '
              f'position correction={h["max_position_correction"]:.5f}m, '
              f'pre/post impact pitch rate={h["touchdown_rate"]:+.2f}/'
              f'{h["postimpact_rate"]:+.2f}rad/s, '
              f'COM vx after impact={h["postimpact_vx"]:+.2f}m/s')
    for e in result.get('recovery_events',[]):
        print(f'  Active emergency attempt at hop {e["hop"]}: {e["event"]} '
              f'({e["duration"]:.2f}s), motor work={e["motor_work"]:+.2f} J, errors={e["errors"]}')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    def make_jsonable(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,(np.generic,)):return x.item()
        if isinstance(x,Path):return str(x)
        if hasattr(x,'__dataclass_fields__'):return asdict(x)
        raise TypeError(str(type(x)))
    report={k:v for k,v in result.items() if k not in ['history','motor_history','contact_history','robot','controller',
           'sim_config','goal_config','planner_config']}
    report['config']=dict(robot=p,controller=c,sim_config=cfg,goal=goal,planner=plan)
    (out/'summary.json').write_text(json.dumps(report,indent=2,default=make_jsonable))
    if result['history']:
        with (out/'trajectory.csv').open('w',newline='') as fh:
            w=csv.writer(fh);w.writerow(['time','x','z','alpha','l','wheel_phi',
                  'vx','vz','alpha_dot','l_dot','wheel_phi_dot','phase'])
            for time,y,phase in result['history']:
                w.writerow([time]+y.tolist()+[phase])
        with (out/'motor.csv').open('w',newline='') as fh:
            w=csv.writer(fh)
            w.writerow(['time','wheel_torque_Nm','current_A','voltage_V',
                        'electrical_power_W','wheel_mechanical_power_W','relative_wheel_speed_rad_s'])
            w.writerows(result['motor_history'])
        with (out/'contacts.csv').open('w',newline='') as fh:
            w=csv.writer(fh)
            w.writerow(['time','foot_normal_impulse_Ns',
                        'foot_tangent_impulse_Ns','foot_tangent_speed_m_s'])
            w.writerows(result['contact_history'])
        save_plots(result,out/'trajectory.png')
        save_actuator_contact_plots(result,out/'motor_and_contact.png')
        if args.gif:
            make_gif(result,out/'hopper.gif')
    print('Saved outputs to',out.resolve())

if __name__=='__main__':main()
