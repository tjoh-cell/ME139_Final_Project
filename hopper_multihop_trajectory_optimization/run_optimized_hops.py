"""CLI: Multi-hop shooting optimization and 2D animation, no thrusters."""
import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from hybrid_hopper import Robot, Control
from trajectory_planner import PlanningOptions, run_optimized_hops
from legacy_run_multi_hop import save_outputs


def main():
    a=argparse.ArgumentParser(description='Hybrid 2D hopper: joint N-hop plan optimization over multiple N')
    a.add_argument('--target',type=float,default=.45,help='Final foot x position [m]')
    a.add_argument('--compression',type=float,default=.12)
    a.add_argument('--scenario',choices=['baseline','aggressive'],default='baseline',
                   help='Aggressive matches your revised 1 m leg/16 Nm motor example')
    a.add_argument('--mass',type=float)
    a.add_argument('--body-inertia',type=float)
    a.add_argument('--wheel-inertia',type=float)
    a.add_argument('--stiffness',type=float)
    a.add_argument('--damping',type=float)
    a.add_argument('--l0',type=float)
    a.add_argument('--min-length',type=float)
    a.add_argument('--tau-max',type=float)
    a.add_argument('--wheel-speed-max',type=float)
    a.add_argument('--mu',type=float)
    a.add_argument('--stance-kp',type=float)
    a.add_argument('--stance-kd',type=float)
    a.add_argument('--max-tilt',type=float,help='Maximum tilt REFERENCE [degrees], not guaranteed takeoff angle')
    a.add_argument('--lookahead',type=int,default=3,help='Compare 1..N optimized hop sequences at a time')
    a.add_argument('--safety-max-hops',type=int,default=8,help='Emergency simulation cap, NOT chosen number of hops')
    a.add_argument('--stop-after-crossings',type=int,default=2,help='Stop after this many reversals across goal; default 2')
    a.add_argument('--tolerance',type=float,default=.05)
    a.add_argument('--maxfev',type=int,default=70,help='Powell max evaluations for each candidate N')
    a.add_argument('--execute-plan',action='store_true',help='Execute chosen multi-hop plan before replanning')
    a.add_argument('--output',default='results_optimized')
    a.add_argument('--no-gif',action='store_true')
    args=a.parse_args()
    p=Robot(); c=Control()
    if args.scenario=='aggressive':
        p=replace(p,l0=1.,tau_max=16.,k=450.,c=1.,m=1.,
                  Ib=.015,Iw=.003,min_length=.18,mu=1.5,
                  wheel_speed_max=160.,max_flight=3.2)
        c=replace(c,stance_kp=20.,max_tilt_reference=.65)
    params=dict(m=args.mass,Ib=args.body_inertia,Iw=args.wheel_inertia,
                k=args.stiffness,c=args.damping,l0=args.l0,
                min_length=args.min_length,tau_max=args.tau_max,
                wheel_speed_max=args.wheel_speed_max,mu=args.mu)
    p=replace(p,**{key:val for key,val in params.items() if val is not None})
    ctl=dict(stance_kp=args.stance_kp,stance_kd=args.stance_kd,
             max_tilt_reference=None if args.max_tilt is None else args.max_tilt*3.141592653589793/180.)
    c=replace(c,**{key:val for key,val in ctl.items() if val is not None})
    if args.lookahead<1 or args.safety_max_hops<1 or args.stop_after_crossings<1:
        a.error('lookahead, safety-max-hops, stop-after-crossings must be positive')
    opts=PlanningOptions(lookahead=args.lookahead,safety_max_hops=args.safety_max_hops,
             max_crossings=args.stop_after_crossings,arrival_tolerance=args.tolerance,
             maxfev_per_horizon=args.maxfev,
             replan_each_hop=not args.execute_plan)
    print(f'Robot: {p}\nControl: {c}')
    print(f'Compare 1..{opts.lookahead} hops, target={args.target:+.3f} m, '
          f'stop after {opts.max_crossings} goal crossings or tolerance')
    result=run_optimized_hops(p,c,args.target,args.compression,opts)
    print(f"RESULT: {result['status']}; x={result['final_foot_x']:+.3f} m; "
          f"executed={len(result['hops'])} hops; crossings={result['goal_crossings']}")
    output=Path(args.output);save_outputs(p,result,output,make_gif=not args.no_gif)
    with (output/'optimization.json').open('w') as f:
        json.dump({'robot':asdict(p),'control':asdict(c),
                   'planning_options':asdict(opts),'status':result['status'],
                   'target':args.target,'hops':result['hops'],
                   'peak_relative_wheel_speed_rad_s':result['peak_relative_wheel_speed'],
                   'peak_abs_motor_mechanical_power_W':result['peak_motor_mechanical_power'],
                   'candidates_at_each_replan':result['plans']},f,indent=2)
    print(f"Diagnostics: peak relative wheel speed {result['peak_relative_wheel_speed']:.1f} rad/s, "
          f"peak abs motor mechanical power {result['peak_motor_mechanical_power']:.1f} W")
    violations=[h['impact_friction_shortfall'] for h in result['hops']]
    if violations and max(violations)>1e-6:
        print('CONTACT WARNING: pinned-foot reset requires a tangential '
              'impulse beyond the modeled Coulomb limit. Optimizer uses '
              'a soft penalty, NOT a realistic slip/impact model.')
    print(f'Outputs: {output.resolve()}')

if __name__=='__main__': main()
