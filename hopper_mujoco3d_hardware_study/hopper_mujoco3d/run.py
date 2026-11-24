"""Run the independent 3D MuJoCo hopping benchmark.

Example:
 python run.py --target 120 --wind-y 3 --gust-y 2 --gif
"""
import argparse
from dataclasses import replace
from pathlib import Path
import json
from model import Plant, write_xml
from control import Actuators, Guidance
from simulator import Wind, Mission, HopperSimulation, mujoco


def parser():
    p=argparse.ArgumentParser(description='Dynamic 3D MuJoCo pogo hopper with wind and independently vectored flight thrusters')
    p.add_argument('--target',type=float,default=15)
    p.add_argument('--compression',type=float,default=0.5)
    p.add_argument('--max-hops',type=int,default=100)
    p.add_argument('--max-time',type=float,default=300)
    p.add_argument('--dt',type=float,default=0.002)
    p.add_argument('--cruise-speed',type=float,default=8.0)
    p.add_argument('--braking-accel',type=float,default=2.0)
    p.add_argument('--forward-gain', type=float, default=Guidance.kp_forward)
    p.add_argument('--foot-placement-gain',type=float,default=Guidance.foot_placement_gain)
    p.add_argument('--apex-target',type=float,default=Guidance.height_target)
    p.add_argument('--position-tolerance',type=float,default=0.05)
    p.add_argument('--max-thrust-each',type=float,default=8.0)
    p.add_argument('--mission-impulse-budget',type=float,default=100.0)
    p.add_argument('--thrust-cutoff',type=float,default=7.0)
    p.add_argument('--height-limit',type=float,default=8.0)
    p.add_argument('--no-thrusters',action='store_true')
    p.add_argument('--planner',choices=('feedback','predictive'),default='feedback',
                   help='predictive: MuJoCo candidate hops + 1 next-hop viability check')
    p.add_argument('--planner-budget',type=int,default=12,
                   help='number of physical candidate hops evaluated per landing')
    p.add_argument('--next-hop-actions',type=int,default=3,
                   help='number of second-hop references tested for viability')
    p.add_argument('--wind-x',type=float,default=0)
    p.add_argument('--wind-y',type=float,default=0)
    p.add_argument('--wind-z',type=float,default=0)
    p.add_argument('--gust-x',type=float,default=0)
    p.add_argument('--gust-y',type=float,default=0)
    p.add_argument('--gust-z',type=float,default=0)
    p.add_argument('--gust-frequency',type=float,default=0.5)
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--gif',action='store_true')
    p.add_argument('--max-gif-frames',type=int,default=240)
    p.add_argument('--quiet',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='write MJCF and validate static configuration, no MuJoCo required')
    return p


def settings(args):
    plant=replace(Plant(),dt=args.dt)
    hw=replace(Actuators(),max_thrust_each=args.max_thrust_each,
               impulse_budget=args.mission_impulse_budget, cutoff_z=args.thrust_cutoff,
               ceiling_z=args.height_limit)
    guide=replace(Guidance(),cruise_speed=args.cruise_speed,
                  braking_accel=args.braking_accel,height_target=args.apex_target,
                  foot_placement_gain=args.foot_placement_gain, kp_forward=args.forward_gain,
                  position_tolerance=args.position_tolerance)
    wind=Wind(x=args.wind_x,y=args.wind_y,z=args.wind_z,
              gust_x=args.gust_x,gust_y=args.gust_y,gust_z=args.gust_z,
              gust_frequency=args.gust_frequency)
    mission=Mission(goal_x=args.target,initial_compression=args.compression,
                    max_hops=args.max_hops,max_time=args.max_time,
                    no_thrusters=args.no_thrusters,
                    planner=args.planner,planner_budget=args.planner_budget,
                    continuation_actions=args.next_hop_actions)
    if not 0 < hw.cutoff_z < hw.ceiling_z:
        raise ValueError('thrust cutoff must be below hard height limit')
    if guide.height_target >= hw.ceiling_z:
        raise ValueError('apex target must be below the height limit')
    if hw.max_thrust_each < 0 or hw.impulse_budget < 0:
        raise ValueError('thrust and impulse limits cannot be negative')
    return plant,hw,guide,wind,mission


def main(argv=None):
    args=parser().parse_args(argv)
    plant,hw,guide,wind,mission=settings(args)
    output=args.output or Path('results_3d')/f'goal_{args.target:g}m'
    output.mkdir(parents=True,exist_ok=True)
    write_xml(output/'robot.xml',plant)
    if args.dry_run:
        summary=dict(status='static_xml_generated_not_simulated',output=str(output),
                     target=args.target)
        (output/'summary.json').write_text(json.dumps(summary,indent=2))
        print(summary)
        return 0
    if mujoco is None:
        print('ERROR: mujoco package unavailable. Install dependencies with: pip install -r requirements.txt')
        return 2
    sim=HopperSimulation(plant,hw,guide,wind,mission,capture_frames=args.gif)
    summary=sim.run(progress=not args.quiet)
    sim.save(output,gif=args.gif,max_frames=args.max_gif_frames)
    print(json.dumps({k:summary[k] for k in ('status','goal','final_x','final_foot_x','final_y',
                  'position_error','hops','total_time','impulse_used','max_height',
                  'max_roll_deg','max_lateral','max_wheel_speed')},indent=2))
    print('Results:',output.resolve())
    return 0 if summary['status']=='goal_tolerance_reached' else 1


if __name__=='__main__':
    raise SystemExit(main())
