"""Explicitly VERIFIES or FAILS five goals with the real MuJoCo simulator.

Do NOT confuse the previous 2D successful trajectories with 3D validation.
Every entry is actually simulated from rest, with the same hardware and
requested wind. Optional controller-gain profiles are separate searches.
"""
import argparse
import csv
from dataclasses import replace
from pathlib import Path
from model import Plant, write_xml
from control import Actuators, Guidance
from simulator import Wind, Mission, HopperSimulation, mujoco

GOALS=(15.0,30.0,60.0,120.0,240.0)
# Name, cruise speed, braking acceleration, apex, foot placement, forward gain.
PROFILES = (
    ('stance_swing', 8.0, 2.0, Guidance.height_target, Guidance.foot_placement_gain, Guidance.kp_forward),
    ('stance_range', 8.0, 2.0, 2.2, 0.44, 0.25),
    ('stance_long_range', 8.0, 2.0, 2.2, 0.415, 0.25),
    ('balanced', 8.0, 2.0, 6.8, 1.0, 2.0),
    ('faster', 10.0, 2.5, 6.8, 1.0, 2.0),
    ('conservative', 6.0, 1.5, 6.6, 1.0, 2.0),
    ('high', 8.0, 2.0, 7.25, 1.0, 2.0),
    ('range', 8.0, 2.0, 6.8, 0.55, 2.0),
    ('low_apex', 8.0, 2.0, 1.8, 0.44, 2.0),
)


def run_suite(goals=GOALS,folder='benchmarks_3d',profiles=1,wind=Wind(),
              impulse_budget=100.0, max_hops=120, max_time=350,
              gif=False, no_thrusters=False, planner="feedback", planner_budget=8):
    if mujoco is None:
        raise RuntimeError('MuJoCo not installed; cannot verify 3D goals')
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    p=Plant();hw=replace(Actuators(),impulse_budget=impulse_budget)
    write_xml(folder/'robot.xml',p)
    rows=[]
    for target in goals:
        best=None
        for name,v,brake,apex,placement,forward_gain in PROFILES[:profiles]:
            g=replace(Guidance(),cruise_speed=v,braking_accel=brake,height_target=apex,
                      foot_placement_gain=placement,
                      kp_forward=forward_gain)
            mission=Mission(goal_x=target,initial_compression=0.5,max_hops=max_hops,
                            max_time=max_time,no_thrusters=no_thrusters,
                            planner=planner,planner_budget=planner_budget)
            print(f'Goal {target:g} m; controller={name}; identical spring/motor/thrusters',flush=True)
            sim=HopperSimulation(p,hw,g,wind,mission,capture_frames=gif)
            summary=sim.run(progress=False)
            sim.save(folder/f'goal_{target:g}_{name}',gif=gif)
            print(f' -> {summary["status"]}; x={summary["final_x"]:.3f}; '
                  f'hops={summary["hops"]}; wind-y peak excursion={summary["max_lateral"]:.3f}',flush=True)
            row=dict(goal=target,profile=name,status=summary['status'],
                     final_x=summary['final_x'],last_touchdown_x=summary['landings'][-1]['foot_x'] if summary['landings'] else None,
                     abs_error=abs(summary['position_error']),
                     hops=summary['hops'],time_s=summary['total_time'],
                     impulse_Ns=summary['impulse_used'],max_height=summary['max_height'],
                     max_roll_deg=summary['max_roll_deg'],max_lateral=summary['max_lateral'])
            rows.append(row)
            if row['status']=='goal_tolerance_reached':
                best=row
                break
        if best is None:
            print(f'Goal {target:g} m not verified: no valid arrival in configured search',flush=True)
    with (folder/'benchmark_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--targets',type=float,nargs='+',default=list(GOALS))
    ap.add_argument('--profiles',type=int,choices=range(1,len(PROFILES)+1),default=1,
                    help='try additional controller tunings; never modifies hardware')
    ap.add_argument('--impulse-budget',type=float,default=100.)
    ap.add_argument('--max-hops',type=int,default=120)
    ap.add_argument('--max-time',type=float,default=350.)
    ap.add_argument('--wind-x',type=float,default=0.)
    ap.add_argument('--wind-y',type=float,default=0.)
    ap.add_argument('--wind-z',type=float,default=0.)
    ap.add_argument('--gust-y',type=float,default=0.)
    ap.add_argument('--output',default='benchmarks_3d')
    ap.add_argument('--gif',action='store_true')
    ap.add_argument('--no-thrusters',action='store_true')
    ap.add_argument('--planner',choices=('feedback','predictive'),default='feedback')
    ap.add_argument('--planner-budget',type=int,default=8)
    a=ap.parse_args()
    rows=run_suite(goals=a.targets,folder=a.output,profiles=a.profiles,
                   impulse_budget=a.impulse_budget,max_hops=a.max_hops,
                   max_time=a.max_time,wind=Wind(x=a.wind_x,y=a.wind_y,z=a.wind_z,
                                                gust_y=a.gust_y),gif=a.gif,no_thrusters=a.no_thrusters,
                   planner=a.planner,planner_budget=a.planner_budget)
    passed=set(r['goal'] for r in rows if r['status']=='goal_tolerance_reached')
    print('\n3D validation summary (real MuJoCo outcomes):')
    for target in a.targets:
        print(f'{target:g} m: {"PASS" if target in passed else "NOT VERIFIED"}')
    if len(passed)<len(a.targets):
        raise SystemExit(1)


if __name__=='__main__':
    main()
