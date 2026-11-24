"""Interactive MuJoCo viewer (requires supported local OpenGL context)."""
import argparse
from bisect import bisect_right
import math
import time
from model import Plant
from control import Actuators, Guidance
from simulator import HopperSimulation, Mission, Wind


def prepare_playback(sim, progress=None):
    """Run the normal closed-loop planner and retain every accepted physics step."""
    from thruster_display import actuator_state
    times, frames = [], []
    while True:
        times.append(sim.d.time)
        frames.append((sim.d.qpos.copy(), sim.d.qvel.copy(),
                       actuator_state(sim.memory), sim.contact_state()[0]))
        if sim.status != 'running' or sim.d.time >= sim.mission.max_time:
            break
        if sim.need_replan:
            message = f'Planning hop {sim.hops + 1} at t={sim.d.time:.2f} s...'
            if progress:
                progress(message)
            else:
                print(message, flush=True)
        sim.forward_step()
    if sim.status == 'running':
        sim.status = 'mission_time_limit'
    return times, frames


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--target',type=float,default=15)
    ap.add_argument('--speed', type=float, default=1.0,
                    help='playback speed multiplier (0.25 for quarter speed, 2 for double)')
    ap.add_argument('--forward-gain', type=float, default=Guidance.kp_forward)
    ap.add_argument('--foot-placement-gain', type=float, default=Guidance.foot_placement_gain)
    ap.add_argument('--apex-target', type=float, default=Guidance.height_target)
    ap.add_argument('--planner',choices=('feedback','predictive'),default='feedback')
    ap.add_argument('--planner-budget',type=int,default=8)
    ap.add_argument('--live-planning', action='store_true',
                    help='run predictive searches in the interactive simulation instead of preparing playback')
    ap.add_argument('--wind-y',type=float,default=0)
    ap.add_argument('--gust-y',type=float,default=0)
    ap.add_argument('--impulse-budget',type=float,default=100)
    ap.add_argument('--no-thrusters', action='store_true')
    ap.add_argument('--setup', action='store_true', help='open the reusable hardware and mission settings window')
    a=ap.parse_args()
    if not math.isfinite(a.speed) or a.speed <= 0:
        ap.error('--speed must be finite and greater than zero')
    settings = (Plant(),Actuators(impulse_budget=a.impulse_budget),
                         Guidance(foot_placement_gain=a.foot_placement_gain,height_target=a.apex_target,kp_forward=a.forward_gain),
                         Wind(y=a.wind_y,gust_y=a.gust_y),Mission(no_thrusters=a.no_thrusters,goal_x=a.target,max_time=300,planner=a.planner,planner_budget=a.planner_budget))
    if a.setup:
        from hardware_setup import open_setup
        open_setup(settings, a.speed, a.live_planning)
        return
    show_simulation(HopperSimulation(*settings), a.speed, a.live_planning)


def show_simulation(sim, speed, live_planning=False, progress=None, on_result=None):
    import mujoco.viewer
    sim.skip_trace = True  # The viewer reads current state directly.
    from thruster_display import actuator_state, draw_nozzles, readout
    replay = sim.mission.planner == 'predictive' and not live_planning
    data = sim.d
    if replay:
        print('Preparing predictive playback. The viewer will open when the mission finishes.', flush=True)
        times, frames = prepare_playback(sim, progress)
        print(f'{sim.status} at {times[-1]:.2f} s. Opening playback.', flush=True)
        data = mujoco.MjData(sim.m)
        data.qpos[:], data.qvel[:] = frames[0][:2]
        mujoco.mj_forward(sim.m, data)
    if on_result and replay:
        on_result(sim)
    if progress:
        progress('Playing predictive run' if replay else 'Running live')
    with mujoco.viewer.launch_passive(sim.m,data) as viewer:
        with viewer.lock():
            viewer.cam.distance = 5.5
            viewer.cam.azimuth = 105
            viewer.cam.elevation = -9
        frame_period = 1 / 60
        start = time.perf_counter()
        achieved_speed = 0.
        while viewer.is_running():
            frame_start = time.perf_counter()
            frame_end = frame_start + frame_period
            target_time = (frame_start - start) * speed
            if replay:
                was_running = data.time < times[-1]
                index = bisect_right(times, target_time) - 1
                qpos, qvel, state, contact = frames[index]
                data.time = times[index]
                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(sim.m, data)
            else:
                was_running = sim.status == 'running'
                while (sim.status == 'running' and data.time + sim.p.dt <= target_time
                       and time.perf_counter() < frame_end):
                    sim.forward_step()
                    if sim.status == 'running' and data.time >= sim.mission.max_time:
                        sim.status = 'mission_time_limit'
                if was_running and sim.status != 'running' and on_result:
                    on_result(sim)
                state = actuator_state(sim.memory)
                contact = sim.last_contact
            if was_running and data.time > 0:
                achieved_speed = data.time / (time.perf_counter() - start)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                draw_nozzles(viewer.user_scn, data, sim.bid, sim.sites, state, sim.a.max_thrust_each)
                viewer.cam.lookat[:] = [float(data.qpos[0]),float(data.qpos[1]),
                                        max(1.45,float(data.qpos[2])-.55)]
            hud = readout(data.time, data.qpos[0], data.qpos[2], sim.mission.goal_x, contact, state)
            if replay:
                hud += '\nPredictive replay'
            hud += f'\nPlayback {speed:g}x requested | {achieved_speed:.1f}x average'
            viewer.set_texts((None, mujoco.mjtGridPos.mjGRID_TOPRIGHT, hud, None))
            viewer.sync()
            remaining = frame_end - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)


if __name__=='__main__':
    main()
