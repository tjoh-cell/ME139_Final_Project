"""End-to-end MuJoCo 3D twin pogo simulator with dynamic ground contact.

No hand-written collision impulse corrections, no teleportation at impacts.
MuJoCo integrates the rigid body, two independent spring legs, toe friction,
and their contact with the plane.  Wind and thrust are actual applied wrenches.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import csv
import json
import math
import numpy as np

from model import Plant, xml_string
from control import (Actuators, Guidance, ThrusterMemory, get_rpy, attitude_state,
                     touchdown_pitch_reference, ballistic_touchdown_time, motor_command,
                     target_vx, allocate_forces, support_differential_thrust, step_thrusters, flight_gate,
                     desired_flight_wrench)

try:
    import mujoco
except ImportError:
    mujoco = None


@dataclass(frozen=True)
class Wind:
    # Winds are air VELOCITIES in m/s in global x, y, z.
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    gust_x: float = 0.0
    gust_y: float = 0.0
    gust_z: float = 0.0
    gust_frequency: float = 0.5
    air_density: float = 1.225
    drag_coefficient: float = 1.0
    area: float = 0.06
    cp_offset_z: float = 0.03

    def velocity(self, t: float) -> np.ndarray:
        w = 2*math.pi*self.gust_frequency*t
        return np.array([self.x+self.gust_x*math.sin(w),
                         self.y+self.gust_y*math.sin(w+0.8),
                         self.z+self.gust_z*math.sin(w+1.7)])

    def drag_force(self, t: float, body_velocity: np.ndarray) -> np.ndarray:
        rel = self.velocity(t) - body_velocity
        return .5*self.air_density*self.drag_coefficient*self.area*np.linalg.norm(rel)*rel


@dataclass(frozen=True)
class Mission:
    goal_x: float = 15.0
    initial_compression: float = 0.5
    max_hops: int = 100
    max_time: float = 300.0
    log_period: float = 0.02
    lateral_tolerance: float = 0.25
    max_lateral_excursion: float = 5.0
    no_thrusters: bool = False
    # You may adjust controller targets but these are NEVER hardware boosts.
    stop_at_goal: bool = True
    # 'feedback' is inexpensive; 'predictive' simulates candidate hops in
    # cloned MuJoCo states before committing the reference to the real plant.
    planner: str = 'feedback'
    planner_budget: int = 12
    continuation_actions: int = 3


class HopperSimulation:
    def __init__(self, plant: Plant = Plant(), actuators: Actuators = Actuators(),
                 guidance: Guidance = Guidance(), wind: Wind = Wind(),
                 mission: Mission = Mission(), capture_frames: bool = False):
        if mujoco is None:
            raise RuntimeError("MuJoCo is not installed. Run: python -m pip install -r requirements.txt")
        self.p, self.a, self.g, self.wind, self.mission = plant, actuators, guidance, wind, mission
        self.m = mujoco.MjModel.from_xml_string(xml_string(plant))
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        if not 0 <= mission.initial_compression < plant.leg_free_length-plant.leg_min_length:
            raise ValueError('Initial spring compression violates leg travel')
        length = plant.leg_free_length-mission.initial_compression
        self.d.joint('slide_left').qpos[0] = length
        self.d.joint('slide_right').qpos[0] = length
        self.d.qpos[2] = length+plant.foot_radius
        mujoco.mj_forward(self.m, self.d)
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, 'trunk')
        self.ground = self.geom_id('ground')
        self.toes = [self.geom_id('toe_left'), self.geom_id('toe_right')]
        self.body_collision = self.geom_id('body_collision')
        self.sites = [mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE,
                                        'nozzle_left'),
                      mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE,
                                        'nozzle_right')]
        self.motor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, 'flywheel_motor')
        self.memory = ThrusterMemory()
        self.wind_applied = np.zeros(3)
        self.mode = 'stance'
        self.air_timer = 0.0
        self.hops = 0
        self.last_contact = True
        self.prev_vz = 0.0
        self.status = 'running'
        self.trace = []
        self.landings = []
        self.max_height = 0.0
        self.max_roll = 0.0
        self.max_pitch = 0.0
        self.max_wheel_rate = 0.0
        self.max_lateral = 0.0
        self.powered_steps = 0
        self.max_motor_current = 0.0
        self.max_motor_voltage = 0.0
        self.frame_states = []
        self.capture_frames = capture_frames
        if capture_frames:
            self._capture_frame()
        self._next_frame_time = 1 / 30
        self.active_command = None
        self.plan_diagnostics = []
        self.need_replan = True
        self.rollout_mode = False
        self.skip_trace = False

    def _capture_frame(self):
        from thruster_display import actuator_state
        self.frame_states.append((float(self.d.time), self.d.qpos.copy(), self.d.qvel.copy(),
                                  actuator_state(self.memory)))

    def fork_for_prediction(self, command):
        """Clone dynamical state with the SAME MuJoCo model/physical limits.

        These branches are used ONLY for action selection. Nothing is added
        to the real simulation, and there are no idealized contact resets.
        """
        fork=object.__new__(HopperSimulation)
        fork.__dict__=self.__dict__.copy()
        fork.d=mujoco.MjData(self.m)
        fork.d.qpos[:]=self.d.qpos
        fork.d.qvel[:]=self.d.qvel
        if self.m.na:
            fork.d.act[:]=self.d.act
        fork.d.ctrl[:]=self.d.ctrl
        fork.d.time=self.d.time
        fork.d.qacc_warmstart[:]=self.d.qacc_warmstart
        mujoco.mj_forward(self.m,fork.d)
        fork.memory=copy.deepcopy(self.memory)
        fork.wind_applied=self.wind_applied.copy()
        fork.trace=[]
        fork.landings=[]
        fork.frame_states=[]
        fork.capture_frames=False
        fork.skip_trace=True
        fork.rollout_mode=True
        fork.need_replan=False
        fork.active_command=command
        fork.plan_diagnostics=[]
        return fork

    def _select_action_if_needed(self):
        if (self.mission.planner!='predictive' or self.rollout_mode or
                not self.need_replan or self.status!='running'):
            return
        from predictive import choose_hop
        plan, diagnostic=choose_hop(
            self, budget=self.mission.planner_budget,
            viable_next=True, next_budget=self.mission.continuation_actions)
        self.active_command=plan
        diagnostic.update(hop=self.hops+1,time=float(self.d.time),
                          chosen=(vars(plan) if plan is not None else None))
        self.plan_diagnostics.append(diagnostic)
        self.need_replan=False
        # An unsuccessful search does not prove the mechanics impossible.
        # It falls back to bounded feedback rather than forcing an unsafe hop.

    def geom_id(self, name):
        return mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, name)

    def contact_state(self):
        toe_contact = False
        body_hit = False
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            geoms = {int(c.geom1), int(c.geom2)}
            if self.ground in geoms:
                if any(i in geoms for i in self.toes):
                    toe_contact = True
                if self.body_collision in geoms:
                    body_hit = True
        clearance = min(float(self.d.geom_xpos[i, 2]-self.p.foot_radius) for i in self.toes)
        return toe_contact, body_hit, clearance

    def telemetry(self):
        R = np.asarray(self.d.xmat[self.bid]).reshape((3,3)).copy()
        rpy = get_rpy(R)
        pos = self.d.qpos[:3].copy()
        vel = self.d.qvel[:3].copy()
        # The gyro measures BODY-FRAME angular velocity, which matches local
        # roll/pitch/yaw feedback when the robot is near upright.
        omega = self.d.sensor('gyro').data.copy()
        wheel_rate = float(self.d.joint('wheel_joint').qvel[0])
        foot_touch, body_hit, clearance = self.contact_state()
        return R, rpy, pos, vel, omega, wheel_rate, foot_touch, body_hit, clearance

    def forward_step(self):
        dt = self.p.dt
        self._select_action_if_needed()
        R, rpy, pos, vel, omega, wheel_rate, foot_contact, body_hit, clear = self.telemetry()
        if body_hit:
            self.status = 'body_collision'
            return
        if pos[2] > self.a.ceiling_z + 1e-3:
            self.status = 'height_ceiling_exceeded'
            return
        if abs(pos[1]) > self.mission.max_lateral_excursion:
            self.status = 'lateral_excursion_limit'
            return

        target = target_vx(self.mission.goal_x-pos[0], self.g)
        stance = bool(foot_contact)
        if stance:
            pitch_ref = (self.active_command.stance_ref if self.active_command is not None and self.active_command.label != "velocity_feedback"
                else float(np.clip(self.g.pitch_gain_v*(target-vel[0]),
                                   -self.g.stance_pitch_limit, self.g.stance_pitch_limit)))
        else:
            t_land=ballistic_touchdown_time(float(pos[2]),float(vel[2]),0.90*self.p.leg_free_length)
            lead=touchdown_pitch_reference(R,float(omega[1]),t_land,self.g)
            # Place the feet ahead of the trunk to receive forward momentum.
            landing_angle = -math.atan2(vel[0], max(2., 9.81*t_land-vel[2]))
            placement_gain = self.g.foot_placement_gain
            if self.active_command is not None and self.active_command.foot_placement_gain is not None:
                placement_gain = self.active_command.foot_placement_gain
            base = placement_gain * landing_angle * (1+self.g.landing_pitch_lead)
            if self.active_command is not None:
                base += self.active_command.flight_ref
            pitch_ref=float(np.clip(base+lead,-self.g.flight_pitch_limit,self.g.flight_pitch_limit))
        tau, voltage, current = motor_command(rpy[1], omega[1], wheel_rate,
                                              stance, pitch_ref, self.g, self.a)
        self._last_pitch_ref = pitch_ref
        self.d.ctrl[self.motor_id] = tau
        self.max_motor_current = max(self.max_motor_current, abs(current))
        self.max_motor_voltage = max(self.max_motor_voltage, abs(voltage))
        self.max_wheel_rate = max(self.max_wheel_rate, abs(wheel_rate))
        self.max_roll = max(self.max_roll, abs(rpy[0]))
        self.max_pitch = max(self.max_pitch, abs(rpy[1]))
        self.max_lateral = max(self.max_lateral, abs(pos[1]))
        self.max_height = max(self.max_height, pos[2])

        # MuJoCo handles all contact impulses; injected wrenches are additive.
        self.d.qfrc_applied[:] = 0
        self.d.xfrc_applied[:] = 0
        self.wind_applied = self.wind.drag_force(float(self.d.time), vel)
        # Apply air drag above the COM so it can produce pitch/roll moments.
        cp = self.d.xpos[self.bid] + R @ np.array([0,0,self.wind.cp_offset_z])
        mujoco.mj_applyFT(self.m, self.d, self.wind_applied, np.zeros(3),
                          cp, self.bid, self.d.qfrc_applied)
        flight_allowed = flight_gate(clear, foot_contact, pos[2], vel[2], self.a)
        flight_allowed = flight_allowed and not self.mission.no_thrusters
        desired_vectors = np.zeros((2,3))
        desired_vx = target
        ballistic_apex = pos[2] + max(vel[2],0)**2/(2*9.81)
        if flight_allowed:
            mass = self.p.body_mass+self.p.foot_mass_total+self.p.wheel_mass
            F_body, t_roll, t_yaw, desired_vx, ballistic_apex = desired_flight_wrench(
                pos,vel,rpy,omega,self.mission.goal_x,self.g,self.a,mass,R,0.90*self.p.leg_free_length, self.p.wheel_inertia*wheel_rate)
            thrust_scale = (self.active_command.thrust_scale if self.active_command is not None else 1.0)
            # The planner scales travel and apex assistance. Retain the world-y
            # correction and both stabilizing moments even on a coasting action.
            travel_force = F_body * thrust_scale
            travel_force += (1 - thrust_scale) * (R[1] @ F_body) * R[1]
            desired_vectors = allocate_forces(travel_force, t_roll, t_yaw, self.p.half_thruster_spacing)
            support_differential_thrust(desired_vectors, self.a)
        force_list = step_thrusters(self.memory, desired_vectors, R, dt, flight_allowed, self.a)
        for sid, F in zip(self.sites, force_list):
            if np.any(F):
                mujoco.mj_applyFT(self.m, self.d, F, np.zeros(3),
                                  self.d.site_xpos[sid], self.bid, self.d.qfrc_applied)
        if any(np.linalg.norm(f)>1e-8 for f in force_list):
            self.powered_steps += 1
        force_sum = force_list.sum(axis=0)
        self.prev_vz = float(vel[2])
        mujoco.mj_step(self.m, self.d)
        mujoco.mj_forward(self.m, self.d)
        self._transition_and_log(rpy, pos, vel, omega, tau, current, voltage,
                                 force_sum, stance, desired_vx, ballistic_apex)

    def _transition_and_log(self, pre_rpy, pre_pos, pre_vel, pre_omega,
                            tau, current, voltage, force_sum, was_stance, desired_vx, apex):
        toe_contact, body_hit, clearance = self.contact_state()
        R, rpy, pos, vel, omega, wheel_rate, _, _, _ = self.telemetry()
        pitch_full, outplane_correction, outplane_norm = attitude_state(R)
        if body_hit:
            self.status = 'body_collision'
        elif pos[2] > self.a.ceiling_z + 0.001:
            self.status = 'height_ceiling_exceeded'
        elif abs(pos[1]) > self.mission.max_lateral_excursion:
            self.status = 'lateral_excursion_limit'
        if not toe_contact:
            self.air_timer += self.p.dt
            self.mode = 'flight'
        elif toe_contact and not self.last_contact and self.air_timer >= 0.08:
            self.hops += 1
            contacting = {int(g) for c in self.d.contact if self.ground in (c.geom1, c.geom2)
                          for g in (c.geom1, c.geom2) if g in self.toes}
            xfoot, yfoot = np.mean([self.d.geom_xpos[i, :2] for i in contacting], axis=0)
            landing = dict(hop=self.hops, time=float(self.d.time), foot_x=xfoot,
                           foot_y=float(yfoot), left_contact=self.toes[0] in contacting,
                           right_contact=self.toes[1] in contacting,
                           com_x=float(pos[0]), com_y=float(pos[1]),
                           pitch=float(rpy[1]), roll=float(rpy[0]),
                           yaw=float(rpy[2]), vx=float(vel[0]), vy=float(vel[1]),
                           pitch_rate=float(omega[1]), roll_rate=float(omega[0]),
                           impulse_used=float(self.memory.impulse_used),
                           offplane_tilt=float(outplane_norm))
            self.landings.append(landing)
            if self.status == 'running':
                self._check_touchdown(xfoot, yfoot, pitch_full, outplane_norm, omega)
            self.air_timer = 0.0
            self.mode = 'stance'
            self.need_replan=True
        elif toe_contact:
            self.mode = 'stance'
            self.air_timer = 0.0
        self.last_contact = toe_contact
        if self.skip_trace:
            return
        row = dict(time=float(self.d.time), x=float(pos[0]), y=float(pos[1]), z=float(pos[2]),
                   vx=float(vel[0]), vy=float(vel[1]), vz=float(vel[2]),
                   roll=float(rpy[0]), pitch=float(pitch_full), yaw=float(rpy[2]),
                   outplane_tilt=float(outplane_norm),
                   planar_tilt_correction_x=float(outplane_correction[0]),
                   planar_tilt_correction_z=float(outplane_correction[2]),
                   omega_x=float(omega[0]), omega_y=float(omega[1]), omega_z=float(omega[2]),
                   leg_left=float(self.d.joint('slide_left').qpos[0]),
                   leg_right=float(self.d.joint('slide_right').qpos[0]),
                   wheel_speed=float(wheel_rate), motor_tau=float(tau),
                   motor_I=float(current), motor_V=float(voltage),
                   thrust_fx=float(force_sum[0]), thrust_fy=float(force_sum[1]),
                   thrust_fz=float(force_sum[2]), T1=float(self.memory.thrust[0]),
                   T2=float(self.memory.thrust[1]),
                   theta1=float(self.memory.theta[0]), theta2=float(self.memory.theta[1]),
                   phi1=float(self.memory.phi[0]), phi2=float(self.memory.phi[1]),
                   wind_fx=float(self.wind_applied[0]), wind_fy=float(self.wind_applied[1]),
                   wind_fz=float(self.wind_applied[2]), wind_speed_x=float(self.wind.velocity(self.d.time)[0]),
                   wind_speed_y=float(self.wind.velocity(self.d.time)[1]),
                   wind_speed_z=float(self.wind.velocity(self.d.time)[2]),
                   impulse=float(self.memory.impulse_used),
                   apex_prediction=float(apex), vx_target=float(desired_vx),
                   pitch_ref=float(self._last_pitch_ref),
                   pre_step_toe_contact=was_stance, pre_step_vz=float(pre_vel[2]),
                   toe_contact=toe_contact, body_contact=body_hit, toe_clearance=clearance,
                   mode=self.mode, hops=self.hops, goal=self.mission.goal_x,
                   status=self.status)
        self.trace.append(row)
        if self.capture_frames and (self.d.time >= self._next_frame_time or self.status != 'running'):
            self._capture_frame()
            self._next_frame_time += 1 / 30

    def _check_touchdown(self, xfoot, yfoot, pitch_full, outplane_norm, omega):
        if (abs(pitch_full) > self.g.safety_pitch or
                outplane_norm > self.g.safety_roll or
                abs(omega[0]) > 3.5 or abs(omega[1]) > 3.5):
            self.status = 'unsafe_touchdown_attitude'
        elif abs(xfoot-self.mission.goal_x) <= self.g.position_tolerance and \
                abs(yfoot)<=self.mission.lateral_tolerance and self.mission.stop_at_goal:
            self.status = 'goal_tolerance_reached'
        elif self.hops >= self.mission.max_hops:
            self.status = 'safety_hop_limit'

    def run(self, progress=False):
        next_status_time = 0.
        mujoco.mj_forward(self.m,self.d)
        # Crucial: MuJoCo-derived contact state, not idealized pinned-foot state.
        self.last_contact = self.contact_state()[0]
        while self.status == 'running' and self.d.time < self.mission.max_time:
            self.forward_step()
            if progress and self.d.time >= next_status_time:
                print(f't={self.d.time:6.1f}s x={self.d.qpos[0]:8.2f}m '
                      f'y={self.d.qpos[1]:+6.2f}m '
                      f'hops={self.hops:3d} impulse={self.memory.impulse_used:.2f}Ns '
                      f'status={self.status}',flush=True)
                next_status_time += 5.
        if self.status == 'running':
            self.status = 'mission_time_limit'
        if not self.trace:
            # A failure at t=0 must still produce a useful report.
            self.trace.append(dict(time=float(self.d.time), x=float(self.d.qpos[0]),
                                   y=float(self.d.qpos[1]), z=float(self.d.qpos[2]),
                                   hops=0, impulse=0, status=self.status))
        return self.summary()

    def summary(self):
        last = self.trace[-1]
        foot_x = (float(self.landings[-1]['foot_x']) if self.landings
                  else float(np.mean([self.d.geom_xpos[i,0] for i in self.toes])))
        return dict(status=self.status, goal=self.mission.goal_x,
                    final_x=last['x'], final_foot_x=foot_x,
                    final_y=last['y'], final_z=last['z'],
                    position_error=float(foot_x-self.mission.goal_x),
                    hops=self.hops, total_time=float(self.d.time),
                    impulse_used=self.memory.impulse_used,
                    thrust_powered_steps=self.powered_steps,
                    max_height=self.max_height, max_roll_deg=math.degrees(self.max_roll),
                    max_pitch_deg=math.degrees(self.max_pitch),
                    planned_hops=self.plan_diagnostics,
                    max_lateral=self.max_lateral,
                    max_wheel_speed=self.max_wheel_rate,
                    max_motor_voltage=self.max_motor_voltage,
                    max_motor_current=self.max_motor_current,
                    plant=asdict(self.p), actuators=asdict(self.a), guidance=asdict(self.g),
                    wind=asdict(self.wind), mission=asdict(self.mission),
                    landings=self.landings)

    def save(self, folder, gif=False, max_frames=250):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        summary=self.summary()
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        (folder/'planner_diagnostics.json').write_text(
            json.dumps(self.plan_diagnostics,indent=2),encoding='utf-8')
        with (folder/'trajectory.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(self.trace[0]))
            writer.writeheader()
            for row in self.trace:
                # In case t=0 failure produces a shorter row, fill missing fields.
                writer.writerow({k:row.get(k,'') for k in writer.fieldnames})
        with (folder/'landings.csv').open('w',newline='',encoding='utf-8') as f:
            if self.landings:
                writer=csv.DictWriter(f,fieldnames=list(self.landings[0]))
                writer.writeheader();writer.writerows(self.landings)
            else:
                f.write('hop,time,foot_x\n')
        try:
            self._plots(folder)
        except Exception as exc:
            (folder/'plot_error.txt').write_text(f'{type(exc).__name__}: {exc}')
        if gif:
            try:
                # Re-running for GIF is intentional: we do not store every
                # per-step qpos for long 240m trajectories in the log.
                # This is a diagnostic rather than a guaranteed renderer.
                self._make_gif(folder,max_frames)
            except Exception as exc:
                (folder/'gif_error.txt').write_text(f'{type(exc).__name__}: {exc}')
        return folder

    def _plots(self, folder):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        rows=self.trace
        def v(key):return np.array([r.get(key,0.0) for r in rows],float)
        t=v('time')
        fig,axes=plt.subplots(4,1,figsize=(12,11),sharex=True,layout='constrained')
        axes[0].plot(v('x'),v('z'),lw=1.2,label='3D COM x-z projection')
        axes[0].axvline(self.mission.goal_x,linestyle='--',label='goal')
        axes[0].axhline(self.a.ceiling_z,linestyle=':',label='height ceiling')
        axes[0].set_ylabel('COM height [m]');axes[0].legend(loc='upper right')
        axes[1].plot(t,v('y'),label='lateral position y')
        axes[1].plot(t,v('pitch')*180/math.pi,label='full-range planar pitch [deg]')
        axes[1].plot(t,v('pitch_ref')*180/math.pi,label='pitch reference [deg]',alpha=.7)
        axes[1].set_ylabel('pitch [deg]');axes[1].legend()
        axroll=axes[2]
        # For historical output compatibility, the remaining panels still show
        # thrust/impulse; separate attitude diagnostics below show out-of-plane.

        axes[2].plot(t,v('thrust_fx'),label='Fx')
        axes[2].plot(t,v('thrust_fy'),label='Fy')
        axes[2].plot(t,v('thrust_fz'),label='Fz')
        axes[2].set_ylabel('thrust [N]');axes[2].legend()
        axes[3].plot(t,v('impulse'),label='thrust impulse')
        axes[3].axhline(self.a.impulse_budget,ls=':',label='budget')
        axes[3].set_ylabel('impulse [Ns]');axes[3].set_xlabel('time [s]');axes[3].legend()
        fig.suptitle(f'MuJoCo 3D hopper: goal {self.mission.goal_x} m; {self.status}')
        fig.savefig(folder/'diagnostics.png',dpi=130)
        plt.close(fig)
        fig,axs=plt.subplots(3,1,figsize=(12,8),sharex=True,layout='constrained')
        axs[0].plot(t,v('roll')*180/math.pi,label='out-of-plane roll error [deg]')
        axs[0].plot(t,v('yaw')*180/math.pi,label='out-of-plane yaw error [deg]')
        axs[0].plot(t,v('outplane_tilt')*180/math.pi,label='out-of-plane tilt magnitude [deg]')
        axs[0].axhline(math.degrees(self.g.safety_roll),ls=':',label='tilt limit')
        axs[0].set_ylabel('out-of-plane [deg]');axs[0].legend()
        axs[1].plot(t,v('omega_y'),label='body pitch rate')
        axs[1].plot(t,v('wheel_speed'),label='relative wheel speed')
        axs[1].set_ylabel('angular speed [rad/s]');axs[1].legend()
        axs[2].plot(t,v('leg_left'),label='left pogo length')
        axs[2].plot(t,v('leg_right'),label='right pogo length')
        axs[2].set_xlabel('time [s]');axs[2].set_ylabel('leg length [m]');axs[2].legend()
        fig.savefig(folder/'attitude_and_legs.png',dpi=130)
        plt.close(fig)
        fig=plt.figure(figsize=(11,6))
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(v('x'),v('y'),v('z'),lw=1.5,label='COM trajectory')
        ax.scatter([self.mission.goal_x],[0],[0],marker='x',s=80,label='ground goal')
        ax.set_xlabel('x [m]');ax.set_ylabel('y (out of plane) [m]');ax.set_zlabel('z (up) [m]')
        ax.set_title(f'3D trajectory — {self.status}')
        ax.legend()
        fig.savefig(folder/'trajectory_3d.png',dpi=140)
        plt.close(fig)

    def _make_gif(self, folder, max_frames):
        """Capture distinct physical qpos snapshots with a readable timestamp.

        Using Pillow's explicit save_all/append_images prevents a silent
        single-frame GIF. We never invent intermediate robot states.
        """
        if len(self.frame_states)<2:
            raise RuntimeError('Need at least two captured states; run with --gif')
        from thruster_display import draw_nozzles, readout
        from PIL import Image, ImageDraw
        renderer=mujoco.Renderer(self.m,height=450,width=800)
        cam=mujoco.MjvCamera()
        cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance=5.5;cam.azimuth=105;cam.elevation=-9
        frames=[]
        indices = np.linspace(0, len(self.frame_states)-1,
                              min(len(self.frame_states), max(2, max_frames))).round().astype(int)
        times = []
        rd=mujoco.MjData(self.m)
        for index in indices:
            t, qpos, qvel, state = self.frame_states[index]
            times.append(t)
            rd.qpos[:]=qpos;rd.qvel[:]=qvel;rd.time=t
            mujoco.mj_forward(self.m,rd)
            cam.lookat[:]=[float(qpos[0]),float(qpos[1]),max(1.45,float(qpos[2])-.55)]
            renderer.update_scene(rd,camera=cam)
            draw_nozzles(renderer.scene, rd, self.bid, self.sites, state, self.a.max_thrust_each)
            frame=Image.fromarray(renderer.render().copy()).convert('RGB')
            draw=ImageDraw.Draw(frame)
            toe_contact = any(self.ground in (c.geom1, c.geom2) and
                              (c.geom1 in self.toes or c.geom2 in self.toes) for c in rd.contact)
            text = readout(t, qpos[0], qpos[2], self.mission.goal_x, toe_contact, state)
            box = draw.multiline_textbbox((12,10), text)
            draw.rectangle((4,4,box[2]+9,box[3]+6),fill=(15,20,26))
            draw.multiline_text((12,10), text, fill='white')
            frames.append(frame)
        renderer.close()
        if len(frames)<2:
            raise RuntimeError('GIF capture yielded fewer than 2 frames')
        # GIF delays have 10 ms resolution. Quantize absolute timestamps so
        # rounding individual delays doesn't accumulate over the mission.
        ticks = np.rint(np.asarray(times)*100).astype(int)
        durations = (10*np.maximum(1, np.diff(ticks))).tolist()
        frames[0].save(folder/'hopper.gif',save_all=True,append_images=frames[1:],
                       duration=durations+[durations[-1]],loop=0,optimize=False,disposal=2)
        with Image.open(folder/'hopper.gif') as im:
            if getattr(im,'n_frames',1)<2:
                raise RuntimeError('Encoded GIF contains only one frame')
