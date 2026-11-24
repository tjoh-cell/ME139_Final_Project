"""Hardware-bounded 3D flight thrust and reaction-wheel control.

Axes: world +x forward, +y lateral (out of former page), +z up.
Two side thrusters at body y=+/-b; differential vertical forces generate
roll torque about body x, differential forward forces generate yaw torque z.
Reaction wheel alone produces body pitch torque about local y.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Actuators:
    # Provisional thruster parameters: NOT selected hardware
    max_thrust_each: float = 8.0
    impulse_budget: float = 100.0
    thrust_slew: float = 120.0  # N/s, each
    gimbal_slew: float = 9.0  # rad/s
    elevation_min: float = 0.0
    elevation_max: float = math.radians(160)
    lateral_gimbal_limit: float = math.radians(45)
    cutoff_z: float = 7.0
    ceiling_z: float = 8.0
    apex_margin: float = 0.12
    # Same DC motor assumptions as the previous 2D simulator
    supply_voltage: float = 48.0
    resistance: float = 0.16
    kt: float = 0.05
    ke: float = 0.05
    gear_ratio: float = 5.0
    gear_eff: float = 0.9
    current_limit: float = 50.0
    wheel_speed_warning: float = 190.0


@dataclass(frozen=True)
class Guidance:
    cruise_speed: float = 8.0           # desired speed, NOT an actuator limit
    braking_accel: float = 2.0          # conservative desired deceleration
    height_target: float = 3.0          # desire, always below ceiling
    stance_pitch_limit: float = math.radians(35)
    pitch_gain_v: float = 0.08
    stance_kp: float = 20.0
    stance_kd: float = 0.65
    flight_kp: float = 3.0
    flight_kd: float = 0.70
    wheel_damping: float = 0.003
    kp_roll: float = 0.9
    kd_roll: float = 0.35
    kp_yaw: float = 0.5
    kd_yaw: float = 0.25
    kp_lateral: float = 2.0
    kd_lateral: float = 2.8
    kp_forward: float = 0.25
    vertical_assist: float = 10.0       # target total thrust while below desired apex
    safety_pitch: float = math.radians(30)
    safety_roll: float = math.radians(20)
    position_tolerance: float = 0.05
    # Landing pitch lead (not extra actuator authority).
    flight_pitch_limit: float = math.radians(25)
    foot_placement_gain: float = 0.55
    landing_pitch_lead: float = 0.50
    landing_time_cap: float = 0.70
    # Small body-frame tilt-error feedback during powered ascent.
    offplane_tilt_limit: float = math.radians(35)


@dataclass
class ThrusterMemory:
    thrust: np.ndarray = field(default_factory=lambda: np.zeros(2))
    theta: np.ndarray = field(default_factory=lambda: np.full(2, math.pi / 2))
    phi: np.ndarray = field(default_factory=lambda: np.zeros(2))
    impulse_used: float = 0.0
    saturation_count: int = 0


def clip(x, lo, hi):
    return float(np.clip(x, lo, hi))


def signed_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rotor_to_world(R_body: np.ndarray, T: float, theta: float, phi: float) -> np.ndarray:
    vec_body = np.array([math.cos(phi)*math.cos(theta),
                         math.sin(phi), math.cos(phi)*math.sin(theta)])
    return R_body @ (T * vec_body)


def attitude_state(R_body: np.ndarray):
    """Full-range planar pitch + out-of-plane rotation error.

    The pitch is computed from the BODY +z axis projected onto the world x-z
    plane.  A pure Ry(120 deg) gives pitch=120 deg and zero out-of-plane tilt,
    unlike XYZ Euler roll/pitch/yaw.  The returned correction vector is the
    BODY-frame rotation that would remove roll/yaw while preserving that pitch.

    Pitch is signed in [-pi,pi]; a real rotation past 180 degrees remains a
    crash/recovery situation, not a magically stable upright configuration.
    """
    R = np.asarray(R_body, dtype=float).reshape(3, 3)
    pitch = math.atan2(float(R[0, 2]), float(R[2, 2]))
    # Symmetric calm-air rollouts stay planar to round-off.
    if abs(R[1, 0]) + abs(R[1, 2]) < 1e-12 and R[1, 1] > 0:
        return pitch, np.zeros(3), 0.0
    desired_planar = Rotation.from_euler('y', pitch).as_matrix()
    correction = Rotation.from_matrix(R.T @ desired_planar).as_rotvec()
    outplane = float(np.linalg.norm(correction[[0, 2]]))
    return pitch, correction, outplane


def attitude_error_to_target(R_body: np.ndarray, pitch_target: float):
    """Shortest SO(3) BODY-FRAME correction toward a planar target attitude."""
    R_des = Rotation.from_euler('y', pitch_target).as_matrix()
    return Rotation.from_matrix(np.asarray(R_body).T @ R_des).as_rotvec()


def get_rpy(R_body: np.ndarray):
    """Backward-compatible reporting with *tilt errors*, not XYZ Euler RPY.

    roll/yaw are deviations from the closest planar pitch orientation.
    Use attitude_state() for physical control.  This diagnostic is not an
    absolute heading representation when the robot is sideways or inverted.
    """
    pitch, correction, _ = attitude_state(R_body)
    return np.array([-correction[0], pitch, -correction[2]])


def touchdown_pitch_reference(R_body: np.ndarray, body_pitch_rate: float,
                              t_land: float, guide: Guidance) -> float:
    """Predict angular drift to touchdown and begin cancelling it in flight.

    Controller still saturates at real DC motor capabilities. This is a
    heuristic angular lead, not an inverse dynamics or hard guarantee.
    """
    pitch, _, _ = attitude_state(R_body)
    future = signed_angle(pitch + body_pitch_rate*min(t_land, guide.landing_time_cap))
    # An opposing pitch command helps avoid accumulating angular momentum.
    lead = -guide.landing_pitch_lead * future
    return clip(lead, -guide.flight_pitch_limit, guide.flight_pitch_limit)


def motor_torque(request: float, relative_wheel_speed: float, a: Actuators):
    """Same current-controlled, voltage-clipped simplified DC motor as 2D.
    Calls motor with flywheel relative hinge speed (already body-relative).
    """
    omega_motor = a.gear_ratio * relative_wheel_speed
    i_req = clip(request / (a.gear_ratio*a.gear_eff*a.kt), -a.current_limit, a.current_limit)
    v_req = a.resistance*i_req + a.ke*omega_motor
    voltage = clip(v_req, -a.supply_voltage, a.supply_voltage)
    current = clip((voltage-a.ke*omega_motor)/a.resistance, -a.current_limit, a.current_limit)
    # Coast if saturation would create an unrequested torque sign.
    if abs(request)<1e-10 or current * request < 0:
        current, voltage = 0.0, 0.0
    torque = a.gear_ratio*a.gear_eff*a.kt*current
    return torque, voltage, current


def motor_command(pitch, pitch_rate, wheel_rate, stance, desired_pitch, guide: Guidance,
                  hw: Actuators):
    kp = guide.stance_kp if stance else guide.flight_kp
    kd = guide.stance_kd if stance else guide.flight_kd
    request = kp*signed_angle(pitch-desired_pitch) + kd*pitch_rate + guide.wheel_damping*wheel_rate
    return motor_torque(request, wheel_rate, hw)


def target_vx(dx: float, guide: Guidance):
    # Near the target, decelerate smoothly, allowing backward correction.
    return math.copysign(min(guide.cruise_speed,
                             math.sqrt(max(0.0, 2*guide.braking_accel*abs(dx)))), dx) if abs(dx)>0 else 0.0


def allocate_forces(F_body: np.ndarray, tau_x: float, tau_z: float, half_span: float):
    """Ideal desired force split, BEFORE gimbal/magnitude/rate clipping.
    With rL=(0,+b,0), rR=(0,-b,0),
      roll torque tau_x=b(Fz_L-Fz_R)
      yaw torque tau_z=b(Fx_R-Fx_L).
    """
    b = half_span
    if b <= 0:
        raise ValueError('Thruster mount spacing must be >0')
    return np.array([
        [F_body[0]/2 - tau_z/(2*b), F_body[1]/2, F_body[2]/2 + tau_x/(2*b)],
        [F_body[0]/2 + tau_z/(2*b), F_body[1]/2, F_body[2]/2 - tau_x/(2*b)],
    ])


def support_differential_thrust(vectors, hw):
    """Add common upward demand so roll/yaw corrections can be realized near coast."""
    # If the nozzle can point upward, lift both demands equally until the
    # x-z directions fit the elevation limits. This preserves their difference.
    if hw.elevation_min < math.pi/2 < hw.elevation_max:
        front_slope = math.tan(hw.elevation_min)
        rear_slope = math.tan(math.pi - hw.elevation_max)
        lift = max(max(fx * front_slope, -fx * rear_slope) - fz for fx, _, fz in vectors)
        vectors[:, 2] += clip(lift, 0., hw.max_thrust_each)
    return vectors


def vector_to_angles(F: np.ndarray):
    T = float(np.linalg.norm(F))
    if T < 1e-10:
        return 0.0, math.pi/2, 0.0
    theta = math.atan2(float(F[2]), float(F[0]))
    phi = math.atan2(float(F[1]), math.hypot(float(F[0]), float(F[2])))
    return T, theta, phi


def step_thrusters(mem: ThrusterMemory, desired_vectors_body: np.ndarray,
                   R_body: np.ndarray, dt: float, allowed: bool, hw: Actuators):
    """Two independent T/theta/phi commands; return actual world forces.

    When flight gate is closed, thrust is EXACTLY zero on that step, even
    though gimbal/throttle dynamics are otherwise rate-limited. Mission
    impulse is integrated from ACTUAL magnitudes, not request magnitudes.
    """
    desired_vectors_body = np.asarray(desired_vectors_body, dtype=float)
    out = np.zeros((2, 3))
    if not allowed or mem.impulse_used >= hw.impulse_budget - 1e-12:
        mem.thrust[:] = 0.0
        return out
    for i in range(2):
        demand, th, ph = vector_to_angles(desired_vectors_body[i])
        if th < hw.elevation_min or th > hw.elevation_max:
            th = min((hw.elevation_min, hw.elevation_max),
                     key=lambda limit: abs(signed_angle(th - limit)))
        ph = clip(ph, -hw.lateral_gimbal_limit, hw.lateral_gimbal_limit)
        mem.theta[i] += clip(signed_angle(th - mem.theta[i]), -hw.gimbal_slew*dt, hw.gimbal_slew*dt)
        mem.phi[i] += clip(signed_angle(ph - mem.phi[i]), -hw.gimbal_slew*dt, hw.gimbal_slew*dt)
        if demand > hw.max_thrust_each:
            mem.saturation_count += 1
        demand = min(hw.max_thrust_each, demand)
        mem.thrust[i] += clip(demand-mem.thrust[i], -hw.thrust_slew*dt, hw.thrust_slew*dt)
    combined = mem.thrust.sum()*dt
    remain = max(0.0, hw.impulse_budget-mem.impulse_used)
    if combined > remain:
        mem.thrust *= remain/combined
        combined = remain
    mem.impulse_used += combined
    for i in range(2):
        out[i] = rotor_to_world(R_body, mem.thrust[i], mem.theta[i], mem.phi[i])
    return out


def flight_gate(foot_clearance: float, foot_contact: bool, com_z: float,
                vz: float, hw: Actuators, gravity: float = 9.81):
    # Do not thrust at takeoff while any foot is still in contact, or on descent.
    if foot_contact or foot_clearance <= 0.03 or vz <= 0.08 or com_z >= hw.cutoff_z:
        return False
    remaining_ballistic_rise = vz**2/(2*gravity)
    return com_z + remaining_ballistic_rise < hw.ceiling_z-hw.apex_margin


def ballistic_touchdown_time(z: float, vz: float, height: float = 0.90,
                             gravity: float = 9.81) -> float:
    """Conservative estimate, NOT a contact solver. Actual MuJoCo impact
    decides touchdown; this estimate only informs early braking.
    """
    d=max(0.0,z-height)
    return max(0.0,(vz+math.sqrt(max(0.0,vz*vz+2*gravity*d)))/gravity)


def desired_flight_wrench(position, velocity, rpy, omega_body, goal_x,
                          guide: Guidance, hw: Actuators, mass: float,
                          R_body: np.ndarray, touchdown_height: float = 0.90, wheel_momentum: float = 0.):
    # Predict where the present ballistic flight would land before choosing
    # horizontal acceleration. Prevents late braking on fast long-range hops.
    t_land=ballistic_touchdown_time(position[2],velocity[2],touchdown_height)
    dx = goal_x-(position[0]+velocity[0]*t_land)
    vx_target = target_vx(dx, guide)
    # For far goals: high cruise; nearby: braking. Horizontal force is not
    # free to exceed thruster hardware: vector allocation enforces limits.
    fx = clip(mass*guide.kp_forward*(vx_target-velocity[0]), -2*hw.max_thrust_each, 2*hw.max_thrust_each)
    fy = clip(mass*(-guide.kp_lateral*position[1] - guide.kd_lateral*velocity[1]), -hw.max_thrust_each, hw.max_thrust_each)
    h_apex = position[2] + max(0.0, velocity[2])**2/(2*9.81)
    fz = guide.vertical_assist if h_apex < guide.height_target else 0.0
    # The body-frame force budget is allocated with differential torque.
    F_body = R_body.T @ np.array([fx, fy, fz])
    # The corrective rotation is measured in the body frame and remains
    # well-defined through large *planar pitch* angles. This avoids the
    # spurious 180-degree roll/yaw jumps from XYZ Euler conversion.
    _, tilt_correction, _ = attitude_state(R_body)
    tau_roll = guide.kp_roll*tilt_correction[0] - guide.kd_roll*omega_body[0]
    tau_yaw = guide.kp_yaw*tilt_correction[2] - guide.kd_yaw*omega_body[2]
    # Turning the wheel axis requires dH/dt = omega_desired x H.
    # Cross-coupled feedforward steers stored momentum back into the pitch plane.
    tilt_rate = 4.0
    tau_roll -= wheel_momentum * tilt_rate * tilt_correction[2]
    tau_yaw += wheel_momentum * tilt_rate * tilt_correction[0]
    return F_body, tau_roll, tau_yaw, vx_target, h_apex
