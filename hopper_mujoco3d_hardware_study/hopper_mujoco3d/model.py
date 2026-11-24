"""Generates a representative, fully dynamic MuJoCo MJCF twin-pogo robot.

World coordinates: x forward, y out-of-plane, z up.  Both legs are passive.
This is deliberately NOT a CAD/detailed joint design: their independent slide
joints and the rigid body are an idealized representative pogo mechanism.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Plant:
    body_mass: float = 1.0
    foot_mass_total: float = 0.035
    body_inertia_pitch: float = 0.015
    wheel_inertia: float = 0.003
    spring_k_total: float = 450.0
    spring_damping_total: float = 1.0
    leg_free_length: float = 1.0
    leg_min_length: float = 0.18
    body_width_x: float = 0.18
    body_depth_y: float = 0.24
    body_height_z: float = 0.22
    half_leg_spacing: float = 0.085
    half_thruster_spacing: float = 0.16
    foot_radius: float = 0.012
    friction_mu: float = 0.85
    dt: float = 0.002
    body_inertia_roll: float = 0.011
    body_inertia_yaw: float = 0.011
    wheel_mass: float = 0.03
    wheel_radius: float = 0.05
    wheel_transverse_inertia: float = 0.0016


def xml_string(p: Plant = Plant()) -> str:
    if not 0 < p.leg_min_length < p.leg_free_length:
        raise ValueError("Invalid leg range")
    # MuJoCo requires inertial triangles. Chosen Ix, Iz approximate transverse
    # inertias, while Iy exactly matches the earlier 2D body pitch inertia.
    ix = p.body_inertia_roll
    iz = p.body_inertia_yaw
    box = f'{p.body_width_x/2} {p.body_depth_y/2} {p.body_height_z/2}'
    def leg(side: str, y: float) -> str:
        return f'''<site name="leg_mount_{side}" pos="0 {y} 0" size="0.003" rgba="0 0 0 0"/>
        <body name="leg_{side}" pos="0 {y} 0">
          <joint name="slide_{side}" type="slide" axis="0 0 -1"
                 limited="true" range="{p.leg_min_length} {p.leg_free_length}"
                 springref="{p.leg_free_length}" stiffness="{p.spring_k_total/2}"
                 damping="{p.spring_damping_total/2}"/>
          <inertial pos="0 0 0" mass="{p.foot_mass_total/2}"
                    diaginertia="0.000005 0.000005 0.000005"/>
          <geom name="toe_{side}" type="sphere" size="{p.foot_radius}"
                pos="0 0 0" rgba="0.08 0.1 0.14 1" group="0"
                friction="{p.friction_mu} 0.005 0.0005"/>
          <site name="leg_end_{side}" size="0.003" rgba="0 0 0 0"/>
          <geom name="slider_sleeve_{side}" type="capsule" fromto="0 0 0 0 0 {p.leg_min_length * 0.85}"
                size="0.019" contype="0" conaffinity="0" mass="0"
                rgba="0.25 0.32 0.37 1"/>
        </body>'''
    return f'''<mujoco model="twin_pogo_thruster_3d">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{p.dt}" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" iterations="40" tolerance="1e-9"/>
  <!-- Set the viewing scale from the robot. The 2 km plane otherwise puts
       the default near clipping plane about 4 m in front of the camera. -->
  <statistic extent="3" center="0 0 1"/>
  <visual><global offwidth="1280" offheight="720"/>
    <headlight ambient="0.45 0.45 0.45" diffuse="0.65 0.65 0.65"/>
  </visual>
  <default><geom condim="3" solref="0.006 1" solimp="0.88 0.95 0.002"/></default>
  <asset>
    <texture name="floor_tex" type="2d" builtin="checker" rgb1="0.58 0.62 0.65" rgb2="0.48 0.53 0.58" width="512" height="512"/>
    <texture name="sky" type="skybox" builtin="gradient" width="128" height="768"
             rgb1="0.34 0.49 0.65" rgb2="0.78 0.85 0.9"/>
    <material name="floor_mat" texture="floor_tex" texuniform="true" texrepeat="1 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 8" dir="0 0 -1"/>
    <geom name="ground" type="plane" pos="0 0 0" size="2000 2000 0.1"
          material="floor_mat" friction="{p.friction_mu} 0.005 0.0005"/>
    <body name="trunk" pos="0 0 0.512">
      <freejoint name="root"/>
      <inertial pos="0 0 0" mass="{p.body_mass}"
                diaginertia="{ix} {p.body_inertia_pitch} {iz}"/>
      <geom name="body_collision" type="box" size="{box}" mass="0"
            rgba="0.12 0.47 0.85 1" friction="{p.friction_mu} 0.005 0.0005"/>
      <site name="imu" pos="0 0 0" size="0.009" rgba="0.8 0.8 0.8 1"/>
      <site name="nozzle_left" pos="0 {p.half_thruster_spacing} 0"
            size="0.02" rgba="1 0.38 0.04 1"/>
      <site name="nozzle_right" pos="0 {-p.half_thruster_spacing} 0"
            size="0.02" rgba="1 0.38 0.04 1"/>
      <geom name="thruster_left_visual" type="sphere"
            pos="0 {p.half_thruster_spacing} 0"
            size="0.04" mass="0" contype="0" conaffinity="0" rgba="0.98 0.45 0.04 1"/>
      <geom name="thruster_right_visual" type="sphere"
            pos="0 {-p.half_thruster_spacing} 0"
            size="0.04" mass="0" contype="0" conaffinity="0" rgba="0.98 0.45 0.04 1"/>
      {leg('left', p.half_leg_spacing)}
      {leg('right', -p.half_leg_spacing)}
      <body name="wheel" pos="-0.025 0 0">
        <joint name="wheel_joint" type="hinge" axis="0 1 0" damping="0.003"/>
        <inertial pos="0 0 0" mass="{p.wheel_mass}" diaginertia="{p.wheel_transverse_inertia} {p.wheel_inertia} {p.wheel_transverse_inertia}"/>
        <geom name="wheel_visual" type="cylinder" fromto="0 -0.13 0 0 0.13 0"
              size="{p.wheel_radius}" mass="0" contype="0" conaffinity="0" rgba="1 0.59 0.12 1"/>
      </body>
    </body>
  </worldbody>
  <!-- Zero-force spatial tendons draw sliders between the actual joint
       endpoints at every extension. Passive springs live on the slide joints. -->
  <tendon>
    <spatial name="shaft_left" width="0.013" rgba="0.65 0.73 0.79 1">
      <site site="leg_mount_left"/><site site="leg_end_left"/>
    </spatial>
    <spatial name="shaft_right" width="0.013" rgba="0.65 0.73 0.79 1">
      <site site="leg_mount_right"/><site site="leg_end_right"/>
    </spatial>
  </tendon>
  <actuator>
    <motor name="flywheel_motor" joint="wheel_joint" gear="1"
           ctrllimited="false"/>
  </actuator>
  <sensor><gyro name="gyro" site="imu"/><framequat name="body_quat" objtype="body" objname="trunk"/></sensor>
  <keyframe>
    <key name="compressed" qpos="0 0 0.512 1 0 0 0 0.5 0.5 0"/>
  </keyframe>
</mujoco>'''


def write_xml(path: str | Path, p: Plant = Plant()) -> Path:
    path = Path(path)
    path.write_text(xml_string(p), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(write_xml(Path(__file__).with_name('robot.xml')))
