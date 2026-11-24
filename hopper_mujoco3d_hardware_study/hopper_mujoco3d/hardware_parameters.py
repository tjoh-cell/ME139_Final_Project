"""Hardware study settings. UI angles are degrees; simulation angles are radians."""
from dataclasses import fields, replace
import math

# Key, display name, units, interpretation. Defaults come from the running model.
SECTIONS = {
    'Mission': [
        ('goal_x', 'Destination x', 'm', 'Touchdown target along the ground.'),
        ('height_target', 'Requested apex', 'm', 'Target trunk height during flight.'),
        ('speed', 'Playback speed', '×', 'Display speed relative to real time.'),
        ('max_hops', 'Hop limit', 'hops', 'End the mission at this many landings.'),
        ('max_time', 'Time limit', 's', 'Maximum simulated mission duration.'),
        ('planner_budget', 'Predictive search budget', 'actions', 'Maximum first-hop candidates per search.'),
        ('y', 'Steady crosswind', 'm/s', 'Signed wind velocity along world y.'),
        ('gust_y', 'Crosswind gust amplitude', 'm/s', 'Amplitude of the sinusoidal gust.'),
    ],
    'Springs & feet': [
        ('leg_free_length', 'Free leg length', 'm', 'Distance from trunk pivot to toe center at rest.'),
        ('leg_min_length', 'Minimum leg length', 'm', 'Length at full compression.'),
        ('initial_compression', 'Launch preload', 'm', 'Initial shortening of each spring, applied once.'),
        ('spring_k_total', 'Combined spring stiffness', 'N/m', 'For two identical springs, enter twice the catalog rate in N/m.'),
        ('spring_damping_total', 'Combined damping', 'N·s/m', 'Sum of both legs’ axial damping coefficients.'),
        ('foot_mass_total', 'Combined moving leg mass', 'kg', 'Total sliding mass lumped at the two feet.'),
        ('foot_radius', 'Toe radius', 'm', 'Radius of each spherical contact surface.'),
        ('half_leg_spacing', 'Leg mount offset', 'm', 'Lateral distance from centerline to one leg.'),
        ('friction_mu', 'Sliding friction', '', 'Toe and ground friction coefficient.'),
    ],
    'Body & wheel': [
        ('body_mass', 'Trunk mass', 'kg', 'Include fixed hardware and payload. Exclude rotor and sliding legs.'),
        ('body_inertia_roll', 'Trunk roll inertia', 'kg·m²', 'About body x through the trunk center of mass.'),
        ('body_inertia_pitch', 'Trunk pitch inertia', 'kg·m²', 'About body y through the trunk center of mass.'),
        ('body_inertia_yaw', 'Trunk yaw inertia', 'kg·m²', 'About body z through the trunk center of mass.'),
        ('body_width_x', 'Trunk length along x', 'm', 'Front-to-back collision-box dimension.'),
        ('body_depth_y', 'Trunk width along y', 'm', 'Side-to-side collision-box dimension.'),
        ('body_height_z', 'Trunk height', 'm', 'Vertical collision-box dimension.'),
        ('wheel_mass', 'Rotor mass', 'kg', 'Rotating assembly only.'),
        ('wheel_radius', 'Rotor outer radius', 'm', 'Sets the drawing and inertia check; enter inertia separately.'),
        ('wheel_inertia', 'Rotor spin inertia', 'kg·m²', 'About the wheel shaft (body y). Use measured or CAD inertia.'),
        ('wheel_transverse_inertia', 'Rotor transverse inertia', 'kg·m²', 'About either axis across the shaft, assuming axial symmetry.'),
    ],
    'Motor & gearbox': [
        ('supply_voltage', 'Available voltage', 'V', 'Available motor-drive bus voltage under load.'),
        ('current_limit', 'Current limit', 'A', 'Controller current cap used by the motor model.'),
        ('resistance', 'Motor resistance', 'Ω', 'Effective winding resistance in the DC-equivalent model.'),
        ('kt', 'Torque constant', 'N·m/A', 'Motor shaft torque per ampere before gearing.'),
        ('ke', 'Back-EMF constant', 'V·s/rad', 'Motor-side generated voltage per rad/s.'),
        ('gear_ratio', 'Gear reduction', 'motor / wheel', '5 means five motor revolutions per wheel revolution.'),
        ('gear_eff', 'Gear efficiency', 'fraction', '0.9 means 90 percent.'),
        ('wheel_speed_warning', 'Wheel speed warning', 'rad/s', 'Flags peak speed in results; the motor can exceed it.'),
    ],
    'Thrusters': [
        ('max_thrust_each', 'Peak thrust per nozzle', 'N', 'Maximum magnitude from one nozzle.'),
        ('impulse_budget', 'Mission impulse budget', 'N·s', 'Shared by both nozzles across all hops.'),
        ('thrust_slew', 'Thrust slew per nozzle', 'N/s', 'Maximum magnitude change while the flight gate is open.'),
        ('gimbal_slew', 'Gimbal slew', 'rad/s', 'Maximum angular rate on each gimbal axis.'),
        ('elevation_min', 'Minimum elevation', 'deg', 'Body-frame angle from forward toward up.'),
        ('elevation_max', 'Maximum elevation', 'deg', 'Maximum angle in the forward/up plane.'),
        ('lateral_gimbal_limit', 'Lateral deflection', 'deg', 'Symmetric side-to-side angular limit.'),
        ('half_thruster_spacing', 'Nozzle mount offset', 'm', 'Distance from centerline to one nozzle.'),
        ('cutoff_z', 'Thrust cutoff height', 'm', 'Disable thrust above this trunk height.'),
        ('ceiling_z', 'Mission height ceiling', 'm', 'Crossing this trunk height ends the run.'),
    ],
    'Guidance': [
        ('cruise_speed', 'Cruise speed', 'm/s', 'Desired horizontal travel speed.'),
        ('braking_accel', 'Braking acceleration', 'm/s²', 'Guidance assumption for slowing near the goal.'),
        ('foot_placement_gain', 'Foot-placement gain', '', 'Scales landing lean. The stance-swing default is 0.55.'),
        ('kp_forward', 'Forward thrust gain', '1/s', 'Scales thrust used to correct horizontal ground speed.'),
        ('vertical_assist', 'Vertical thrust request', 'N', 'Desired total vertical assist below the requested apex.'),
        ('position_tolerance', 'Goal tolerance along x', 'm', 'Maximum foot-position error accepted at touchdown.'),
    ],
}
PARAMETERS = {key: (label, unit, hint) for section in SECTIONS.values() for key, label, unit, hint in section}
ANGLES = { 'elevation_min', 'elevation_max', 'lateral_gimbal_limit'}
ZERO_OK = {'initial_compression', 'spring_damping_total', 'friction_mu', 'max_thrust_each',
           'impulse_budget', 'elevation_min', 'lateral_gimbal_limit', 'kp_forward',
           'foot_placement_gain', 'vertical_assist', 'position_tolerance', 'height_target'}
SIGNED = {'goal_x', 'y', 'gust_y'}
COUNTS = {'max_hops', 'planner_budget'}


def field_values(settings, speed):
    values = {'speed': speed}
    for obj in settings:
        for field in fields(obj):
            if field.name in PARAMETERS:
                value = getattr(obj, field.name)
                values[field.name] = math.degrees(value) if field.name in ANGLES else value
    return {key: str(value) for key, value in values.items()}


def make_settings(base, entries, planner, thrust_enabled):
    numbers = {}
    for key, (label, unit, _) in PARAMETERS.items():
        try:
            value = float(entries[key])
        except (TypeError, ValueError, KeyError):
            raise ValueError(f'Enter a number for {label} ({unit}).') from None
        if not math.isfinite(value):
            raise ValueError(f'{label} must be finite.')
        if key in COUNTS and not value.is_integer():
            raise ValueError(f'{label} must be a whole number.')
        if key not in SIGNED and (value < 0 or (value == 0 and key not in ZERO_OK)):
            raise ValueError(f'Check {label}. Enter a {"nonnegative" if key in ZERO_OK else "positive"} value.')
        numbers[key] = int(value) if key in COUNTS else math.radians(value) if key in ANGLES else value
    settings = [replace(obj, **{f.name: numbers[f.name] for f in fields(obj) if f.name in numbers})
                for obj in base]
    plant, hw, guide, wind, mission = settings
    if planner not in ('feedback', 'predictive'):
        raise ValueError('Choose feedback or predictive control.')
    if not 0 <= mission.initial_compression < plant.leg_free_length - plant.leg_min_length:
        raise ValueError('Free length must exceed minimum length plus launch preload.')
    if plant.leg_free_length - mission.initial_compression + plant.foot_radius <= plant.body_height_z / 2:
        raise ValueError('The trunk would start against the ground. Reduce preload or increase leg length.')
    inertia = sorted((plant.body_inertia_roll, plant.body_inertia_pitch, plant.body_inertia_yaw))
    if inertia[2] > inertia[0] + inertia[1] or plant.wheel_inertia > 2 * plant.wheel_transverse_inertia:
        raise ValueError('Each principal inertia must be at most the sum of the other two. Check Body & wheel.')
    if hw.gear_eff > 1:
        raise ValueError('Gear efficiency must be at most 1.')
    if not hw.elevation_min <= hw.elevation_max <= math.pi or hw.lateral_gimbal_limit > math.pi/2:
        raise ValueError('Use ordered elevation limits within 0–180 degrees and lateral deflection within 0–90 degrees.')
    if max(hw.cutoff_z, guide.height_target) >= hw.ceiling_z:
        raise ValueError('Requested apex and thrust cutoff must each be below the mission ceiling.')
    return (plant, hw, guide, wind, replace(mission, planner=planner, no_thrusters=not thrust_enabled)), numbers['speed']
