"""Render the applied gimbal state without adding forces or moving masses."""
import math

import mujoco
import numpy as np

from control import rotor_to_world


def actuator_state(memory):
    # Copy all six values for replay after live actuator memory has changed.
    return np.column_stack((memory.theta, memory.phi, memory.thrust))


def _segment(scene, kind, radius, start, end, color):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, kind, np.zeros(3), start, np.eye(3).ravel(), color)
    mujoco.mjv_connector(geom, kind, radius, start, end)
    scene.ngeom += 1


def draw_nozzles(scene, data, body_id, sites, state, max_force):
    rotation = data.xmat[body_id].reshape(3, 3)
    for site, (elevation, lateral, thrust) in zip(sites, state):
        mount = data.site_xpos[site]
        exhaust = -rotor_to_world(rotation, 1., elevation, lateral)
        exit_pos = mount + .12 * exhaust
        _segment(scene, mujoco.mjtGeom.mjGEOM_CYLINDER, .033, mount, exit_pos,
                 [.32, .39, .45, 1.])
        if thrust > 0 and max_force > 0:
            # Exaggerate weak jets so firing is legible in the tracking view.
            tip = exit_pos + (.12 + .85 * math.sqrt(thrust / max_force)) * exhaust
            _segment(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, .036, exit_pos, tip,
                     [1., .36, .04, 1.])


def readout(time, body_x, body_z, goal_x, contact, state):
    phase = 'toe contact' if contact else 'airborne'
    lines = [f'{time:.2f} s    Body x {body_x:.2f} m    Goal {goal_x:g} m    Height {body_z:.2f} m    {phase}']
    for side, (theta, phi, force) in zip(('Left', 'Right'), state):
        lines.append(f'{side}   {force:.2f} N    elevation {math.degrees(theta):.0f} deg'
                     f'    lateral {math.degrees(phi):+.0f} deg')
    return '\n'.join(lines)
