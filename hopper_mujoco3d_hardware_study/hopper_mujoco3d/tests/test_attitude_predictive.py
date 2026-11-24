"""No-MuJoCo unit checks; these guard against the Euler 180° roll artifact."""
import unittest
from pathlib import Path
import sys
import math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scipy.spatial.transform import Rotation
from control import (attitude_state,attitude_error_to_target,get_rpy,
                     touchdown_pitch_reference, Guidance, motor_command, Actuators)
from predictive import candidates,HopCommand

class AttitudeAndSearchTests(unittest.TestCase):
    def test_pure_planar_pitch_beyond_90_is_not_roll(self):
        for deg in (-170,-120,-90,-30,0,30,90,120,170):
            R=Rotation.from_euler('y',math.radians(deg)).as_matrix()
            pitch, correction, outplane=attitude_state(R)
            self.assertAlmostEqual(math.degrees(pitch),deg,places=8)
            self.assertLess(outplane,1e-9)
            rpy=get_rpy(R)
            self.assertAlmostEqual(rpy[0],0,places=8)
            self.assertAlmostEqual(rpy[2],0,places=8)

    def test_small_out_of_plane_tilt_measured(self):
        R=Rotation.from_euler('y',math.radians(120)).as_matrix() @ Rotation.from_euler('x',math.radians(8)).as_matrix()
        pitch, correction, outplane=attitude_state(R)
        self.assertGreater(outplane,math.radians(5))
        self.assertLess(outplane,math.radians(12))
        self.assertLess(abs(pitch),math.pi)

    def test_geodesic_target_error_at_large_planar_pitch(self):
        R=Rotation.from_euler('y',math.radians(120)).as_matrix()
        e=attitude_error_to_target(R,0)
        self.assertAlmostEqual(e[1],-math.radians(120),places=8)
        self.assertAlmostEqual(e[0],0,places=8)
        self.assertAlmostEqual(e[2],0,places=8)

    def test_flight_pitch_lead_bounded(self):
        R=Rotation.from_euler('y',math.radians(70)).as_matrix()
        ref=touchdown_pitch_reference(R,.5,1.0,Guidance())
        self.assertLess(ref,0)
        self.assertLessEqual(abs(ref),Guidance().flight_pitch_limit+1e-12)

    def test_motor_voltage_current_still_limited(self):
        a=Actuators()
        tau,V,I=motor_command(math.radians(140),2.0,100,False,0.0,Guidance(),a)
        self.assertLessEqual(abs(V),a.supply_voltage+1e-9)
        self.assertLessEqual(abs(I),a.current_limit+1e-9)

    def test_candidate_controls_within_guidance_limits(self):
        class Dummy:
            class D:
                qpos=np.array([2.,0,1.])
                qvel=np.array([3.,0,0.])
            d=D()
            class Mission:goal_x=15.
            mission=Mission()
            g=Guidance()
        cs=candidates(Dummy())
        self.assertTrue(len(cs)>5)
        self.assertTrue(any(c.stance_ref<0 for c in cs))
        self.assertTrue(all(abs(c.stance_ref)<=Dummy.g.stance_pitch_limit+1e-8 for c in cs))
        self.assertTrue(all(0<=c.thrust_scale<=1 for c in cs))

if __name__=='__main__':unittest.main()
