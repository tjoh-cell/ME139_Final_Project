"""Tests that run with or without the native MuJoCo library."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import math
import numpy as np
import xml.etree.ElementTree as ET
from model import Plant, xml_string
from control import (Actuators, Guidance, ThrusterMemory, allocate_forces,
                     flight_gate, motor_torque, step_thrusters,
                     rotor_to_world, target_vx, vector_to_angles)
from simulator import Wind
from control import support_differential_thrust, desired_flight_wrench

class StaticTests(unittest.TestCase):
    def test_roll_allocation_near_reverse_thrust(self):
        hw = Actuators()
        requests = allocate_forces(np.array([-3., 0., 0.]), .32, .12, .16)
        difference = requests[0] - requests[1]
        support_differential_thrust(requests, hw)
        self.assertTrue(np.allclose(requests[0] - requests[1], difference))
        for request in requests:
            _, angle, _ = vector_to_angles(request)
            self.assertGreaterEqual(angle, hw.elevation_min)
            self.assertLessEqual(angle, hw.elevation_max + 1e-12)

    def test_wheel_momentum_couples_roll_error_to_yaw_torque(self):
        roll = .05
        c, s = math.cos(roll), math.sin(roll)
        R = np.array([[1,0,0],[0,c,-s],[0,s,c]])
        args = (np.array([0.,0.,2.]), np.array([0.,0.,4.]),
                np.array([roll,0.,0.]), np.zeros(3), 60., Guidance(), Actuators(), 1., R)
        positive = desired_flight_wrench(*args, wheel_momentum=.4)
        negative = desired_flight_wrench(*args, wheel_momentum=-.4)
        self.assertLess(positive[2], 0.)
        self.assertAlmostEqual(positive[2], -negative[2])

    def test_backward_demand_below_horizontal_uses_rear_stop(self):
        hw = Actuators()
        memory = ThrusterMemory()
        step_thrusters(memory, np.array([[-1.,0.,-.01]]*2), np.eye(3), .002, True, hw)
        self.assertTrue(np.all(memory.theta > math.pi/2))

    def test_mjcf_representative_legs(self):
        root=ET.fromstring(xml_string(Plant()))
        self.assertEqual(root.tag,'mujoco')
        self.assertEqual(len(root.findall('.//joint[@type="slide"]')),2)
        self.assertEqual(len(root.findall('.//freejoint')),1)
        self.assertEqual(len(root.findall('.//joint[@name="wheel_joint"]')),1)
        self.assertEqual(len(root.findall('.//site[@name="nozzle_left"]')),1)
        self.assertEqual(len(root.findall('.//site[@name="nozzle_right"]')),1)
        self.assertEqual(len(root.findall('.//motor')),1)
        self.assertEqual(sum(float(j.attrib['stiffness']) for j in root.findall('.//joint[@type="slide"]')),450)
    def test_roll_and_yaw_allocation(self):
        b=.16
        f=allocate_forces(np.array([4.,2.,10.]),tau_x=.48,tau_z=-.32,half_span=b)
        self.assertTrue(np.allclose(f.sum(axis=0),[4,2,10]))
        rL=np.array([0.,b,0.]);rR=np.array([0.,-b,0.])
        moment=np.cross(rL,f[0])+np.cross(rR,f[1])
        self.assertAlmostEqual(moment[0],.48,places=10)
        self.assertAlmostEqual(moment[2],-.32,places=10)
        self.assertAlmostEqual(moment[1],0.,places=10)
    def test_independent_thrusters(self):
        a=Actuators(max_thrust_each=8,impulse_budget=.25,thrust_slew=1000,gimbal_slew=100)
        m=ThrusterMemory()
        demand=np.array([[2.,1.,4.],[2.,-1.,5.]])
        f=step_thrusters(m,demand,np.eye(3),dt=.02,allowed=True,hw=a)
        self.assertNotAlmostEqual(f[0,1],f[1,1])
        self.assertLessEqual(max(m.thrust),a.max_thrust_each)
        self.assertGreater(m.impulse_used,0)
        before=m.impulse_used
        off=step_thrusters(m,demand,np.eye(3),dt=.02,allowed=False,hw=a)
        self.assertTrue(np.allclose(off,0))
        self.assertEqual(m.impulse_used,before)
    def test_budget_never_exceeded(self):
        a=Actuators(max_thrust_each=8,impulse_budget=.1,thrust_slew=1000,gimbal_slew=100)
        m=ThrusterMemory()
        for k in range(10):
            step_thrusters(m,np.array([[8.,0,0],[8.,0,0]]),np.eye(3),.02,True,a)
        self.assertLessEqual(m.impulse_used,a.impulse_budget+1e-10)
    def test_ascent_only_cutoff_and_apex(self):
        a=Actuators()
        self.assertFalse(flight_gate(.1,True,1.,5.,a))
        self.assertFalse(flight_gate(.1,False,7.,1.,a))
        self.assertFalse(flight_gate(.1,False,2.,-1.,a))
        self.assertFalse(flight_gate(.1,False,7.6,2.4,a))
        self.assertTrue(flight_gate(.5,False,2.,3.,a))
    def test_gimbal_angles_realize_requested_force(self):
        f=np.array([-2.,1.,6.]);T,th,ph=vector_to_angles(f)
        recovered=rotor_to_world(np.eye(3),T,th,ph)
        self.assertTrue(np.allclose(f,recovered))
    def test_dc_motor_limits(self):
        a=Actuators()
        for omega in [-400,-100,0,100,400]:
            tau,v,i=motor_torque(300.,omega,a)
            self.assertLessEqual(abs(v),a.supply_voltage+1e-10)
            self.assertLessEqual(abs(i),a.current_limit+1e-10)
            self.assertLessEqual(abs(tau),a.gear_ratio*a.gear_eff*a.kt*a.current_limit+1e-10)
    def test_braking_direction(self):
        g=Guidance()
        self.assertGreater(target_vx(20.,g),0)
        self.assertLess(target_vx(-20.,g),0)
        self.assertLess(abs(target_vx(.2,g)),abs(target_vx(20.,g)))
    def test_wind_aerodynamic(self):
        w=Wind(y=6.)
        a=w.drag_force(0,np.zeros(3))
        self.assertGreater(a[1],0)
        self.assertAlmostEqual(a[0],0,places=10)
        self.assertTrue(np.allclose(w.drag_force(0,w.velocity(0)),0))
        self.assertNotAlmostEqual(Wind(gust_y=2.).velocity(0)[1],
                                  Wind(gust_y=2.).velocity(.25)[1])

if __name__=='__main__':
    unittest.main()
