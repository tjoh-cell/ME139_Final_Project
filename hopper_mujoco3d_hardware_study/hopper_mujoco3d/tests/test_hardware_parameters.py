"""Regression checks for hardware settings shared by the setup panel and viewer."""
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hardware_parameters import field_values, make_settings
from simulator import Actuators, Guidance, HopperSimulation, Mission, Plant, Wind, mujoco


class HardwareSettingsTests(unittest.TestCase):
    def setUp(self):
        self.base = (Plant(), Actuators(), Guidance(), Wind(), Mission())
        self.values = field_values(self.base, 1.)

    def test_defaults_round_trip(self):
        self.assertEqual(make_settings(self.base, self.values, 'feedback', True), (self.base, 1.))

    def test_invalid_geometry_and_nonfinite_inputs(self):
        for changes in (dict(initial_compression=2), dict(kt='nan'), dict(max_hops=3.5),
                        dict(gear_eff=1.5), dict(wheel_inertia=.01), dict(elevation_max=181)):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                make_settings(self.base, self.values | changes, 'feedback', True)

    @unittest.skipUnless(mujoco is not None, 'native MuJoCo required')
    def test_candidate_values_reach_native_model(self):
        changes = dict(leg_free_length=.8, initial_compression=.3, spring_k_total=320.,
                       wheel_mass=.08, wheel_inertia=.0001, wheel_transverse_inertia=.00006,
                       elevation_max=120., goal_x=30.)
        settings, _ = make_settings(self.base, self.values | changes, 'predictive', False)
        sim = HopperSimulation(*settings)
        self.assertAlmostEqual(sim.m.body('wheel').mass.item(), .08)
        self.assertAlmostEqual(sim.m.body('wheel').inertia[1], .0001)
        self.assertAlmostEqual(sim.m.joint('slide_left').stiffness.item(), 160.)
        self.assertAlmostEqual(sim.d.joint('slide_right').qpos.item(), .5)
        self.assertAlmostEqual(sim.a.elevation_max, math.radians(120))
        self.assertEqual(sim.mission.goal_x, 30.)
        self.assertTrue(sim.mission.no_thrusters)
        self.assertEqual(sim.mission.planner, 'predictive')
