"""Only execute these integration tests if native MuJoCo is installed."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Plant, xml_string
from control import Actuators
from simulator import HopperSimulation, Mission, mujoco

@unittest.skipUnless(mujoco is not None, 'MuJoCo package unavailable in this environment')
class IntegrationTests(unittest.TestCase):
    def test_model_compiles(self):
        m=mujoco.MjModel.from_xml_string(xml_string(Plant()))
        self.assertEqual(m.nq,10)
        self.assertEqual(m.nv,9)
    def test_initial_foot_contacts_and_dynamics(self):
        s=HopperSimulation(mission=Mission(goal_x=2,max_time=.1))
        self.assertAlmostEqual(s.d.joint('slide_left').qpos[0],.5)
        self.assertAlmostEqual(s.d.joint('slide_right').qpos[0],.5)
        for _ in range(50):
            s.forward_step()
            if s.status!='running':break
        self.assertGreater(s.d.time,0)
        self.assertTrue(s.status in ('running','body_collision','height_ceiling_exceeded',
                                    'unsafe_touchdown_attitude','goal_tolerance_reached'))

if __name__=='__main__':
    unittest.main()
