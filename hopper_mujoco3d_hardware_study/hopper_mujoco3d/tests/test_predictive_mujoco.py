"""Native-engine clone/physical rollout smoke test; skipped without MuJoCo."""
import unittest
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator import HopperSimulation, Mission, mujoco
from simulator import Wind
from control import Guidance
from predictive import HopCommand

@unittest.skipUnless(mujoco is not None,'MuJoCo package unavailable')
class PredictiveIntegrationTests(unittest.TestCase):
    def test_crosswind_during_high_apex_approach(self):
        sim = HopperSimulation(guidance=Guidance(height_target=7.), wind=Wind(y=.3),
                               mission=Mission(goal_x=60., max_time=40.))
        result = sim.run()
        self.assertEqual(result['status'], 'goal_tolerance_reached')
        self.assertLess(result['max_lateral'], .25)
        self.assertLess(result['impulse_used'], sim.a.impulse_budget)
        self.assertTrue(all(not row['pre_step_toe_contact'] and row['pre_step_vz'] > 0
                            for row in sim.trace if row['T1'] or row['T2']))

    def test_powered_multihop_arrival(self):
        sim = HopperSimulation(mission=Mission(goal_x=15, max_time=15))
        summary = sim.run()
        self.assertEqual(summary['status'], 'goal_tolerance_reached')
        self.assertGreater(sim.hops, 1)
        self.assertLessEqual(abs(summary['position_error']), sim.g.position_tolerance)
        self.assertTrue(all(l['left_contact'] or l['right_contact'] for l in sim.landings))
        powered = [row for row in sim.trace if row['T1'] or row['T2']]
        self.assertTrue(powered)
        self.assertTrue(all(not row['pre_step_toe_contact'] and row['pre_step_vz'] > 0
                            for row in powered))
        liftoffs = [row for row in sim.trace if row['pre_step_toe_contact']
                    and not row['toe_contact'] and row['vz'] > 1.]
        self.assertEqual(len(liftoffs), sim.hops)
        for touchdown, liftoff in zip(sim.landings, liftoffs[1:]):
            self.assertLess(touchdown['pitch'], 0.)
            self.assertGreater(liftoff['pitch'], 0.)
            self.assertGreater(liftoff['vx'], 0.)
            self.assertEqual(liftoff['T1'] + liftoff['T2'], 0.)

    def test_branch_is_independent(self):
        s=HopperSimulation(mission=Mission(goal_x=2,max_time=.05))
        branch=s.fork_for_prediction(HopCommand(0.,0.,0.))
        self.assertIsNot(branch.d,s.d)
        self.assertIsNot(branch.memory,s.memory)
        original=s.d.qpos.copy()
        branch.forward_step()
        self.assertTrue(np.allclose(s.d.qpos,original))
        self.assertGreater(branch.d.time,s.d.time)
        self.assertEqual(len(branch.trace),0)

    def test_nozzle_replay_uses_independent_applied_states(self):
        from thruster_display import actuator_state, draw_nozzles
        sim = HopperSimulation()
        scene = mujoco.MjvScene(sim.m, maxgeom=8)
        sim.memory.theta[:] = [0., np.pi / 2]
        sim.memory.phi[:] = [0., -np.pi / 4]
        sim.memory.thrust[:] = [0., 4.]
        snapshot = actuator_state(sim.memory)
        sim.memory.thrust[:] = 0.
        draw_nozzles(scene, sim.d, sim.bid, sim.sites, snapshot, 8.)
        self.assertEqual(scene.ngeom, 3)  # Two nozzles, only the right one firing.
        np.testing.assert_allclose(scene.geoms[0].mat.reshape(3, 3)[:, 2], [-1., 0., 0.], atol=1e-6)
        np.testing.assert_allclose(scene.geoms[2].mat.reshape(3, 3)[:, 2],
                                   [0., 2**-0.5, -2**-0.5], atol=1e-6)

if __name__=='__main__':unittest.main()
