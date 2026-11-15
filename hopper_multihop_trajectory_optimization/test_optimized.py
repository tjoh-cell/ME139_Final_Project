"""Deterministic smoke tests for multi-hop optimization."""
import unittest
import numpy as np
from hybrid_hopper import Robot,Control,initial_state
from trajectory_planner import (
    rollout,optimize_horizons,PlanningOptions,run_optimized_hops)

class TestOptimizer(unittest.TestCase):
    def test_candidate_n_are_independent_and_bounded(self):
        p=Robot(); c=Control()
        opts=PlanningOptions(lookahead=2,maxfev_per_horizon=20,verbose=False)
        chosen,choices=optimize_horizons(initial_state(p),0.,.30,p,c,opts)
        self.assertEqual([k['n'] for k in choices],[1,2])
        self.assertTrue(all(len(k['refs'])==k['n'] for k in choices))
        self.assertTrue(all(np.all(np.abs(k['refs'])<=c.max_tilt_reference+1e-10) for k in choices))
        self.assertIsNotNone(chosen)

    def test_rollout_can_choose_different_angles_each_hop(self):
        p=Robot(); c=Control()
        r=rollout([.12,-.10],initial_state(p),0.,.3,p,c)
        self.assertTrue(r['ok'])
        self.assertEqual(r['completed'],2)
        self.assertEqual(r['steps'][1]['alpha_ref'],-.10)

    def test_overshoot_can_be_corrected_with_backward_hop(self):
        p=Robot();c=Control()
        first=rollout([.27],initial_state(p),0.,.08,p,c)
        self.assertTrue(first['ok'])
        self.assertGreater(first['foot_x'],.08)
        opts=PlanningOptions(lookahead=2,maxfev_per_horizon=40,verbose=False)
        selected,_=optimize_horizons(first['y'],first['foot_x'],.08,p,c,opts)
        self.assertIsNotNone(selected)
        self.assertLess(selected['refs'][0],0.)
        self.assertLess(abs(selected['rollout']['foot_x']-.08),.005)
        forced=rollout([.27,-.27],initial_state(p),0.,.08,p,c)
        self.assertEqual(forced['crossings'],2)

    def test_stops_when_close_without_waiting_for_zero_energy(self):
        p=Robot();c=Control()
        o=PlanningOptions(lookahead=2,maxfev_per_horizon=25,
                          safety_max_hops=5,arrival_tolerance=.10,verbose=False)
        r=run_optimized_hops(p,c,.15,.12,o)
        self.assertTrue(r['hops'])
        self.assertLessEqual(len(r['hops']),5)
        self.assertGreater(r['total_time'],0.)

if __name__=='__main__':unittest.main()
