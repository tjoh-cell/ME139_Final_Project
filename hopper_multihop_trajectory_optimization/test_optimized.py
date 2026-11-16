"""Regression tests for hybrid multi-hop stance+flight planning and landing checks."""
import unittest
import numpy as np
from hybrid_hopper import Robot, Control, initial_state
from trajectory_planner import (
    rollout, optimize_horizons, PlanningOptions, run_optimized_hops, _one_hop,
)


class TestOptimizer(unittest.TestCase):
    def test_independent_horizons_have_both_reference_sequences(self):
        p, c = Robot(), Control()
        opts = PlanningOptions(lookahead=2, maxfev_per_horizon=20, verbose=False)
        selected, choices = optimize_horizons(initial_state(p), 0., .30, p, c, opts)
        self.assertEqual([h['n'] for h in choices], [1, 2])
        for h in choices:
            self.assertEqual(len(h['refs']), h['n'])
            self.assertEqual(len(h['flight_refs']), h['n'])
            self.assertTrue(np.all(np.abs(h['refs']) <= c.max_tilt_reference + 1e-10))
            self.assertTrue(np.all(np.abs(h['flight_refs']) <= np.deg2rad(opts.max_flight_reference_deg)+1e-10))
        self.assertIsNotNone(selected)

    def test_rollout_allows_distinct_stance_and_flight_angles_each_hop(self):
        p, c = Robot(), Control()
        r = rollout([.12, -.10], initial_state(p), 0., .30, p, c,
                    flight_refs=[-.04, .05])
        self.assertTrue(r['ok'], r['event'])
        self.assertEqual(r['completed'], 2)
        self.assertEqual(r['steps'][1]['alpha_ref'], -.10)
        self.assertEqual(r['steps'][0]['flight_ref'], -.04)
        self.assertEqual(r['steps'][1]['flight_ref'], .05)
        self.assertNotAlmostEqual(r['steps'][0]['touchdown_pitch_rad'],
                                  r['steps'][1]['touchdown_pitch_rad'], places=3)

    def test_flight_reference_actually_changes_touchdown_angle(self):
        p, c = Robot(), Control()
        y0 = initial_state(p)
        a = _one_hop(y0, 0., .10, p, c, flight_ref=-.2)
        b = _one_hop(y0, 0., .10, p, c, flight_ref=.2)
        self.assertTrue(a['ok'])
        self.assertTrue(b['ok'])
        self.assertLess(a['landing']['touchdown_pitch_rad'],
                        b['landing']['touchdown_pitch_rad']-.25)

    def test_unsafe_attitude_rejected_even_if_landing_location_fits(self):
        p, c = Robot(), Control()
        y0 = initial_state(p)
        first = _one_hop(y0, 0., .1, p, c, flight_ref=.6)
        self.assertFalse(first['ok'])
        self.assertEqual(first['event'], 'unsafe_touchdown_pitch')
        # Even if the requested target is the unsafe touchdown coordinate,
        # it cannot be returned as a valid goal-reaching rollout.
        target = first['foot_x']
        r = rollout([.1], y0, 0., target, p, c, flight_refs=[.6])
        self.assertFalse(r['ok'])
        self.assertEqual(r['event'], 'unsafe_touchdown_pitch')

    def test_unsafe_pitch_rate_is_rejected(self):
        p, c = Robot(), Control()
        opts = PlanningOptions(max_touchdown_pitch_rate=.01)
        result = _one_hop(initial_state(p), 0., .1, p, c,
                          flight_ref=.2, options=opts)
        self.assertFalse(result['ok'])
        self.assertEqual(result['event'], 'unsafe_touchdown_pitch_rate')

    def test_short_flight_rejected(self):
        p, c = Robot(), Control()
        opts = PlanningOptions(min_flight_time=1.0)
        result = _one_hop(initial_state(p), 0., .1, p, c,
                          flight_ref=0., options=opts)
        self.assertFalse(result['ok'])
        self.assertEqual(result['event'], 'degenerate_flight_time')

    def test_repeated_no_progress_is_rejected(self):
        p, c = Robot(), Control()
        opts = PlanningOptions(max_stalled_hops=2, min_progress=.01)
        r = rollout([0., 0., 0.], initial_state(p), 0., .45, p, c, options=opts)
        self.assertFalse(r['ok'])
        self.assertEqual(r['event'], 'stalled_no_progress')
        self.assertEqual(r['completed'], 2)

    def test_backward_correction_remains_possible(self):
        p, c = Robot(), Control()
        first = rollout([.27], initial_state(p), 0., .08, p, c)
        self.assertTrue(first['ok'])
        self.assertGreater(first['foot_x'], .08)
        opts = PlanningOptions(lookahead=2, maxfev_per_horizon=40, verbose=False)
        selected, _ = optimize_horizons(first['y'], first['foot_x'], .08, p, c, opts)
        self.assertIsNotNone(selected)
        self.assertLess(selected['refs'][0], 0.)
        self.assertLess(abs(selected['rollout']['foot_x']-.08), .02)

    def test_can_stop_near_goal_without_zero_energy(self):
        p, c = Robot(), Control()
        opts = PlanningOptions(lookahead=2, maxfev_per_horizon=25,
                               safety_max_hops=5, arrival_tolerance=.10,
                               verbose=False)
        r = run_optimized_hops(p, c, .15, .12, opts)
        self.assertTrue(r['hops'])
        self.assertLessEqual(len(r['hops']), 5)
        self.assertGreater(r['total_time'], 0.)
        self.assertTrue(all(abs(h['touchdown_pitch_deg']) <= opts.max_touchdown_angle_deg+1e-6
                            and abs(h['touchdown_pitch_rate']) <= opts.max_touchdown_pitch_rate+1e-6
                            for h in r['hops']))


if __name__ == '__main__':
    unittest.main()
