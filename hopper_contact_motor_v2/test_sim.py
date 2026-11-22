"""Basic regression checks. Run: python -m unittest -v test_sim.py"""
import unittest
import numpy as np
from physics import (Robot,Controller,SimulationConfig,initial_state,
                     foot_kinematics,body_corners,contact_candidates,
                     mass_matrix,solve_impulses,motor_from_request,mechanical_energy,step)
from simulator import GoalOptions,one_hop
from planner import PlannerOptions,run_controller

class TestContactAndActuator(unittest.TestCase):
    def setUp(self):
        self.p=Robot()
        self.c=Controller()
        self.cfg=SimulationConfig(dt=.004,contact_iterations=20)

    def test_motor_voltage_current_and_torque_speed(self):
        v=np.zeros(5)
        out=motor_from_request(100.,v,self.p)
        self.assertLessEqual(abs(out['voltage']),self.p.supply_voltage)
        self.assertLessEqual(abs(out['current']),self.p.current_limit)
        self.assertAlmostEqual(out['torque'],11.25,places=4)
        v[4]=500.
        fast=motor_from_request(100.,v,self.p)
        self.assertLess(abs(fast['torque']),abs(out['torque']))
        self.assertAlmostEqual(motor_from_request(0.,v,self.p)['current'],0.)

    def test_mass_matrix_positive_definite(self):
        q=initial_state(self.p,.3)[:5]
        M=mass_matrix(q,self.p)
        self.assertTrue(np.allclose(M,M.T))
        self.assertGreater(np.min(np.linalg.eigvalsh(M)),0.)

    def test_foot_contact_impulses_respect_friction(self):
        q=initial_state(self.p,.2)[:5]
        M=mass_matrix(q,self.p)
        contacts=contact_candidates(q,self.p,self.cfg)
        self.assertTrue(any(x['name']=='foot' for x in contacts))
        # Highly oblique incoming foot velocity should permit slip,
        # not demand impossible tangential impulses.
        vel=np.array([20.,-4.,0.,0.,0.])
        v, impulses=solve_impulses(vel,M,contacts,self.cfg,self.cfg.dt)
        for impulse in impulses:
            self.assertGreaterEqual(impulse['normal_impulse'],-1e-9)
            self.assertLessEqual(abs(impulse['tangent_impulse']),
                                 impulse['mu']*impulse['normal_impulse']+1e-9)
        f=[x for x in impulses if x['name']=='foot'][0]
        self.assertGreater(abs(f['tangential_speed']),.03)

    def test_body_corners_have_finite_collision_geometry(self):
        q=initial_state(self.p,.25)[:5].copy()
        q[2]=np.pi/2
        q[1]=.05
        self.assertLess(min(pt[1][1] for pt in body_corners(q,self.p)),0.)
        self.assertTrue(any(e['name'].startswith('body') for e in
                            contact_candidates(q,self.p,self.cfg)))

    def test_passive_energy_does_not_explode(self):
        c=Controller(stance_kp=0.,stance_kd=0.,flight_kp=0.,flight_kd=0.,
                     wheel_rate_damping=0.)
        state=initial_state(self.p,.3)
        E0=mechanical_energy(state,self.p)
        peak=E0
        for _ in range(350):
            state,_=step(state,0.,0.,True,self.p,c,self.cfg)
            peak=max(peak,mechanical_energy(state,self.p))
        # Finite-step numerical energy error is possible; this is a bound,
        # not a claim of perfect energy preservation.
        self.assertLess(peak,E0+5.)
        self.assertLess(mechanical_energy(state,self.p),E0+2.)

    def test_one_hop_land_safe(self):
        result=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        self.assertEqual(result['event'],'touchdown')
        self.assertTrue(result['ok'])
        self.assertLess(abs(result['touchdown_pitch']),self.cfg.max_touchdown_pitch)

    def test_two_hop_planner_and_goal(self):
        result=run_controller(target=3.,compression=.4,p=self.p,c=self.c,cfg=self.cfg,
              goal=GoalOptions(target_x=3.,safety_max_hops=8),
              options=PlannerOptions(lookahead=2,maxfev=12,verbose=False))
        self.assertEqual(result['status'],'goal_tolerance_reached')
        self.assertGreaterEqual(len(result['hop_info']),2)
        self.assertLess(abs(result['final_x']-3.),.05)
        self.assertTrue(all(abs(h['touchdown_pitch'])<self.cfg.max_touchdown_pitch
                            for h in result['hop_info']))

# Extra tests for the recovery-aware patch (kept separate from original suite).
from unittest.mock import patch
from simulator import recovery_probe
from planner import trajectory_cost, arrival_speed_cost

class TestRecoveryChanges(unittest.TestCase):
    def setUp(self):
        self.p=Robot();self.c=Controller();self.cfg=SimulationConfig(dt=.004)

    def test_impact_changes_pitch_rate_and_is_reported(self):
        h=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        self.assertEqual(h['event'],'touchdown')
        self.assertTrue(np.isfinite(h['postimpact_rate']))
        self.assertTrue(np.isfinite(h['postimpact_vx']))
        self.assertGreater(abs(h['postimpact_rate']-h['touchdown_rate']),0.1)

    def test_recovery_probe_does_not_alter_state(self):
        h=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        state=h['state'].copy()
        probe=recovery_probe(state,self.p,self.c,self.cfg,.04)
        np.testing.assert_array_equal(state,h['state'])
        self.assertEqual(len(probe['attempts']),3)
        self.assertTrue(all('event' in r for r in probe['attempts']))

    def test_arrival_penalty_accounts_for_velocity_and_direction(self):
        o=PlannerOptions()
        fast=arrival_speed_cost(14.,8.,15.,o)
        slow=arrival_speed_cost(14.,1.,15.,o)
        away=arrival_speed_cost(16.,8.,15.,o)
        self.assertGreater(fast,slow)
        self.assertGreater(away,fast)

    def test_optimizer_cost_accounts_for_postimpact_pitch_rate(self):
        o=PlannerOptions()
        hop=dict(event='touchdown',touchdown_pitch=0.,touchdown_rate=0.,
                 slip_duration=0.,postimpact_vx=0.,postimpact_rate=0.)
        base=dict(foot_x=1.,ok=True,steps=[hop],final_state=np.zeros(10))
        a=trajectory_cost(base,[(0.,0.)],GoalOptions(target_x=1.),o)
        hop2=dict(hop,postimpact_rate=4.)
        b=trajectory_cost(dict(base,steps=[hop2]),[(0.,0.)],GoalOptions(target_x=1.),o)
        self.assertGreater(b,a)

    def test_no_plan_uses_active_emergency_control_and_records_trace(self):
        # A planning failure must not automatically switch the motor OFF.
        fake_plan=dict(ref=(.20,0.),score=0.,predicted_event='touchdown',
                       predicted_safe=True,evaluations=1)
        with patch('planner.optimize_horizons',return_value=(None,[])), \
             patch('planner.optimize_emergency',return_value=fake_plan):
            r=run_controller(target=10.,compression=.3,p=self.p,c=self.c,cfg=self.cfg,
                goal=GoalOptions(target_x=10.,safety_max_hops=1),
                options=PlannerOptions(lookahead=1,verbose=False))
        self.assertGreater(r['total_time'],.1)
        self.assertTrue(r['history'])
        self.assertTrue(r['recovery_events'])
        self.assertTrue(r['hop_info'][0]['active_emergency'])
        self.assertEqual(r['recovery_events'][0]['control_mode'],'active_emergency')
        self.assertTrue(any(abs(row[1])>1e-6 for row in r['motor_history']))

    def test_full_hop_probe_simulates_complete_next_landing(self):
        from simulator import full_hop_probe
        first=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        saved=first['state'].copy()
        result=full_hop_probe(saved,self.p,self.c,self.cfg,
                              references=[(0.,0.)])
        np.testing.assert_allclose(saved,first['state'])
        self.assertEqual(result['tested_full_hops'],1)
        self.assertEqual(result['attempts'][0]['event'],'touchdown')
        self.assertGreater(result['attempts'][0]['time'],self.cfg.min_flight_duration)

    def test_terminal_full_hop_can_reject_plan(self):
        from planner import _rollout_with_terminal_recovery
        with patch('planner.full_hop_probe',return_value=dict(
                viable=False,best=None,attempts=[],tested_full_hops=5)):
            r=_rollout_with_terminal_recovery(initial_state(self.p,.4),[(.2,0.)],
                self.p,self.c,self.cfg,GoalOptions(target_x=3.),PlannerOptions())
        self.assertFalse(r['ok'])
        self.assertEqual(r['event'],'no_safe_complete_next_hop_found')

    def test_at_target_terminal_does_not_need_another_hop(self):
        from planner import _rollout_with_terminal_recovery
        first=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        with patch('planner.full_hop_probe') as probe:
            r=_rollout_with_terminal_recovery(initial_state(self.p,.4),[(.2,0.)],
                self.p,self.c,self.cfg,GoalOptions(target_x=first['foot_x']),
                PlannerOptions())
        self.assertTrue(r['ok'])
        probe.assert_not_called()

    def test_emergency_optimizer_returns_bounded_control_references(self):
        from planner import optimize_emergency
        h=one_hop(initial_state(self.p,.4),.2,0.,self.p,self.c,self.cfg)
        opts=PlannerOptions(emergency_maxfev=10)
        r=optimize_emergency(h['state'],GoalOptions(target_x=5.),
                             self.p,self.c,self.cfg,opts)
        self.assertLessEqual(abs(r['ref'][0]),self.c.max_stance_ref+1e-8)
        self.assertLessEqual(abs(r['ref'][1]),self.c.max_flight_ref+1e-8)
        self.assertGreaterEqual(r['evaluations'],9)


if __name__=='__main__':unittest.main()
