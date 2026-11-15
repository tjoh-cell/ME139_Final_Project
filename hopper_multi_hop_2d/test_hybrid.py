"""Sanity checks: mechanics, hybrid events, and target tracking."""
import unittest
import numpy as np
from hybrid_hopper import (Robot, Control, initial_state, impact_reset,
                           impact_impulse_diagnostics, com_from_stance,
                           run_hops, stance_energy, simulate_stance,
                           simulate_flight, to_flight, flight_energy)


class HybridHopperTests(unittest.TestCase):
    def test_impact_does_not_create_kinetic_energy_and_pins_foot(self):
        p=Robot()
        # Touchdown exactly at z=l0*cos(alpha), with downward foot velocity.
        alpha=0.16
        y=np.array([0.23,p.l0*np.cos(alpha),0.32,-1.8,alpha,1.5,0.30,-20.0])
        foot_x,s,loss=impact_reset(y,p)
        self.assertGreaterEqual(loss,-1e-10)
        self.assertGreater(s[5]*s[5],0)
        x,z,vx,vz=com_from_stance(s,foot_x)
        self.assertAlmostEqual(x-p.l0*np.sin(alpha),foot_x,places=10)
        self.assertAlmostEqual(z-p.l0*np.cos(alpha),0,places=10)
        # Under a pinned foot, foot velocity is identically zero.
        self.assertAlmostEqual(vx-s[5]*np.sin(alpha)-s[2]*s[3]*np.cos(alpha),0,places=10)
        self.assertAlmostEqual(vz-s[5]*np.cos(alpha)+s[2]*s[3]*np.sin(alpha),0,places=10)
        # Do not accidentally treat tangential impulse as friction-feasible.
        jx,jz,shortfall=impact_impulse_diagnostics(y,s,foot_x,p)
        self.assertGreaterEqual(shortfall,0)

    def test_single_vertical_passive_hop_energy_decreases(self):
        # With tau_max=0, there is no external work from the motor.
        p=Robot(tau_max=0,c=2.0)
        c=Control()
        y0=initial_state(p)
        e0=stance_energy(y0,p)
        st=simulate_stance(y0,0,0,p,c,accurate=True)
        self.assertEqual(st['event'],'takeoff')
        e_to=stance_energy(st['state'],p)
        self.assertLess(e_to,e0-0.01)
        f=simulate_flight(to_flight(st['state'],0,p),p,c,st['end'],accurate=True)
        self.assertEqual(f['event'],'touchdown')
        self.assertAlmostEqual(flight_energy(f['state'],p),e_to,places=5)
        newfoot,yplus,loss=impact_reset(f['state'],p)
        self.assertLessEqual(stance_energy(yplus,p,newfoot),flight_energy(f['state'],p)+1e-7)

    def test_three_hops_toward_goal(self):
        p=Robot(); c=Control()
        result=run_hops(p,c,target=.45,n_hops=5,verbose=False)
        self.assertGreaterEqual(len(result['hops']),3)
        self.assertLess(abs(result['final_foot_x']-.45),.025)
        for i,seg in enumerate(result['segments']):
            if i%2==0:
                self.assertEqual(seg['event'],'takeoff')
                self.assertGreater(seg['normal_min'],-1e-4)
                self.assertLess(seg['max_slip'],1e-4)
                self.assertGreater(seg['minimum_length'],p.min_length-1e-5)
            else:
                self.assertEqual(seg['event'],'touchdown')
                self.assertAlmostEqual(seg['state'][1]-p.l0*np.cos(seg['state'][4]),0.,places=5)


if __name__=='__main__':
    unittest.main()
