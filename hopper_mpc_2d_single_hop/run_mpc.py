"""Receding-horizon single-hop MPC using the simulator in hopper_2d.py.

Re-solves the nonlinear stance optimization every --mpc-period seconds,
using the freshly simulated robot state, until the final short stance
interval. Each solve changes the prediction horizon to the remaining
estimated stance time. A simple attitude PD controller handles flight.

This is a non-real-time demonstration: solving IPOPT on a PC can be much
slower than the physical motor command interval.
"""
from __future__ import annotations
import argparse
import numpy as np
from scipy.integrate import solve_ivp

from hopper_2d import (Robot, initial_state, stance_rhs, plan_one_hop,
                       simulate_one_hop, plot_results, ground_reaction)


def advance_prefix(p, y0, plan, duration):
    """Apply only the first duration of the optimized control schedule."""
    dt=plan['t_takeoff']/len(plan['U'])
    time=[0.]
    ys=[np.asarray(y0,dtype=float)]
    us=[float(plan['U'][0])]
    for k,u in enumerate(plan['U']):
        t0=k*dt
        if t0>=duration-1e-10: break
        t1=min((k+1)*dt,duration)
        # Stop if contact releases unexpectedly.
        def event(t,y):return y[2]-p.l0
        event.direction=1
        event.terminal=True
        res=solve_ivp(lambda t,s:stance_rhs(s,float(u),p),
                      (t0,t1),ys[-1],events=event,
                      max_step=0.001,rtol=1e-9,atol=1e-10)
        for j in range(1,len(res.t)):
            time.append(float(res.t[j]))
            ys.append(res.y[:,j])
            us.append(float(u))
        if len(res.t_events[0]):
            return np.array(time),np.array(ys),np.array(us),True
    return np.array(time),np.array(ys),np.array(us),False


def _finish_contact_checks(result, p):
    # Recompute contact validity over the *entire* receding-horizon rollout.
    force=np.array([ground_reaction(y,u,p) for y,u in zip(
        result['stance_y'],result['stance_u'])])
    result['min_normal']=float(force[:,1].min())
    result['max_friction_violation']=float(
        np.maximum(0,np.abs(force[:,0])-p.mu*force[:,1]).max()
    )


def run_feedback(p, y0, target, nodes=24, period=0.02):
    tnow=0.
    y=np.array(y0,dtype=float)
    all_t=[0.]
    all_y=[y.copy()]
    all_u=[0.]
    solves=0
    for cycle in range(15):
        plan=plan_one_hop(p,y,target,N=nodes)
        solves+=1
        if plan['t_takeoff'] <= 1.5*period:
            # Don't replan in a very short last interval.
            result=simulate_one_hop(p,y,plan)
            result['stance_t']+=tnow
            result['flight_t']+=tnow
            result['takeoff_time']+=tnow
            result['touchdown_time']+=tnow
            result['stance_t']=np.r_[all_t,result['stance_t'][1:]]
            result['stance_y']=np.vstack([all_y,result['stance_y'][1:]])
            result['stance_u']=np.r_[all_u,result['stance_u'][1:]]
            _finish_contact_checks(result,p)
            result['mpc_solves']=solves
            return plan,result
        t_prefix,y_prefix,u_prefix,early=advance_prefix(p,y,plan,period)
        if early:
            # Rare numerical case: safely complete the entire last plan.
            result=simulate_one_hop(p,y,plan)
            result['stance_t']+=tnow
            result['flight_t']+=tnow
            result['takeoff_time']+=tnow
            result['touchdown_time']+=tnow
            result['stance_t']=np.r_[all_t,result['stance_t'][1:]]
            result['stance_y']=np.vstack([all_y,result['stance_y'][1:]])
            result['stance_u']=np.r_[all_u,result['stance_u'][1:]]
            _finish_contact_checks(result,p)
            result['mpc_solves']=solves
            return plan,result
        all_t.extend((t_prefix[1:]+tnow).tolist())
        all_y.extend(y_prefix[1:])
        all_u.extend(u_prefix[1:])
        tnow+=period
        y=y_prefix[-1]
    raise RuntimeError('MPC stance exceeded 15 cycles. Check motor/spring parameters.')


def main():
    parser=argparse.ArgumentParser(description='Receding-horizon one-hop MPC')
    parser.add_argument('--target',type=float,default=0.20)
    parser.add_argument('--compression',type=float,default=0.12)
    parser.add_argument('--mpc-period',type=float,default=0.020)
    parser.add_argument('--nodes',type=int,default=24)
    parser.add_argument('--output',default='results_mpc')
    parser.add_argument('--no-gif',action='store_true')
    args=parser.parse_args()
    p=Robot()
    y0=initial_state(p,args.compression)
    plan,result=run_feedback(p,y0,args.target,args.nodes,args.mpc_period)
    print(f'Re-optimizations in stance: {result["mpc_solves"]}')
    print(f'Predicted landing, final solve: {plan["predicted_x"]:.4f} m')
    print(f'Simulated foot landing: {result["landing_x"]:.4f} m')
    print(f'Landing error: {result["landing_x"]-args.target:+.4f} m')
    print(f'Takeoff: {result["takeoff_time"]:.4f} s, '
          f'touchdown: {result["touchdown_time"]:.4f} s')
    print(f'Touchdown tilt: {np.degrees(result["touchdown_alpha"]):.2f} deg')
    print(f'Minimum stance normal: {result["min_normal"]:.3f} N')
    print(f'Max friction violation: {result["max_friction_violation"]:.3f} N')
    plot_results(p,plan,result,args.target,args.output,animate=not args.no_gif)
    print(f'Plots saved under {args.output}')

if __name__=='__main__':
    main()
