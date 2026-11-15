"""2D passive-spring, reaction-wheel hopping robot: one-hop optimal control demo.

State during stance: [alpha, phi, l, alpha_dot, phi_dot, l_dot]
 alpha: leg/body angle from vertical, positive tilts toward +x
 phi: absolute (inertial-frame) flywheel angle
 l: distance from planted foot to center of mass

The stance foot is a pinned no-slip contact. The flywheel is colocated with
body COM. The massless spring/leg is attached to the body and has zero
uncompressed force at l=l0. No actuator compresses the spring.

The one-hop planner optimizes stance torques and a free takeoff time with
CasADi/Ipopt. Takeoff is approximated by l=l0, l_dot>0; an independent
hybrid simulation ends stance at that event and detects landing by foot
contact during ballistic flight. The flight motor has a simple PD controller,
not a separate optimized phase. Thrust is supported in the flight simulation
interface but is zero by default.

This is a teaching prototype, not a validated hardware controller.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


@dataclass(frozen=True)
class Robot:
    m: float = 1.0        # total translating mass [kg]
    Ib: float = 0.015     # body pitch inertia about COM [kg m^2]
    Iw: float = 0.003     # flywheel spin inertia [kg m^2]
    k: float = 350.0      # passive spring stiffness [N/m]
    l0: float = 0.35      # spring free length [m]
    g: float = 9.81
    tau_max: float = 16.0  # motor torque bound [Nm]
    wheel_speed_max: float = 160.0  # absolute flywheel speed [rad/s]
    mu: float = 1.0       # contact friction coefficient
    flight_kp: float = 1.0  # reaction-wheel attitude PD [Nm/rad]
    flight_kd: float = 0.20 # [Nm s/rad]


def stance_rhs(y, tau, p: Robot):
    """NumPy or CasADi-compatible stance dynamics."""
    a, ph, l, ad, phd, ld = (y[i] for i in range(6))
    trig_sin = ca.sin if isinstance(a, (ca.SX, ca.MX, ca.DM)) else np.sin
    trig_cos = ca.cos if isinstance(a, (ca.SX, ca.MX, ca.DM)) else np.cos
    s, c = trig_sin(a), trig_cos(a)

    # Lagrange equations for stance; inertial flywheel angle ph.
    add = (-tau - 2*p.m*l*ld*ad + p.m*p.g*l*s) / (p.Ib+p.m*l*l)
    phdd = tau/p.Iw
    ldd = l*ad*ad - p.g*c - p.k*(l-p.l0)/p.m
    if isinstance(a, (ca.SX, ca.MX, ca.DM)):
        return ca.vertcat(ad, phd, ld, add, phdd, ldd)
    return np.array([ad, phd, ld, add, phdd, ldd], dtype=float)


def ground_reaction(y, tau, p: Robot):
    """Required ground reaction for the pinned-foot stance assumption."""
    a, _, l, ad, _, ld = (y[i] for i in range(6))
    dy = stance_rhs(y, tau, p)
    add, ldd = dy[3], dy[5]
    trig_sin = ca.sin if isinstance(a, (ca.SX, ca.MX, ca.DM)) else np.sin
    trig_cos = ca.cos if isinstance(a, (ca.SX, ca.MX, ca.DM)) else np.cos
    s, c = trig_sin(a), trig_cos(a)
    xdd = ldd*s + 2*ld*ad*c + l*add*c - l*ad*ad*s
    zdd = ldd*c - 2*ld*ad*s - l*add*s - l*ad*ad*c
    return p.m*xdd, p.m*(zdd+p.g)


def center_of_mass(y, p: Robot, foot_x=0.0):
    a, _, l, ad, _, ld = y
    x = foot_x + l*np.sin(a)
    z = l*np.cos(a)
    vx = ld*np.sin(a)+l*ad*np.cos(a)
    vz = ld*np.cos(a)-l*ad*np.sin(a)
    return x, z, vx, vz


def rk4_symbolic(x, u, dt, p: Robot):
    k1 = stance_rhs(x, u, p)
    k2 = stance_rhs(x+0.5*dt*k1, u, p)
    k3 = stance_rhs(x+0.5*dt*k2, u, p)
    k4 = stance_rhs(x+dt*k3, u, p)
    return x+(dt/6)*(k1+2*k2+2*k3+k4)


def initial_state(p: Robot, compression=0.12):
    """Spring compression is preloaded. This initial potential energy powers hop 1."""
    return np.array([0., 0., p.l0-compression, 0., 0., 0.])


def baseline_takeoff(p: Robot, y0):
    """Feasible zero-torque trajectory to seed the NLP."""
    def event(t, y):
        return y[2]-p.l0
    event.direction = 1
    event.terminal = True
    sol=solve_ivp(lambda t,y: stance_rhs(y,0.,p), (0.,0.5), y0,
                  events=event, dense_output=True, rtol=1e-10, atol=1e-11,
                  max_step=0.001)
    if len(sol.t_events[0]) == 0:
        raise ValueError('No takeoff with this preload. Increase compression or change parameters.')
    return float(sol.t_events[0][0]), sol.sol


def plan_one_hop(p: Robot, y0, target_x, N=24, verbose=False):
    """IPOPT optimizes a single stance trajectory and its takeoff time.

    Predicted touchdown uses ballistic COM trajectory and upright touchdown
    with COM height l0. It is an approximation: we independently simulate
    touchdown with the full flight attitude controller and foot-ground event.
    """
    base_T, base_sol = baseline_takeoff(p,y0)
    opti=ca.Opti()
    X=opti.variable(6,N+1)
    U=opti.variable(N)
    T=opti.variable()  # unknown stance duration
    dt=T/N

    opti.subject_to(X[:,0]==y0)
    opti.subject_to(opti.bounded(max(0.002,base_T*0.6),T,min(0.25,base_T*1.6)))
    opti.subject_to(opti.bounded(-p.tau_max,U,p.tau_max))
    opti.subject_to(opti.bounded(-p.wheel_speed_max,X[4,:],p.wheel_speed_max))
    opti.subject_to(opti.bounded(0.15,X[2,:],p.l0))
    opti.subject_to(X[2,N]==p.l0)
    opti.subject_to(X[5,N]>=0.20)  # extending at takeoff

    J=0
    for k in range(N):
        xk=X[:,k]
        u=U[k]
        opti.subject_to(X[:,k+1]==rk4_symbolic(xk,u,dt,p))
        fx, fn = ground_reaction(xk,u,p)
        # no adhesive force, no foot slip (checked at shooting nodes)
        opti.subject_to(fn>=0)
        opti.subject_to(fx<=p.mu*fn)
        opti.subject_to(-fx<=p.mu*fn)
        # Also check the end of the interval before the torque switches.
        fx_e, fn_e = ground_reaction(X[:,k+1], u, p)
        opti.subject_to(fn_e>=-0.005)
        opti.subject_to(fx_e<=p.mu*(fn_e+0.005))
        opti.subject_to(-fx_e<=p.mu*(fn_e+0.005))
        du=u if k==0 else u-U[k-1]
        J += dt*(0.015*u*u + 0.025*du*du)

    # Near takeoff, a disappearing contact must exert negligible force.
    fx_to, fn_to = ground_reaction(X[:,N], U[N-1], p)
    opti.subject_to(opti.bounded(-0.03,fx_to,0.03))
    opti.subject_to(opti.bounded(-0.03,fn_to,0.03))

    # Predicted COM position and velocity at takeoff.
    a,_,l,ad,phd,ld=[X[i,N] for i in range(6)]
    x_to=l*ca.sin(a)
    z_to=l*ca.cos(a)
    vx_to=ld*ca.sin(a)+l*ad*ca.cos(a)
    vz_to=ld*ca.cos(a)-l*ad*ca.sin(a)
    opti.subject_to(vz_to>=0.25)
    # Assume flight attitude controller makes the leg vertical at touchdown:
    z_td=p.l0
    radicand=vz_to*vz_to+2*p.g*(z_to-z_td)
    opti.subject_to(radicand>=0.01)
    t_flight=(vz_to+ca.sqrt(radicand))/p.g
    predicted_foot_landing_x=x_to+vx_to*t_flight

    # Target landing vs secondary safety/effort objectives.
    J+=1200*(predicted_foot_landing_x-target_x)**2
    J+=0.2*a*a + 0.015*ad*ad + 0.00015*phd*phd
    opti.minimize(J)
    opti.solver('ipopt',{'expand':True,'print_time':verbose},
                {'print_level':5 if verbose else 0,'max_iter':1200,
                 'tol':1e-7,'acceptable_tol':1e-5})

    # Physically feasible seed: unpowered hop with same spring preload.
    opti.set_initial(T,base_T)
    opti.set_initial(U,np.zeros(N))
    opti.set_initial(X,np.column_stack([base_sol(base_T*k/N) for k in range(N+1)]))
    try:
        sol=opti.solve()
    except RuntimeError as exc:
        opti.debug.show_infeasibilities(1e-5)
        raise RuntimeError(
            'IPOPT did not converge. Try a smaller target or different robot parameters.'
        ) from exc
    return {'t_takeoff':float(sol.value(T)),
            'X':np.array(sol.value(X)),
            'U':np.asarray(sol.value(U),dtype=float).ravel(),
            'predicted_x':float(sol.value(predicted_foot_landing_x)),
            'predicted_flight_time':float(sol.value(t_flight)),
            'cost':float(sol.value(J))}


def simulate_one_hop(p: Robot, y0, plan, flight_thrust=None):
    """Independent hybrid ODE simulation using planned stance torques.

    flight_thrust: callable(t, state)->(Fx,Fz), or None (no thrusters).
    Returns dense samples of stance + flight, including true foot touchdown.
    """
    N=len(plan['U'])
    dt_plan=plan['t_takeoff']/N
    stance_times=[0.]
    stance_states=[np.asarray(y0,dtype=float).copy()]
    stance_torques=[float(plan['U'][0])]
    y=np.asarray(y0,dtype=float).copy()
    toe_time=None
    for k in range(N+1):
        if k<N:
            tau=float(plan['U'][k]); t0=k*dt_plan; t1=(k+1)*dt_plan
        else:
            # Small safety margin for integration drift after planned terminal time.
            tau=0.;t0=N*dt_plan;t1=t0+0.04
        def takeoff(t,x):return x[2]-p.l0
        takeoff.terminal=True
        takeoff.direction=1
        step=solve_ivp(lambda t,z: stance_rhs(z,tau,p),(t0,t1),y,
                       events=takeoff,rtol=1e-9,atol=1e-10,max_step=0.001)
        for j in range(1,len(step.t)):
            stance_times.append(float(step.t[j]))
            stance_states.append(step.y[:,j].copy())
            stance_torques.append(tau)
        y=step.y[:,-1]
        if len(step.t_events[0]):
            toe_time=float(step.t_events[0][0])
            y=step.y_events[0][0]
            break
    if toe_time is None:
        raise RuntimeError('Simulated spring never reached takeoff length. Try lower target or finer N.')

    # Flight: [xCOM,zCOM,vxCOM,vzCOM,alpha,phi,alpha_dot,phi_dot]
    x,z,vx,vz=center_of_mass(y,p)
    flight_y0=np.array([x,z,vx,vz,y[0],y[1],y[3],y[4]])

    def flight_motor(s):
        # Positive wheel torque counters positive body angle/rotation.
        t=np.clip(p.flight_kp*s[4]+p.flight_kd*s[6], -p.tau_max,p.tau_max)
        if s[7] >= p.wheel_speed_max and t>0: t=0.
        if s[7] <=-p.wheel_speed_max and t<0: t=0.
        return t

    def flight_rhs(t,s):
        ux,uz=(0.,0.) if flight_thrust is None else flight_thrust(t,s)
        tau=flight_motor(s)
        return [s[2],s[3],ux/p.m,uz/p.m-p.g,
                s[6],s[7],-tau/p.Ib,tau/p.Iw]

    def contact_event(t,s):
        # The leg is uncompressed and length l0 in flight.
        return s[1]-p.l0*np.cos(s[4])
    contact_event.terminal=True
    contact_event.direction=-1

    flight=solve_ivp(flight_rhs,(toe_time,toe_time+2.0),flight_y0,
                     events=contact_event,rtol=1e-9,atol=1e-10,max_step=0.002)
    if not len(flight.t_events[0]):
        raise RuntimeError('No ground contact during flight simulation.')
    touchdown=flight.y_events[0][0]
    landing_x=float(touchdown[0]-p.l0*np.sin(touchdown[4]))

    # Diagnostics: contact validity (ground reaction should be compressive and
    # inside Coulomb cone during stance). Fine-grid checks are not enforced
    # by optimizer, so a violation warns that this idealization was exceeded.
    forces=np.array([ground_reaction(s,u,p) for s,u in
                     zip(stance_states,stance_torques)])
    min_normal=float(forces[:,1].min())
    max_slip_violation=float(np.maximum(0,np.abs(forces[:,0])-p.mu*forces[:,1]).max())

    return {
        'stance_t':np.asarray(stance_times),
        'stance_y':np.asarray(stance_states),
        'stance_u':np.asarray(stance_torques),
        'flight_t':flight.t,
        'flight_y':flight.y.T,
        'takeoff_time':toe_time,
        'touchdown_time':float(flight.t_events[0][0]),
        'landing_x':landing_x,
        'touchdown_alpha':float(touchdown[4]),
        'touchdown_vx':float(touchdown[2]),
        'touchdown_vz':float(touchdown[3]),
        'min_normal':min_normal,
        'max_friction_violation':max_slip_violation
    }


def plot_results(p: Robot, plan, result, target, out_dir, animate=True):
    out_dir=Path(out_dir)
    out_dir.mkdir(parents=True,exist_ok=True)
    sy=result['stance_y'];fy=result['flight_y']
    sx=sy[:,2]*np.sin(sy[:,0]);sz=sy[:,2]*np.cos(sy[:,0])
    fx,fz=fy[:,0],fy[:,1]
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    axs[0,0].plot(sx,sz,'-',label='Stance COM')
    axs[0,0].plot(fx,fz,'-',label='Flight COM')
    axs[0,0].axvline(target,color='black',ls='--',alpha=.6,label='Desired foot x')
    axs[0,0].plot(result['landing_x'],p.l0*np.cos(result['touchdown_alpha']),
                  'o',label='Actual landing COM')
    axs[0,0].set(xlabel='x [m]',ylabel='z [m]',title='COM trajectory')
    axs[0,0].axis('equal');axs[0,0].legend(fontsize=8);axs[0,0].grid(True)
    axs[0,1].step(result['stance_t'],result['stance_u'],where='post');axs[0,1].set(xlabel='Stance time [s]',
                  ylabel='Flywheel torque [Nm]',title='Optimized stance torque')
    axs[0,1].grid(True)
    axs[1,0].plot(result['stance_t'],sy[:,0],label='Stance')
    axs[1,0].plot(result['flight_t'],fy[:,4],label='Flight')
    axs[1,0].set(xlabel='Time [s]',ylabel='Body angle [rad]',title='Body orientation')
    axs[1,0].legend();axs[1,0].grid(True)
    axs[1,1].plot(result['stance_t'],sy[:,2],label='Leg length')
    axs[1,1].axhline(p.l0,color='gray',ls='--',label='Free length')
    axs[1,1].set(xlabel='Time [s]',ylabel='Leg length [m]',title='Spring expansion')
    axs[1,1].legend();axs[1,1].grid(True)
    fig.savefig(out_dir/'one_hop_plots.png',dpi=160)
    plt.close(fig)

    if not animate:
        return

    # Animation samples are interpolated from independent simulation.
    total=result['touchdown_time']
    samples=np.linspace(0,total,110)
    all_t=np.r_[result['stance_t'],result['flight_t'][1:]]
    body_x=np.r_[sx,fx[1:]];body_z=np.r_[sz,fz[1:]]
    all_a=np.r_[sy[:,0],fy[1:,4]]
    all_l=np.r_[sy[:,2],np.full(len(fy)-1,p.l0)]
    cx=np.interp(samples,all_t,body_x)
    cz=np.interp(samples,all_t,body_z)
    ang=np.interp(samples,all_t,all_a)
    lengths=np.interp(samples,all_t,all_l)
    toe_x=cx-lengths*np.sin(ang)
    toe_z=cz-lengths*np.cos(ang)
    fig,ax=plt.subplots(figsize=(8,4.5))
    ax.axhline(0,color='black',linewidth=1)
    ax.axvline(target,color='tab:green',ls='--',alpha=.6,label='Target foot location')
    ax.plot(body_x,body_z,color='0.75',lw=1,label='COM trajectory')
    line,=ax.plot([],[],'-',color='tab:blue',lw=4)
    body,=ax.plot([],[],'o',color='tab:blue',ms=16)
    wheel,=ax.plot([],[],'o',color='white',mec='tab:blue',ms=7)
    ground_tip,=ax.plot([],[],'o',color='tab:orange',ms=5)
    msg=ax.text(.02,.95,'',transform=ax.transAxes,va='top')
    xmin=min(-.25,float(np.min(toe_x))-.2,target-.2)
    xmax=max(.4,float(np.max(toe_x))+.2,target+.2)
    ax.set(xlim=(xmin,xmax),ylim=(-.05,max(.7,float(np.max(cz))+.18)),
           xlabel='Horizontal distance [m]',ylabel='Height [m]',title='One-hop hybrid simulation')
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='upper right',fontsize=8)

    def update(i):
        line.set_data([toe_x[i],cx[i]],[toe_z[i],cz[i]])
        body.set_data([cx[i]],[cz[i]])
        wheel.set_data([cx[i]],[cz[i]])
        ground_tip.set_data([toe_x[i]],[toe_z[i]])
        mode='STANCE' if samples[i]<=result['takeoff_time'] else 'FLIGHT'
        msg.set_text(f't = {samples[i]:.2f} s | {mode}')
        return line,body,wheel,ground_tip,msg
    ani=FuncAnimation(fig,update,frames=len(samples),interval=45,blit=True)
    ani.save(out_dir/'one_hop.gif',writer=PillowWriter(fps=22))
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='2D hopper: single-hop CasADi/IPOPT + hybrid simulation')
    parser.add_argument('--target',type=float,default=0.20,help='Desired first-hop foot landing x [m]')
    parser.add_argument('--compression',type=float,default=0.12,help='Initial spring compression [m]')
    parser.add_argument('--nodes',type=int,default=24)
    parser.add_argument('--output',default='results')
    parser.add_argument('--no-gif',action='store_true')
    parser.add_argument('--verbose-ipopt',action='store_true')
    args=parser.parse_args()
    p=Robot()
    y0=initial_state(p,args.compression)
    plan=plan_one_hop(p,y0,args.target,N=args.nodes,verbose=args.verbose_ipopt)
    result=simulate_one_hop(p,y0,plan)
    print(f'Desired foot landing x:    {args.target:+.3f} m')
    print(f'Predicted foot landing x:  {plan["predicted_x"]:+.3f} m')
    print(f'Simulated foot landing x:  {result["landing_x"]:+.3f} m')
    print(f'Landing position error:    {result["landing_x"]-args.target:+.3f} m')
    print(f'Takeoff time:              {result["takeoff_time"]:.3f} s')
    print(f'Landing time:              {result["touchdown_time"]:.3f} s')
    print(f'Touchdown body angle:      {np.degrees(result["touchdown_alpha"]):+.2f} deg')
    print(f'Minimum stance normal:     {result["min_normal"]:.3f} N')
    print(f'Max friction violation:    {result["max_friction_violation"]:.3f} N')
    if result['min_normal'] < -0.05 or result['max_friction_violation'] > 0.1:
        print('WARNING: Contact model violated between shooting nodes; results are not physically feasible.')
    plot_results(p,plan,result,args.target,args.output,animate=not args.no_gif)
    print(f'Output saved to: {Path(args.output).resolve()}')


if __name__=='__main__':
    main()
