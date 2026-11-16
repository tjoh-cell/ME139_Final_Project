"""2-D passive-spring / reaction-wheel hopping robot with repeated contacts.

This is a reduced-order TEACHING MODEL, not a validated contact model.
Stance: point foot anchored, leg/body pitch angle alpha, massless leg.
Flight: rigid body plus wheel, no thrusters, leg held at its free length.
Touchdown: a massless sticking point foot, with a dissipative inelastic
velocity projection to satisfy the fixed-foot stance constraints.

State in stance: [alpha, phi, l, alpha_dot, phi_dot, l_dot]
State in flight: [x, z, vx, vz, alpha, phi, alpha_dot, phi_dot]
phi is flywheel inertial angle, not wheel angle relative to the body.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Robot:
    m: float = 1.0
    Ib: float = 0.015
    Iw: float = 0.003
    k: float = 450.0
    c: float = 1.0      # spring compression damping [N s/m]
    l0: float = 0.45
    g: float = 9.81
    tau_max: float = 2.0
    wheel_speed_max: float = 160.0
    mu: float = 1.5
    min_length: float = 0.18
    max_stance: float = 1.2
    max_flight: float = 2.0


@dataclass(frozen=True)
class Control:
    stance_kp: float = 6.0
    stance_kd: float = 0.65
    flight_kp: float = 1.3
    flight_kd: float = 0.24
    flight_target: float = 0.0
    max_tilt_reference: float = 0.27


def initial_state(p: Robot, compression=0.12):
    if not 0 < compression < (p.l0-p.min_length):
        raise ValueError('Compression outside allowed range')
    return np.array([0.,0.,p.l0-compression,0.,0.,0.],dtype=float)


def com_from_stance(y, foot_x):
    a, _, l, ad, _, ld = y
    x = foot_x+l*np.sin(a)
    z = l*np.cos(a)
    vx = ld*np.sin(a)+l*ad*np.cos(a)
    vz = ld*np.cos(a)-l*ad*np.sin(a)
    return np.array([x,z,vx,vz],dtype=float)


def spring_force(y, p: Robot):
    l=float(y[2]); ld=float(y[5])
    if l >= p.l0 and ld >= 0: return 0.0
    return max(0.0, p.k*max(p.l0-l,0.0)-p.c*ld)


def stance_rhs(y,tau,p:Robot):
    a,phi,l,ad,phd,ld=np.asarray(y,dtype=float)
    f=spring_force(y,p)
    add=(-tau-2*p.m*l*ld*ad+p.m*p.g*l*np.sin(a))/(p.Ib+p.m*l*l)
    ldd=l*ad*ad-p.g*np.cos(a)+f/p.m
    return np.array([ad,phd,ld,add,tau/p.Iw,ldd])


def ground_reaction(y,tau,p:Robot):
    a,_,l,ad,_,ld=np.asarray(y,dtype=float)
    d=stance_rhs(y,tau,p)
    add,ldd=d[3],d[5]
    xdd=ldd*np.sin(a)+2*ld*ad*np.cos(a)+l*add*np.cos(a)-l*ad*ad*np.sin(a)
    zdd=ldd*np.cos(a)-2*ld*ad*np.sin(a)-l*add*np.sin(a)-l*ad*ad*np.cos(a)
    return float(p.m*xdd),float(p.m*(zdd+p.g))


def limit_wheel_torque(tau, omega, p:Robot):
    tau=float(np.clip(tau,-p.tau_max,p.tau_max))
    if omega >= p.wheel_speed_max and tau>0: return 0.0
    if omega <= -p.wheel_speed_max and tau<0: return 0.0
    return tau


def contact_limited_torque(desired,y,p:Robot):
    """Choose nearest torque satisfying normal>=0, |Fx|<=mu*N.
    For this model the reactions are affine functions of tau.
    No feasible torque -> returns zero + false (event logged as invalid).
    """
    lo,hi=-p.tau_max,p.tau_max
    omega=y[4]
    if omega>=p.wheel_speed_max: hi=min(hi,0.0)
    if omega<=-p.wheel_speed_max: lo=max(lo,0.0)
    f0,n0=ground_reaction(y,0.,p)
    f1_,n1_=ground_reaction(y,1.,p)
    f1,n1=f1_-f0,n1_-n0
    # Expressions of form slope*tau <= intercept
    conditions=[(-n1,n0), (f1-p.mu*n1,p.mu*n0-f0),
                (-f1-p.mu*n1,p.mu*n0+f0)]
    for slope,intercept in conditions:
        if abs(slope)<1e-12:
            if intercept<-1e-8: return 0.0,False
            continue
        bound=intercept/slope
        if slope>0: hi=min(hi,bound)
        else: lo=max(lo,bound)
    if hi<lo-1e-8: return 0.0,False
    return float(np.clip(desired,lo,hi)),True


def stance_torque(y,alpha_ref,p:Robot,c:Control):
    # Tau>0 accelerates the wheel positively and the body negatively.
    desired=c.stance_kp*(y[0]-alpha_ref)+c.stance_kd*y[3]
    return contact_limited_torque(desired,y,p)


def flight_torque(y,p:Robot,c:Control,flight_target=None):
    """Wheel attitude torque. Optional per-hop flight target keeps old API working.

    Targets are body pitch angles relative to upright (not launch angles).
    """
    if flight_target is None:
        flight_target=c.flight_target
    error=np.arctan2(np.sin(y[4]-flight_target),
                     np.cos(y[4]-flight_target))
    desired=c.flight_kp*error+c.flight_kd*y[6]
    return limit_wheel_torque(desired,y[7],p)


def flight_rhs(t,y,p:Robot,c:Control,flight_target=None):
    tau=flight_torque(y,p,c,flight_target)
    x,z,vx,vz,a,phi,ad,phd=y
    return np.array([vx,vz,0.,-p.g,ad,phd,-tau/p.Ib,tau/p.Iw])


def to_flight(y,foot_x,p:Robot):
    x,z,vx,vz=com_from_stance(y,foot_x)
    return np.array([x,z,vx,vz,y[0],y[1],y[3],y[4]],dtype=float)


def foot_position_flight(y,p:Robot):
    return float(y[0]-p.l0*np.sin(y[4])),float(y[1]-p.l0*np.cos(y[4]))


def impact_reset(y,p:Robot):
    """Sticking massless point-foot, perfectly inelastic tangential impact.

    The generalized mass matrix for [COM-x,COM-z,alpha,phi] has diagonal
    (m,m,Ib,Iw). Project pre-impact velocity onto the admissible fixed-foot
    stance velocity subspace. The leg can shorten immediately after contact.

    ld+ = e_r dot v_com-
    alpha_dot+ = (m*l*(e_t dot v_com-) + Ib*alpha_dot-)/(m*l^2+Ib)
    phi_dot+ = phi_dot- (no impulsive torque on wheel).
    This guarantees E_kinetic+ <= E_kinetic- up to rounding.
    """
    x,z,vx,vz,a,ph,ad,phd=np.asarray(y,dtype=float)
    l=p.l0
    er=np.array([np.sin(a),np.cos(a)])
    et=np.array([np.cos(a),-np.sin(a)])
    v=np.array([vx,vz])
    ldplus=float(er@v)
    adplus=float((p.m*l*(et@v)+p.Ib*ad)/(p.m*l*l+p.Ib))
    yplus=np.array([a,ph,l,adplus,phd,ldplus],dtype=float)
    foot_x=float(x-l*np.sin(a))
    e_before=0.5*p.m*(vx*vx+vz*vz)+0.5*p.Ib*ad*ad+0.5*p.Iw*phd*phd
    vplus=com_from_stance(yplus,foot_x)[2:]
    e_after=0.5*p.m*np.dot(vplus,vplus)+0.5*p.Ib*adplus*adplus+0.5*p.Iw*phd*phd
    loss=float(e_before-e_after)
    if loss<-1e-8: raise AssertionError('Impact created kinetic energy')
    return foot_x,yplus,max(0.0,loss)


def impact_impulse_diagnostics(yminus,yplus,foot_x,p:Robot):
    """Effective COM impulse needed for the sticking reset.

    This model's reset does NOT enforce the impulsive Coulomb friction cone.
    These diagnostics expose that approximation instead of hiding it.
    """
    vminus=np.asarray(yminus[2:4],dtype=float)
    vplus=com_from_stance(yplus,foot_x)[2:]
    jx,jz=p.m*(vplus-vminus)
    return float(jx),float(jz),float(max(0.,abs(jx)-p.mu*max(0.,jz)))


def stance_energy(y,p:Robot,foot_x=0.):
    _,z,vx,vz=com_from_stance(y,foot_x)
    return (0.5*p.m*(vx**2+vz**2)+0.5*p.Ib*y[3]**2+
            0.5*p.Iw*y[4]**2+p.m*p.g*z+
            0.5*p.k*max(p.l0-y[2],0.)**2)


def flight_energy(y,p:Robot):
    return (0.5*p.m*(y[2]**2+y[3]**2)+0.5*p.Ib*y[6]**2+
            0.5*p.Iw*y[7]**2+p.m*p.g*y[1])


def simulate_stance(y0, foot_x, alpha_ref, p:Robot, c:Control,
                    start_time=0., accurate=False):
    """Run compression/rebound until leg returns to free length in extension.
    No extension motor, no ground thrust. Returns solver plus peak compression.
    If the leg cannot return to free length, takeoff=False.
    """
    y0=np.asarray(y0,dtype=float)
    def f(t,y):
        tau,_=stance_torque(y,alpha_ref,p,c)
        return stance_rhs(y,tau,p)
    def takeoff(t,y): return y[2]-p.l0
    takeoff.terminal=True;takeoff.direction=1
    def bottom(t,y): return y[5]
    bottom.terminal=False;bottom.direction=1
    def hardstop(t,y): return y[2]-p.min_length
    hardstop.terminal=True;hardstop.direction=-1
    res=solve_ivp(f,(start_time,start_time+p.max_stance),y0,
                  events=[takeoff,bottom,hardstop],
                  rtol=1e-9 if accurate else 2e-6,
                  atol=1e-10 if accurate else 1e-8,
                  max_step=0.0015 if accurate else 0.012)
    event_name='takeoff' if len(res.t_events[0]) else (
               'hardstop' if len(res.t_events[2]) else 'stalled')
    # enforce (and diagnose) ground force and friction throughout stance
    ts=np.linspace(res.t[0],res.t[-1],max(75,int((res.t[-1]-res.t[0])/(0.003 if accurate else 0.012))+1))
    ys=np.array([np.interp(ts,res.t,res.y[i]) for i in range(6)]).T
    taus=[]; valid=[]; fxs=[];fns=[]
    for y in ys:
        tau,feasible=stance_torque(y,alpha_ref,p,c)
        fx,fn=ground_reaction(y,tau,p)
        taus.append(tau);valid.append(feasible);fxs.append(fx);fns.append(fn)
    valid_contact=bool(all(valid) and min(fns)>-1e-5 and
                       max(np.abs(fxs)-p.mu*np.array(fns))<1e-5)
    if event_name=='takeoff' and res.y[5,-1]<=0.: event_name='invalid_takeoff'
    if event_name=='takeoff' and not valid_contact: event_name='invalid_contact'
    return dict(phase='stance',t=ts,y=ys,u=np.array(taus),
                state=res.y[:,-1].copy(),end=float(res.t[-1]),
                event=event_name,foot_x=float(foot_x),
                normal_min=float(min(fns)),
                max_slip=float(max(np.abs(fxs)-p.mu*np.array(fns))),
                bottom_t=res.t_events[1].tolist(),
                minimum_length=float(min(res.y[2])))


def simulate_flight(y0,p:Robot,c:Control,start_time=0.,accurate=False,flight_target=None):
    y0=np.asarray(y0,dtype=float)
    def touchdown(t,y):return y[1]-p.l0*np.cos(y[4])
    touchdown.terminal=True;touchdown.direction=-1
    res=solve_ivp(lambda t,y:flight_rhs(t,y,p,c,flight_target),
                  (start_time,start_time+p.max_flight),y0,events=touchdown,
                  rtol=1e-9 if accurate else 2e-6,
                  atol=1e-10 if accurate else 1e-8,
                  max_step=0.0015 if accurate else 0.012)
    event_name='touchdown' if len(res.t_events[0]) else 'no_touchdown'
    ts=np.linspace(res.t[0],res.t[-1],max(75,int((res.t[-1]-res.t[0])/(0.003 if accurate else 0.012))+1))
    ys=np.array([np.interp(ts,res.t,res.y[i]) for i in range(8)]).T
    us=np.array([flight_torque(y,p,c,flight_target) for y in ys])
    return dict(phase='flight',t=ts,y=ys,u=us,
                state=res.y[:,-1].copy(),end=float(res.t[-1]),event=event_name,
                flight_reference=float(c.flight_target if flight_target is None else flight_target))


def plan_tilt_one_hop(y0,foot_x,goal,p:Robot,c:Control,step_limit=0.17,
                      alpha_grid=9):
    """Event-triggered, one-hop rollout search over an attitude setpoint.

    This is not a high-rate IPOPT MPC! Replans once after every landing.
    Uses exactly the same hybrid simulation as the plant (perfect model).
    """
    waypoint=foot_x+np.clip(goal-foot_x,-step_limit,step_limit)
    feasible=[]
    for ref in np.linspace(-c.max_tilt_reference,c.max_tilt_reference,alpha_grid):
        ss=simulate_stance(y0,foot_x,ref,p,c)
        if ss['event']!='takeoff':continue
        ff=simulate_flight(to_flight(ss['state'],foot_x,p),p,c,ss['end'])
        if ff['event']!='touchdown':continue
        xfeet,_=foot_position_flight(ff['state'],p)
        # Favor accurate landing, then small motor effort/rotation.
        cost=(xfeet-waypoint)**2+0.001*(ref**2)+0.003*(ff['state'][4]**2)
        feasible.append((cost,float(ref),float(xfeet)))
    if not feasible:return None
    best=min(feasible)
    return dict(alpha_ref=best[1],predicted_landing=best[2],
                waypoint=float(waypoint),n_feasible=len(feasible))


def run_hops(p:Robot, c:Control, target=0.45, n_hops=5, compression=0.12,
             arrival_tolerance=0.025, step_limit=0.17, alpha_grid=9,
             verbose=True):
    if n_hops<1:raise ValueError('n_hops must be >=1')
    y=initial_state(p,compression)
    foot_x=0.;t=0.; segments=[]; landings=[];hop_info=[]
    status='maximum hop count reached'
    for i in range(n_hops):
        plan=plan_tilt_one_hop(y,foot_x,target,p,c,
                               step_limit=step_limit,alpha_grid=alpha_grid)
        if plan is None:
            status=f'no feasible planned takeoff at hop {i+1}'
            break
        ss=simulate_stance(y,foot_x,plan['alpha_ref'],p,c,t,accurate=True)
        segments.append(ss)
        if ss['event']!='takeoff':
            status=f"stance stopped: {ss['event']} at hop {i+1}"
            break
        ff=simulate_flight(to_flight(ss['state'],foot_x,p),p,c,ss['end'],accurate=True)
        segments.append(ff)
        if ff['event']!='touchdown':
            status=f"flight stopped: {ff['event']} at hop {i+1}"
            break
        foot_next,y_next,loss=impact_reset(ff['state'],p)
        jx,jz,impact_friction_shortfall=impact_impulse_diagnostics(
            ff['state'],y_next,foot_next,p)
        # The idealized sticking reset does not enforce the impact friction cone.
        info=dict(hop=i+1,foot_start=foot_x,foot_end=foot_next,
                  alpha_reference=plan['alpha_ref'],waypoint=plan['waypoint'],
                  predicted_landing=plan['predicted_landing'],
                  touchdown_time=ff['end'],impact_loss=loss,
                  spring_speed_after_impact=y_next[5],
                  takeoff_time=ss['end'],min_stance_normal=ss['normal_min'],
                  max_friction_margin=ss['max_slip'],
                  spring_min_length=ss['minimum_length'],
                  impact_impulse_x=jx,impact_impulse_z=jz,
                  impact_friction_shortfall=impact_friction_shortfall)
        landings.append((float(ff['end']),foot_next))
        hop_info.append(info)
        if verbose:
            print(f"Hop {i+1}: foot {foot_x:+.3f} -> {foot_next:+.3f} m; "
                  f"next waypoint {plan['waypoint']:+.3f}; "
                  f"tilt ref {np.degrees(plan['alpha_ref']):+.1f} deg; "
                  f"impact loss {loss:.3f} J")
        y=y_next; foot_x=foot_next; t=ff['end']
        if abs(foot_x-target)<arrival_tolerance:
            status='target reached within tolerance'
            break
    return dict(segments=segments,landings=landings,hops=hop_info,
                final_foot_x=foot_x,target=target,status=status,
                last_stance_y=y,total_time=t)
