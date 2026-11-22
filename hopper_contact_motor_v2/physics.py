"""Planar reaction-wheel spring hopper with finite foot mass, body collision,
friction-cone impulse contacts and voltage/current-limited DC motor.

Coordinates q=[COM x, COM z, body pitch, leg length, ABSOLUTE wheel angle].
The leg and its spring are massless; the finite point-foot mass regularizes
impact and permits sliding. The body is a rotating rectangle.

The contact solver is a semi-implicit, velocity-level, projected Gauss-Seidel
approximation. It is not continuous-time exact Coulomb complementarity.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np

X,Z,TH,L,PH=range(5)

@dataclass(frozen=True)
class Robot:
    body_mass: float = 1.0
    foot_mass: float = 0.035
    body_inertia: float = 0.015
    wheel_inertia: float = 0.003
    body_width: float = 0.18
    body_height: float = 0.22
    foot_radius: float = 0.012
    leg_free_length: float = 1.0
    leg_min_length: float = 0.18
    spring_k: float = 450.0
    spring_c: float = 1.0
    gravity: float = 9.81
    friction_mu: float = 0.85
    # DC motor -> geared flywheel (example powertrain, NOT hardware selection)
    supply_voltage: float = 48.0
    armature_resistance: float = 0.16
    torque_constant: float = 0.050  # Nm/A
    back_emf_constant: float = 0.050 # V*s/rad
    gear_ratio: float = 5.0
    gear_efficiency: float = 0.90
    current_limit: float = 50.0
    relative_wheel_speed_warning: float = 190.0

    def validate(self):
        if min(self.body_mass,self.foot_mass,self.body_inertia,self.wheel_inertia,
               self.body_width,self.body_height,self.leg_free_length,
               self.spring_k,self.armature_resistance,self.gear_ratio,
               self.torque_constant,self.back_emf_constant,
               self.supply_voltage,self.current_limit) <= 0:
            raise ValueError('Positive masses, dimensions, spring and motor parameters required')
        if not (0 < self.leg_min_length < self.leg_free_length):
            raise ValueError('leg_min_length must be between 0 and leg_free_length')
        if not (0 < self.gear_efficiency <= 1) or self.friction_mu < 0:
            raise ValueError('Invalid friction or gearbox efficiency')

@dataclass(frozen=True)
class Controller:
    stance_kp: float = 20.0
    stance_kd: float = 0.65
    flight_kp: float = 3.0
    flight_kd: float = 0.70
    max_stance_ref: float = math.radians(37)
    max_flight_ref: float = math.radians(25)
    # To reduce motor overspeed / mechanical power, feedback suppresses wheel speed
    wheel_rate_damping: float = 0.003

@dataclass(frozen=True)
class SimulationConfig:
    dt: float = 0.003
    max_stance_time: float = 1.5
    max_flight_time: float = 3.5
    min_flight_duration: float = 0.06
    min_flight_rise: float = 0.025
    max_touchdown_pitch: float = math.radians(30)
    max_touchdown_rate: float = 2.0
    contact_iterations: int = 18
    contact_margin: float = 0.0015
    contact_beta: float = 0.12
    allow_body_contact: bool = False
    record_stride: int = 1


def wrap_angle(v: float)->float:
    return math.atan2(math.sin(v), math.cos(v))


def initial_state(p: Robot, compression: float=0.25, pitch:float=0.) ->np.ndarray:
    p.validate()
    if compression <= 0 or compression > p.leg_free_length-p.leg_min_length:
        raise ValueError('Initial compression outside leg travel')
    l=p.leg_free_length-compression
    # contact plane at z=0; foot sphere lowest point at 0
    q=np.array([l*math.sin(pitch), l*math.cos(pitch)+p.foot_radius,
                pitch,l,0.],dtype=float)
    return np.r_[q,np.zeros(5)]


def foot_kinematics(q):
    x,z,theta,l,_=q
    s=math.sin(theta); c=math.cos(theta)
    pos=np.array([x-l*s,z-l*c])
    J=np.array([[1.,0.,-l*c,-s,0.],
                [0.,1.,l*s,-c,0.]])
    return pos,J


def body_corners(q,p:Robot):
    x,z,theta,_,_=q
    c=math.cos(theta);s=math.sin(theta)
    out=[]
    for dx in (-p.body_width/2,p.body_width/2):
        for dz in (-p.body_height/2,p.body_height/2):
            xp=x+c*dx+s*dz
            zp=z-s*dx+c*dz
            J=np.array([[1.,0.,-s*dx+c*dz,0.,0.],
                        [0.,1.,-c*dx-s*dz,0.,0.]])
            out.append((f'body_{dx:+.2f}_{dz:+.2f}',np.array([xp,zp]),J))
    return out


def mass_matrix(q,p:Robot):
    _,J=foot_kinematics(q)
    M=np.diag([p.body_mass,p.body_mass,p.body_inertia,0.,p.wheel_inertia])
    M+=p.foot_mass*(J.T@J)
    return M


def acceleration(q,v,tau,p:Robot,M=None):
    theta=float(q[TH]);l=float(q[L]);ad=float(v[TH]);ld=float(v[L])
    s=math.sin(theta);c=math.cos(theta)
    _,J=foot_kinematics(q)
    # Foot acceleration = J qdd + b(q,qdot)
    b=np.array([l*s*ad*ad-2*c*ad*ld,
                l*c*ad*ad+2*s*ad*ld])
    generalized=np.array([0.,-(p.body_mass+p.foot_mass)*p.gravity,
                          -p.foot_mass*p.gravity*l*s-tau,
                          p.foot_mass*p.gravity*c,
                          tau])
    # compression-only linear spring/dashpot, limited to l < l0
    compression=max(0.,p.leg_free_length-l)
    if compression>0:
        generalized[L]+=p.spring_k*compression-p.spring_c*ld
    generalized-=p.foot_mass*(J.T@b)
    if M is None: M=mass_matrix(q,p)
    return np.linalg.solve(M,generalized)


def motor_from_request(tau_req,qdot,p:Robot):
    """Ideal current-control drive with actual voltage saturation and copper loss.
    A positive motor torque accelerates absolute wheel angle and applies the
    opposite reaction on the body. The motor shaft speed is relative speed*N.
    The electrical dynamics L dI/dt are neglected.
    """
    w_rel=float(qdot[PH]-qdot[TH])
    w_motor=p.gear_ratio*w_rel
    # Ask for desired wheel torque; invert gearbox torque transfer
    i_request=tau_req/(p.gear_ratio*p.gear_efficiency*p.torque_constant)
    i_request=float(np.clip(i_request,-p.current_limit,p.current_limit))
    v_request=p.armature_resistance*i_request+p.back_emf_constant*w_motor
    voltage=float(np.clip(v_request,-p.supply_voltage,p.supply_voltage))
    current=float(np.clip((voltage-p.back_emf_constant*w_motor)/p.armature_resistance,
                          -p.current_limit,p.current_limit))
    # A drive can choose to coast (open circuit). If voltage saturation would
    # cause torque OPPOSITE the requested sign, coast instead of forcing an
    # uncontrolled regenerative/braking torque.
    if abs(tau_req)<1e-10 or (current*tau_req < 0):
        current=0.0
        voltage=0.0
    # Simplified symmetric gearbox efficiency in motoring and regeneration.
    wheel_torque=p.gear_ratio*p.gear_efficiency*p.torque_constant*current
    electrical_power=voltage*current
    wheel_power=wheel_torque*w_rel
    return dict(torque=float(wheel_torque),voltage=voltage,current=current,
                omega_motor=float(w_motor),omega_relative=w_rel,
                p_electrical=float(electrical_power),p_mechanical=float(wheel_power),
                saturation=abs(v_request)>p.supply_voltage+1e-8 or
                           abs(tau_req)>p.gear_ratio*p.gear_efficiency*p.torque_constant*p.current_limit)


def controller_request(state,stance_ref,flight_ref,contact_phase,p:Robot,c:Controller):
    q=state[:5];v=state[5:]
    ref=float(stance_ref if contact_phase else flight_ref)
    err=wrap_angle(q[TH]-ref)
    kp=c.stance_kp if contact_phase else c.flight_kp
    kd=c.stance_kd if contact_phase else c.flight_kd
    torque=kp*err+kd*v[TH]+c.wheel_rate_damping*(v[PH]-v[TH])
    return motor_from_request(torque,v,p)


def potential_energy(q,p):
    theta=q[TH]
    zfoot=q[Z]-q[L]*math.cos(theta)
    return ((p.body_mass*q[Z]+p.foot_mass*zfoot)*p.gravity +
            0.5*p.spring_k*max(p.leg_free_length-q[L],0.)**2)


def mechanical_energy(y,p):
    q=y[:5];v=y[5:]
    return float(0.5*v@mass_matrix(q,p)@v+potential_energy(q,p))


def contact_candidates(q_pred,p:Robot,cfg:SimulationConfig):
    foot,J=foot_kinematics(q_pred)
    contacts=[]
    gap=float(foot[1]-p.foot_radius)
    if gap<=cfg.contact_margin:
        contacts.append(dict(name='foot',gap=gap,jn=J[1].copy(),jt=J[0].copy(),
                             mu=p.friction_mu))
    for name,pos,jac in body_corners(q_pred,p):
        if pos[1]<=cfg.contact_margin:
            contacts.append(dict(name=name,gap=float(pos[1]),jn=jac[1].copy(),
                                 jt=jac[0].copy(),mu=p.friction_mu))
    if q_pred[L]>=p.leg_free_length-cfg.contact_margin:
        contacts.append(dict(name='leg_extension_stop',
                             gap=float(p.leg_free_length-q_pred[L]),
                             jn=np.array([0.,0.,0.,-1.,0.]),jt=None,mu=0.))
    if q_pred[L]<=p.leg_min_length+cfg.contact_margin:
        contacts.append(dict(name='leg_compression_stop',
                             gap=float(q_pred[L]-p.leg_min_length),
                             jn=np.array([0.,0.,0.,1.,0.]),jt=None,mu=0.))
    return contacts


def solve_impulses(v_free,M,contacts,cfg:SimulationConfig,dt:float):
    """Normal impulses nonnegative, tangent impulses bounded by mu*normal.

    Sequential PGS / Coulomb friction impulse approximation. Tangential slip
    is permitted when static sticking requires more than the friction cone.
    """
    v=v_free.copy(); data=[]
    for con in contacts:
        jn=con['jn']; jt=con['jt']
        nvec=np.linalg.solve(M,jn)
        tvec=np.linalg.solve(M,jt) if jt is not None else None
        wn=max(float(jn@nvec),1e-12)
        wt=max(float(jt@tvec),1e-12) if jt is not None else 0.
        # bias removes small finite-step interpenetration; bounded to prevent kicks
        target=min(0.6, cfg.contact_beta*max(0.,-con['gap'])/dt)
        data.append(dict(**con,nvec=nvec,tvec=tvec,wn=wn,wt=wt,
                         bias=target,pn=0.,pt=0.))
    for _ in range(cfg.contact_iterations):
        for item in data:
            dn=max(0.,item['pn']+(item['bias']-item['jn']@v)/item['wn'])-item['pn']
            if dn:
                item['pn']+=dn
                v+=item['nvec']*dn
            if item['jt'] is not None:
                max_t=item['mu']*item['pn']
                pt_new=float(np.clip(item['pt']-(item['jt']@v)/item['wt'],
                                     -max_t,max_t))
                dtan=pt_new-item['pt']
                if dtan:
                    item['pt']=pt_new
                    v+=item['tvec']*dtan
    return v,[dict(name=e['name'],normal_impulse=float(e['pn']),
                   tangent_impulse=float(e['pt']),mu=float(e['mu']),
                   gap_before=float(e['gap']),
                   tangential_speed=float(e['jt']@v) if e['jt'] is not None else 0.)
              for e in data]


def step(state,stance_ref,flight_ref,stance_phase,p:Robot,c:Controller,cfg:SimulationConfig):
    """Take one semi-implicit step; contacts and actuators resolved each step."""
    h=cfg.dt
    y=np.asarray(state,dtype=float);q=y[:5].copy();v=y[5:].copy()
    motor=controller_request(y,stance_ref,flight_ref,stance_phase,p,c)
    M=mass_matrix(q,p)
    a=acceleration(q,v,motor['torque'],p,M)
    free_v=v+h*a
    q_pred=q+h*free_v
    contacts=contact_candidates(q_pred,p,cfg)
    v_next,impulses=solve_impulses(free_v,M,contacts,cfg,h)
    q_next=q+h*v_next
    # Residual geometric correction for finite time step, diagnostics included.
    # Not an unreported energy-preserving operation.
    corrected=0.
    bottom=foot_kinematics(q_next)[0][1]-p.foot_radius
    body_min=min(float(item[1][1]) for item in body_corners(q_next,p))
    pen=max(0.,-min(bottom,body_min))
    if pen>0:
        q_next[Z]+=pen
        corrected=pen
    if q_next[L]>p.leg_free_length:
        q_next[L]=p.leg_free_length
        if v_next[L]>0:v_next[L]=0.
    if q_next[L]<p.leg_min_length:
        q_next[L]=p.leg_min_length
        if v_next[L]<0:v_next[L]=0.
    y_next=np.r_[q_next,v_next]
    if not np.all(np.isfinite(y_next)):
        raise FloatingPointError('Nonfinite hopper dynamics')
    return y_next, dict(motor=motor,impulses=impulses,
                        foot_contact=any(e['name']=='foot' and e['normal_impulse']>1e-8 for e in impulses),
                        body_contact=any(e['name'].startswith('body') and e['normal_impulse']>1e-8 for e in impulses),
                        foot_slip=any(e['name']=='foot' and abs(e['tangential_speed'])>0.03 and
                                      e['normal_impulse']>1e-8 for e in impulses),
                        positional_correction=corrected,
                        kinetic_impulse_energy_change=float(.5*v_next@M@v_next-.5*free_v@M@free_v),
                        mechanical_energy=mechanical_energy(y_next,p))
