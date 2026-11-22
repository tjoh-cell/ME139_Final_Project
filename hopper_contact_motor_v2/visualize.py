"""Plots and animated geometric 2D simulation (no external physics engine)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Circle
from physics import foot_kinematics


def _arrays(result):
    if not result['history']:
        raise ValueError('No successful trajectory samples to render')
    t=np.array([a[0] for a in result['history']],dtype=float)
    y=np.array([a[1] for a in result['history']],dtype=float)
    return t,y


def save_plots(result,path):
    t,y=_arrays(result)
    p=result['robot']
    fpos=np.array([foot_kinematics(z[:5])[0] for z in y])
    energy=np.array([__import__('physics').mechanical_energy(z,p) for z in y])
    fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    axs[0,0].plot(y[:,0],y[:,1],lw=1.2,label='COM')
    axs[0,0].scatter(fpos[:,0],fpos[:,1],s=1,alpha=.13,label='foot')
    axs[0,0].axvline(result['target'],ls='--',c='tab:green',label='target')
    axs[0,0].set(xlabel='x [m]',ylabel='z [m]',title='Center of mass / foot')
    axs[0,0].legend(loc='best',fontsize=8)
    axs[0,1].plot(t,np.degrees(y[:,2]),lw=1)
    axs[0,1].axhline(30,ls=':',c='red');axs[0,1].axhline(-30,ls=':',c='red')
    axs[0,1].set(xlabel='time [s]',ylabel='pitch [deg]',title='Body angle')
    axs[1,0].plot(t,p.leg_free_length-y[:,3])
    axs[1,0].set(xlabel='time [s]',ylabel='compression [m]',title='Passive spring motion')
    axs[1,1].plot(t,energy)
    axs[1,1].set(xlabel='time [s]',ylabel='mechanical energy [J]',title='Mechanical energy (motor can add energy)')
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.suptitle(f"Planar hopper: {result['status']} — final x={result['final_x']:.3f} m",fontsize=13)
    fig.savefig(path,dpi=140)
    plt.close(fig)


def _body_points(y,p):
    x,z,th=y[0],y[1],y[2]
    c=np.cos(th);s=np.sin(th)
    local=np.array([[-p.body_width/2,-p.body_height/2],
                    [p.body_width/2,-p.body_height/2],
                    [p.body_width/2,p.body_height/2],
                    [-p.body_width/2,p.body_height/2]])
    # theta positive pitches clockwise from upright
    rot=np.array([[c,s],[-s,c]])
    return local@rot.T+np.array([x,z])


def make_gif(result,path,max_frames=140,fps=18):
    t,y=_arrays(result)
    p=result['robot']
    inds=np.unique(np.linspace(0,len(t)-1,min(max_frames,len(t)),dtype=int))
    pts=y[inds]
    fpss=np.array([foot_kinematics(v[:5])[0] for v in pts])
    all_x=np.r_[pts[:,0],fpss[:,0],result['target']]
    xmin=min(-.5,float(np.min(all_x))-.5);xmax=max(.5,float(np.max(all_x))+.5)
    zmax=max(1.5,float(np.max(pts[:,1]))+0.5)
    # fit in a readable frame despite large vertical jumping
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.set_xlim(xmin,xmax);ax.set_ylim(-.2,zmax)
    ax.set_aspect('auto');ax.grid(alpha=.15)
    ax.set_xlabel('horizontal position [m]');ax.set_ylabel('height [m]')
    ax.axhline(0,c='k',lw=1.6)
    ax.axvline(result['target'],ls='--',c='tab:green',label='goal')
    body=Polygon(_body_points(pts[0],p),closed=True,fc='#8CB8D9',ec='#325C7A',lw=1.4)
    foot=Circle(fpss[0],p.foot_radius,fc='#DAA65B',ec='#8C5927')
    wheel=Circle((pts[0,0],pts[0,1]),min(.07,p.body_width*.35),fc='#E5F1F7',ec='#325C7A')
    springline,=ax.plot([],[],c='#597A9A',lw=2.5)
    trajline,=ax.plot([],[],c='.65',lw=.7,alpha=.8)
    timestamp=ax.text(.015,.98,'',va='top',transform=ax.transAxes)
    ax.add_patch(body);ax.add_patch(foot);ax.add_patch(wheel)
    ax.legend(loc='upper right')
    def update(k):
        v=pts[k]; f=fpss[k]
        body.set_xy(_body_points(v,p));foot.center=f;wheel.center=(v[0],v[1])
        # short zigzag spring, anchored at COM and foot, in world coordinates
        d=np.array([v[0]-f[0],v[1]-f[1]]); length=np.linalg.norm(d)
        if length<1e-9:length=1
        tangent=np.array([-d[1],d[0]])/length
        n=11;s=np.linspace(.08,.92,n)
        amp=np.array([0 if i in (0,n-1) else (-1)**i*.018 for i in range(n)])
        ptspring=f[None,:]+s[:,None]*d[None,:]+amp[:,None]*tangent[None,:]
        springline.set_data(ptspring[:,0],ptspring[:,1])
        trajline.set_data(pts[:k+1,0],pts[:k+1,1])
        # The final outcome must not appear at t=0: show live phase until
        # the final frame, then display the termination reason.
        phase=result['history'][inds[k]][2]
        label=result['status'] if k==len(pts)-1 else phase
        timestamp.set_text(f't = {t[inds[k]]:.2f} s  |  {label}')
        return body,foot,wheel,springline,trajline,timestamp
    anim=FuncAnimation(fig,update,frames=len(pts),interval=1000/fps,blit=False)
    anim.save(path,writer=PillowWriter(fps=fps))
    plt.close(fig)


def save_actuator_contact_plots(result,path):
    data=np.asarray(result['motor_history'],dtype=float)
    ct=np.asarray(result['contact_history'],dtype=float)
    if data.size==0:return
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
    axs[0].plot(data[:,0],data[:,1],label='wheel torque [Nm]')
    axs[0].plot(data[:,0],data[:,2],label='voltage [V]',alpha=.65)
    axs[0].plot(data[:,0],data[:,3],label='current [A]',alpha=.65)
    axs[0].legend(loc='upper right',fontsize=9)
    axs[0].set(ylabel='torque / voltage / current',title='Motor and contact diagnostics')
    axs[1].plot(data[:,0],data[:,4],label='electrical power [W]')
    axs[1].plot(data[:,0],data[:,5],label='mechanical wheel power [W]',alpha=.8)
    axs[1].plot(data[:,0],data[:,6],label='relative wheel speed [rad/s]',alpha=.8)
    axs[1].legend(loc='upper right',fontsize=9)
    axs[1].set(ylabel='power / speed')
    if ct.size:
        axs[2].plot(ct[:,0],ct[:,1],label='normal impulse [N·s]')
        axs[2].plot(ct[:,0],ct[:,2],label='friction impulse [N·s]')
        axs[2].plot(ct[:,0],ct[:,3],label='slip speed [m/s]')
        axs[2].legend(loc='upper right',fontsize=9)
    axs[2].set(xlabel='time [s]',ylabel='contact impulse / speed')
    for ax in axs:ax.grid(alpha=.18)
    fig.savefig(path,dpi=130)
    plt.close(fig)
