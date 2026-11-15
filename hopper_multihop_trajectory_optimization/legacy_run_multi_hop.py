"""Command line runner, plots, and animated visualization for the 2D hopper."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from hybrid_hopper import Robot, Control, run_hops, com_from_stance


def build_samples(p,segments):
    t=[];phase=[];foot=[];cx=[];cz=[];alpha=[];length=[];torque=[];energy=[]
    for seg in segments:
        tt=seg['t']; yy=seg['y']; u=seg['u']
        for k,(tk,yk) in enumerate(zip(tt,yy)):
            if seg['phase']=='stance':
                com=com_from_stance(yk,seg['foot_x']); fx=seg['foot_x'];leg=yk[2]
                en=0.5*p.m*(com[2]**2+com[3]**2)+0.5*p.Ib*yk[3]**2+0.5*p.Iw*yk[4]**2+p.m*p.g*com[1]+0.5*p.k*max(0,p.l0-leg)**2
                ang=yk[0]
            else:
                com=yk[:4];leg=p.l0;ang=yk[4]; fx=yk[0]-leg*np.sin(ang)
                en=0.5*p.m*(yk[2]**2+yk[3]**2)+0.5*p.Ib*yk[6]**2+0.5*p.Iw*yk[7]**2+p.m*p.g*yk[1]
            if t and tk <= t[-1]:continue
            t.append(tk);phase.append(seg['phase']);foot.append(fx);cx.append(com[0]);cz.append(com[1]);alpha.append(ang);length.append(leg);torque.append(u[k]);energy.append(en)
    return {name:np.array(vals) for name,vals in locals().items() if name in ('t','phase','foot','cx','cz','alpha','length','torque','energy')}


def save_outputs(p,report,output,make_gif=True):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    sm=build_samples(p,report['segments'])
    if not len(sm['t']):return
    fig,ax=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    ax[0,0].plot(sm['cx'],sm['cz'],lw=2)
    for _,xx in report['landings']:
        ax[0,0].scatter(xx,0.,s=25,marker='x',color='tab:orange')
    ax[0,0].axvline(report['target'],ls='--',color='tab:green',label='Goal foot x')
    ax[0,0].set(xlabel='COM x [m]',ylabel='COM z [m]',title='Center-of-mass trajectory')
    ax[0,0].grid(True);ax[0,0].legend(fontsize=8)
    ax[0,1].plot(sm['t'],sm['length'],label='Leg length')
    ax[0,1].axhline(p.l0,ls='--',color='gray',label='Spring free length')
    ax[0,1].set(xlabel='Time [s]',ylabel='Length [m]',title='Landing compression and rebounds')
    ax[0,1].grid(True);ax[0,1].legend(fontsize=8)
    ax[1,0].plot(sm['t'],np.rad2deg(sm['alpha']),label='Body angle')
    ax[1,0].plot(sm['t'],sm['torque'],alpha=.7,label='Motor torque [Nm]')
    ax[1,0].set(xlabel='Time [s]',title='Body attitude (deg) and motor torque (Nm)')
    ax[1,0].grid(True);ax[1,0].legend(fontsize=8)
    ax[1,1].plot(sm['t'],sm['energy'],label='Mechanical energy')
    for lt,_ in report['landings']:ax[1,1].axvline(lt,color='tab:orange',alpha=.2)
    ax[1,1].set(xlabel='Time [s]',ylabel='Energy [J]',title='Mechanical energy (motor can add energy)')
    ax[1,1].grid(True)
    fig.savefig(output/'multi_hop_plots.png',dpi=165);plt.close(fig)

    import csv
    with (output/'trajectory.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['t_s','phase','com_x_m','com_z_m','foot_x_m','angle_rad','length_m','torque_Nm','energy_J'])
        for i in range(len(sm['t'])):
            w.writerow([sm['t'][i],sm['phase'][i],sm['cx'][i],sm['cz'][i],sm['foot'][i],sm['alpha'][i],sm['length'][i],sm['torque'][i],sm['energy'][i]])
    with (output/'summary.txt').open('w') as f:
        f.write(f"Goal x: {report['target']:.4f} m\nFinal foot x: {report['final_foot_x']:.4f} m\nStatus: {report['status']}\n")
        for r in report['hops']:
            f.write(str(r)+'\n')

    if not make_gif:return
    times=np.linspace(sm['t'][0],sm['t'][-1],min(340,max(100,int(sm['t'][-1]*50))))
    cx=np.interp(times,sm['t'],sm['cx']); cz=np.interp(times,sm['t'],sm['cz'])
    alpha=np.interp(times,sm['t'],sm['alpha']);l=np.interp(times,sm['t'],sm['length'])
    fx=cx-l*np.sin(alpha);fz=cz-l*np.cos(alpha)
    fig,ax=plt.subplots(figsize=(9,4.8))
    ax.axhline(0,color='black',lw=1.5)
    ax.axvline(report['target'],color='tab:green',ls='--',label='Goal')
    ax.plot(sm['cx'],sm['cz'],color='0.8',lw=1,label='COM trace')
    rod,=ax.plot([],[],color='tab:blue',lw=4)
    body,=ax.plot([],[],'o',color='tab:blue',ms=16)
    wheel,=ax.plot([],[],'o',color='white',mec='tab:blue',ms=7)
    toe,=ax.plot([],[],'o',color='tab:orange',ms=5)
    caption=ax.text(0.02,.96,'',transform=ax.transAxes,va='top')
    xmin=min(-.2,float(np.min(fx))-.2,report['target']-.2)
    xmax=max(.4,float(np.max(fx))+.2,report['target']+.2)
    ax.set(xlim=(xmin,xmax),ylim=(-.06,max(.8,float(np.max(cz))+.2)),xlabel='x [m]',ylabel='z [m]',title='2D passive spring hopper — multiple contacts')
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='upper right',fontsize=8)
    def frame(k):
        rod.set_data([fx[k],cx[k]],[fz[k],cz[k]])
        body.set_data([cx[k]],[cz[k]])
        wheel.set_data([cx[k]],[cz[k]])
        toe.set_data([fx[k]],[fz[k]])
        idx=np.searchsorted(sm['t'],times[k],side='right')-1
        caption.set_text(f't={times[k]:.2f} s  |  {sm["phase"][max(0,idx)].upper()}')
        return rod,body,wheel,toe,caption
    ani=FuncAnimation(fig,frame,frames=len(times),interval=45,blit=True)
    ani.save(output/'multi_hop.gif',writer=PillowWriter(fps=22));plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description='2D passive-spring repeated hopping toward x target')
    parser.add_argument('--target',type=float,default=.45,help='Goal foot position in meters')
    parser.add_argument('--hops',type=int,default=5)
    parser.add_argument('--compression',type=float,default=.12)
    parser.add_argument('--damping',type=float,default=1.0)
    parser.add_argument('--output',default='results_multihop')
    parser.add_argument('--no-gif',action='store_true')
    parser.add_argument('--step-limit',type=float,default=.17)
    parser.add_argument('--alpha-grid',type=int,default=9)
    args=parser.parse_args()
    p=Robot(c=args.damping)
    c=Control()
    result=run_hops(p,c,target=args.target,n_hops=args.hops,
                    compression=args.compression,step_limit=args.step_limit,
                    alpha_grid=args.alpha_grid)
    print(f"Result: {result['status']}; final x={result['final_foot_x']:+.3f} m; "
          f"target={args.target:+.3f} m; hops={len(result['hops'])}")
    violations=[h['impact_friction_shortfall'] for h in result['hops']]
    if violations and max(violations)>1e-6:
        print('IDEALIZATION WARNING: the instantaneous pinned-foot impact ' 
              'requires more tangential impulse than a dry Coulomb '
              'friction cone supplies. This is a massless point-foot '
              'planning model; use a compliant/sliding-contact model '
              'before interpreting landing as hardware-feasible.')
    save_outputs(p,result,args.output,make_gif=not args.no_gif)
    print(f'Results in {Path(args.output).resolve()}')

if __name__=='__main__':main()
