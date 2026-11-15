"""Render a saved trajectory.csv without re-running the optimizer."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='results_optimized/trajectory.csv')
    ap.add_argument('--target',type=float,default=.45)
    ap.add_argument('--output',default=None)
    ap.add_argument('--frames',type=int,default=145)
    a=ap.parse_args()
    data=np.genfromtxt(a.csv,delimiter=',',names=True,dtype=None,encoding='utf-8')
    tt=data['t_s'];cx=data['com_x_m'];cz=data['com_z_m'];
    fx=data['foot_x_m'];alpha=data['angle_rad'];length=data['length_m']
    times=np.linspace(tt[0],tt[-1],a.frames)
    cxi=np.interp(times,tt,cx);czi=np.interp(times,tt,cz)
    ai=np.interp(times,tt,alpha);li=np.interp(times,tt,length)
    feetx=cxi-li*np.sin(ai);feetz=czi-li*np.cos(ai)
    fig,ax=plt.subplots(figsize=(9,5.4))
    ax.axhline(0.,c='black',lw=1.1)
    ax.axvline(a.target,color='tab:green',ls='--',label='Target')
    ax.plot(cx,cz,c='0.75',lw=1.1,label='COM trace')
    rod,=ax.plot([],[],lw=3.5,c='tab:blue')
    body,=ax.plot([],[],'o',ms=13,c='tab:blue')
    foot,=ax.plot([],[],'o',ms=5,c='tab:orange')
    txt=ax.text(.02,.95,'',transform=ax.transAxes,va='top')
    ax.set(xlim=(min(-.2,cx.min()-.5,feetx.min()-.5,a.target-.2),
                 max(.5,cx.max()+.5,feetx.max()+.5,a.target+.2)),
           ylim=(-.1,max(.8,cz.max()+.4)),
           xlabel='Horizontal position [m]',ylabel='Height [m]',
           title='Jointly optimized multi-hop trajectory')
    ax.set_aspect('equal',adjustable='box');ax.legend(loc='upper right')
    def frame(i):
        rod.set_data([feetx[i],cxi[i]],[feetz[i],czi[i]])
        body.set_data([cxi[i]],[czi[i]])
        foot.set_data([feetx[i]],[feetz[i]])
        idx=max(0,np.searchsorted(tt,times[i],side='right')-1)
        txt.set_text(f't = {times[i]:.2f}s | {data["phase"][idx].upper()}')
        return rod,body,foot,txt
    ani=FuncAnimation(fig,frame,frames=len(times),interval=50,blit=True)
    path=Path(a.output) if a.output else Path(a.csv).with_name('optimized_hops.gif')
    ani.save(path,writer=PillowWriter(fps=20))
    print(f'Saved {path.resolve()}')
    plt.close(fig)

if __name__=='__main__':main()
