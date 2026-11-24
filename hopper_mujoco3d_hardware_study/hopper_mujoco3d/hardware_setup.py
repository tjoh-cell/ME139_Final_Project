"""Persistent Tk setup panel; each run owns an independent MuJoCo viewer process."""
import json
import multiprocessing as mp
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from hardware_parameters import SECTIONS, field_values, make_settings


def run_viewer(settings, speed, live, channel):
    from simulator import HopperSimulation
    from view import show_simulation

    def report(sim):
        toe = sim.landings[-1]['foot_x'] if sim.landings else None
        channel.send(('result', (sim.status, toe, sim.hops, sim.max_height, sim.memory.impulse_used,
                                 sim.max_motor_current, sim.max_motor_voltage, sim.max_wheel_rate,
                                 sim.max_wheel_rate > sim.a.wheel_speed_warning)))

    try:
        sim = HopperSimulation(*settings)
        show_simulation(sim, speed, live, lambda msg: channel.send(('progress', msg)), report)
    except Exception as exc:
        channel.send(('error', f'{type(exc).__name__}: {exc}'))
    finally:
        channel.close()


class HardwareSetup:
    def __init__(self, root, settings, speed, live):
        self.root, self.base = root, settings
        self.process = self.reader = None
        self.run_id = 0
        self.last_event = ''
        self.run_goal = ''
        self.completed = []
        self.defaults = field_values(settings, speed)
        self.inputs = {k: tk.StringVar(root, value=v) for k, v in self.defaults.items()}
        self.planner = tk.StringVar(root, value=settings[-1].planner)
        self.thrust = tk.BooleanVar(root, value=not settings[-1].no_thrusters)
        self.live = tk.BooleanVar(root, value=live)
        self.status = tk.StringVar(root, value='Choose a goal and click Run simulation.')
        self.design = tk.StringVar(root)
        root.title('Twin-pogo hopper | Hardware setup')
        root.geometry('960x760')
        panel = ttk.Frame(root, padding=14)
        panel.pack(fill='both', expand=True)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        bar = ttk.Frame(panel)
        bar.grid(row=0, column=0, sticky='ew', pady=(0,10))
        ttk.Button(bar, text='Run simulation', command=self.start).pack(side='left')
        self.stop_button = ttk.Button(bar, text='Stop run', command=self.stop, state='disabled')
        self.stop_button.pack(side='left', padx=6)
        ttk.Button(bar, text='Save settings…', command=self.save).pack(side='right')
        ttk.Button(bar, text='Load settings…', command=self.load).pack(side='right', padx=6)
        tabs = ttk.Notebook(panel)
        tabs.grid(row=1, column=0, sticky='nsew')
        for section, params in SECTIONS.items():
            tab = ttk.Frame(tabs)
            tabs.add(tab, text=section)
            canvas = tk.Canvas(tab, highlightthickness=0)
            scroll = ttk.Scrollbar(tab, command=canvas.yview)
            scroll.pack(side='right', fill='y')
            canvas.pack(side='left', fill='both', expand=True)
            canvas.configure(yscrollcommand=scroll.set)
            form = ttk.Frame(canvas, padding=12)
            window = canvas.create_window(0, 0, window=form, anchor='nw')
            form.bind('<Configure>', lambda _, c=canvas: c.configure(scrollregion=c.bbox('all')))
            canvas.bind('<Configure>', lambda e, c=canvas, w=window: c.itemconfigure(w, width=e.width))
            for row, (key, label, unit, hint) in enumerate(params):
                ttk.Label(form, text=label).grid(row=row, column=0, sticky='w', pady=4)
                entry = ttk.Entry(form, textvariable=self.inputs[key], width=13)
                entry.grid(row=row, column=1, padx=(12,5))
                ttk.Label(form, text=unit).grid(row=row, column=2, sticky='w')
                ttk.Label(form, text=hint, wraplength=410).grid(row=row, column=3, sticky='w', padx=14)
            if section == 'Mission':
                row = len(params)
                ttk.Label(form, text='Controller').grid(row=row, column=0, sticky='w', pady=7)
                ttk.Combobox(form, values=('feedback','predictive'), state='readonly',
                             textvariable=self.planner, width=12).grid(row=row,column=1)
                ttk.Checkbutton(form,text='Enable thrusters',variable=self.thrust).grid(row=row,column=3,sticky='w')
                ttk.Checkbutton(form,text='Plan live in predictive mode (pauses at touchdown)',
                                variable=self.live).grid(row=row+1,column=0,columnspan=4,sticky='w')
        ttk.Label(panel,textvariable=self.design,wraplength=880).grid(row=2,column=0,sticky='w',pady=(10,4))
        ttk.Label(panel,textvariable=self.status,wraplength=880).grid(row=3,column=0,sticky='w',pady=(0,8))
        history = ttk.LabelFrame(panel,text='Completed runs this session',padding=5)
        history.grid(row=4,column=0,sticky='ew')
        cols = ('Run','Goal m','Outcome','Toe x m','Hops','Peak z m','Impulse N·s','Peak A','Peak V','Wheel rad/s')
        self.results = ttk.Treeview(history,columns=cols,show='headings',height=4)
        self.results.pack(fill='x')
        ttk.Button(history,text='Export completed runs…',command=self.export_runs).pack(anchor='e',pady=(4,0))
        for col,width in zip(cols,(35,55,154,68,42,68,82,56,56,90)):
            self.results.heading(col,text=col)
            self.results.column(col,width=width,minwidth=30,anchor='center')
        for k in ('body_mass','wheel_mass','foot_mass_total','wheel_radius','wheel_inertia',
                  'spring_k_total','initial_compression','max_thrust_each'):
            self.inputs[k].trace_add('write', self.design_summary)
        self.design_summary()
        root.protocol('WM_DELETE_WINDOW', self.close)
        self.timer = root.after(150, self.poll)

    def design_summary(self, *_):
        try:
            def n(key):
                return float(self.inputs[key].get())
            mass = n('body_mass') + n('wheel_mass') + n('foot_mass_total')
            energy = 0.5*n('spring_k_total')*n('initial_compression')**2
            bound = n('wheel_mass')*n('wheel_radius')**2
            line = f'Modeled mass {mass:.3g} kg   |   Initial spring energy {energy:.3g} J   |   Peak vertical thrust / weight {2*n("max_thrust_each")/(mass*9.81):.2f}'
            if n('wheel_inertia') > bound:
                line += f'\nRotor check: spin inertia exceeds m·r² ({bound:.4g} kg·m²). Check the CAD values.'
            self.design.set(line)
        except (ValueError, ZeroDivisionError, OverflowError):
            self.design.set('Complete the hardware values to see the design calculations.')

    def profile(self):
        return dict(version=1, values={k:v.get() for k,v in self.inputs.items()},
                    planner=self.planner.get(), thrust_enabled=self.thrust.get(), live=self.live.get())

    def validate(self, profile):
        if not isinstance(profile,dict) or profile.get('version') != 1:
            raise ValueError('Choose a version 1 hopper settings file.')
        if not isinstance(profile.get('values'),dict) or set(profile['values']) != set(self.inputs):
            raise ValueError('This file does not contain the expected hardware settings.')
        for key in ('thrust_enabled','live'):
            if type(profile.get(key)) is not bool:
                raise ValueError(f'Missing or invalid setting: {key}')
        return make_settings(self.base, profile['values'], profile.get('planner'), profile['thrust_enabled'])

    def start(self):
        profile = self.profile()
        try:
            settings, speed = self.validate(profile)
        except ValueError as exc:
            messagebox.showerror('Check settings', str(exc), parent=self.root)
            return
        self.stop()
        self.run_id += 1
        self.run_goal = f'{settings[-1].goal_x:g}'
        self.active_profile = profile
        ctx = mp.get_context('spawn')
        self.reader, writer = ctx.Pipe(duplex=False)
        self.process = ctx.Process(target=run_viewer,args=(settings,speed,self.live.get(),writer),daemon=True)
        self.process.start()
        writer.close()
        self.last_event = ''
        self.status.set(f'Opening run {self.run_id} for {self.run_goal} m…')
        self.stop_button.configure(state='normal')

    def save(self):
        profile = self.profile()
        try:
            self.validate(profile)
            name = filedialog.asksaveasfilename(parent=self.root,defaultextension='.json',
                                                filetypes=[('Hopper settings','*.json')])
            if name:
                Path(name).write_text(json.dumps(profile,indent=2),encoding='utf-8')
                self.status.set(f'Saved {Path(name).name}.')
        except (OSError, ValueError) as exc:
            messagebox.showerror('Save settings',str(exc),parent=self.root)

    def export_runs(self):
        if not self.completed:
            self.status.set('Complete a mission to export its settings and measurements.')
            return
        name = filedialog.asksaveasfilename(parent=self.root,defaultextension='.json',
                                            filetypes=[('Run comparison','*.json')])
        if name:
            try:
                Path(name).write_text(json.dumps(self.completed,indent=2),encoding='utf-8')
                self.status.set(f'Exported {len(self.completed)} completed runs.')
            except OSError as exc:
                messagebox.showerror('Export runs',str(exc),parent=self.root)

    def load(self):
        name = filedialog.askopenfilename(parent=self.root,filetypes=[('Hopper settings','*.json')])
        if not name:
            return
        try:
            profile = json.loads(Path(name).read_text(encoding='utf-8-sig'))
            self.validate(profile)
        except (ValueError, OSError) as exc:
            messagebox.showerror('Load settings',str(exc),parent=self.root)
            return
        for key, value in profile['values'].items():
            self.inputs[key].set(value)
        self.planner.set(profile['planner'])
        self.thrust.set(profile['thrust_enabled'])
        self.live.set(profile['live'])
        self.status.set(f'Loaded {Path(name).name}. Click Run simulation to apply.')

    def stop(self):
        if self.process is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join()
            self.process.close()
            self.process = None
            self.reader.close()
            self.reader = None
            self.status.set('Run closed. Edit settings and run again.')
        self.stop_button.configure(state='disabled')

    def poll(self):
        if self.reader is not None:
            try:
                while self.reader.poll():
                    event, payload = self.reader.recv()
                    if event == 'result':
                        status, x, hops, z, impulse, amps, volts, wheel, warning = payload
                        self.completed.append(dict(run=self.run_id, settings=self.active_profile,
                            status=status, last_touchdown_x_m=x, hops=hops, peak_height_m=z,
                            impulse_Ns=impulse, peak_current_A=amps, peak_voltage_V=volts,
                            peak_wheel_rad_s=wheel, wheel_speed_warning=warning))
                        row = (self.run_id,self.run_goal,status.replace('_',' '),
                               '—' if x is None else f'{x:.3f}',hops,f'{z:.2f}',f'{impulse:.2f}',
                               f'{amps:.1f}',f'{volts:.1f}',f'{wheel:.1f}' + (' ⚠' if warning else ''))
                        self.results.see(self.results.insert('','end',values=row))
                    self.last_event = event
                    self.status.set(f'Run {self.run_id}: {payload if event != "result" else status.replace("_"," ")}')
            except (EOFError, OSError):
                pass
            if not self.process.is_alive():
                last = self.status.get()
                failed = self.process.exitcode
                self.stop()
                if self.last_event == 'error':
                    self.status.set(last)
                elif failed:
                    self.status.set(f'Viewer exited with code {failed}.')
        self.timer = self.root.after(150,self.poll)

    def close(self):
        self.root.after_cancel(self.timer)
        self.stop()
        self.root.destroy()


def open_setup(settings, speed, live):
    root = tk.Tk()
    HardwareSetup(root,settings,speed,live)
    root.mainloop()
