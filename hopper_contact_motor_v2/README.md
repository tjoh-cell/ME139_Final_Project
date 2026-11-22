# Planar Hopper V2 — Finite-Body Contact + DC Motor

This is a **standalone** Python project. It does not overwrite or import the older `hopper_multihop_trajectory_optimization` directory.

It is a **reduced-order 2D engineering simulator**, intended for evaluating candidate mechanics and controllers, **not a validated rigid-body physics engine** or a validated digital twin. In particular, neither the friction coefficient nor the numerical motor values have been measured on hardware.

## Run

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_sim.py
python run.py --target 3 --compression 0.4 --lookahead 2 --maxfev 12 --gif
```

Outputs appear in `results_contact_motor/`:

- `summary.json`: hop-by-hop landing, power, slip and optimization reports
- `trajectory.csv`: complete 2D body/leg/wheel state history
- `motor.csv`: wheel torque, motor current/voltage, electrical/mechanical power and relative wheel speed
- `contacts.csv`: foot normal/tangential impulses and foot sliding speed
- `trajectory.png`: COM trajectory, attitude, compression, energy
- `motor_and_contact.png`: motor and contact time series
- `hopper.gif`: animation when `--gif` is passed

### More examples

```bash
# Short / backward targets
python run.py --target 1.0 --compression 0.4 --lookahead 2
python run.py --target -1.0 --compression 0.4 --lookahead 2

# Longer, more expensive trajectory planning
python run.py --target 5.0 --compression 0.4 --lookahead 3 --maxfev 30

# Experiment with contact / powertrain limits
python run.py --target 3.0 --compression 0.4 --mu 0.15 --voltage 24 --current-limit 20 --gear-ratio 4

# Stricter numerical integration (slower)
python run.py --target 3 --compression 0.4 --dt 0.002 --contact-iterations 30
```

**Do not assume a target is feasible:** The planner can return `no_feasible_plan`, `unsafe_actual_hop:body_collision`, `unsafe_actual_hop:unsafe_touchdown_pitch`, `stalled_no_progress` or `safety_hop_limit`. This is intentional. If you change time step, spring, inertia, contact, or powertrain parameters, the predicted optimal trajectory can change or fail. More candidate optimizations (`--maxfev` / `--lookahead`) may help local-search failures, but cannot make an impossible motor/contact trajectory physically feasible.

## Files

- `physics.py`: 5-DOF Lagrangian dynamics, actual rectangular body corners, finite point-foot contact, hard leg travel, PGS Coulomb impulses and DC motor limits
- `simulator.py`: takes the plant through stance -> flight -> touchdown and checks landing safety; calls *the same physics* as the planner
- `planner.py`: independently solves 1-, 2-, ... , `lookahead`-hop constrained nonlinear trajectory searches over paired attitude setpoints, replanning after touchdown
- `run.py`: CLI, data export, and plotting
- `visualize.py`: time-history plots and GIF animation
- `test_sim.py`: 7 unit/integration tests

## Model and assumptions

### Coordinates and energy

Generalized coordinates are

\[
q=[x_{COM},z_{COM},\alpha,l,\phi]^T
\]

with flywheel **absolute** angle \(\phi\). The relative flywheel speed at the shaft is \(\dot\phi-\dot\alpha\). The rigid-body geometric collision shape is a rectangle centered at the COM, with width and height in `Robot`. A **small finite foot mass** (default 0.035 kg) is included instead of a mathematically zero-mass foot: this permits a nonsingular impact mass matrix and dynamically meaningful slipping. The leg is massless.

The foot location is

\[
p_{foot}=(x-l\sin\alpha,\ z-l\cos\alpha).
\]

The mass matrix is assembled from the body inertia, flywheel inertia, and foot translational Jacobian; the spring compresses passively with stiffness `spring_k` and damping `spring_c`. A rigid leg-extension stop prevents the spring/leg from extending beyond its free length. A second stop enforces minimum leg length. The initial state **starts with the spring compressed**; a real prototype would need a latch/preload mechanism to establish this first-hop condition.

### Frictional contact and body impact

Every time step detects ground contact candidates for the foot and all four body corners. A **semi-implicit time-stepping projected Gauss–Seidel (PGS)** solver applies normal and tangential impulses. For each contact, it enforces

\[
J_n\ge0,\qquad |J_t|\le \mu J_n.
\]

The foot can therefore **slide** if it would require greater tangential impulse to stick; no fictitious infinite-friction reset is applied. Contact impulses are in newton-seconds and are logged to `contacts.csv`. The solver uses zero impact restitution, a finite number of iterations, and a small positional correction if a time step caused limited penetration.

A body corner contacting the ground is **resolved at impact and labeled a crash**, terminating that candidate. We do not simulate a prolonged tumbling/scraping recovery after a body collision.

The time-stepping solver is an **approximation to exact unilateral contact and Coulomb friction**: convergence depends on timestep, iteration count, and contact geometry. Higher-fidelity verification should use an established contact engine (e.g. MuJoCo/Drake) or dedicated complementarity optimization once the mechanical geometry is defined.

### Motor and battery

The motor drive uses a quasi-steady brushed-DC approximation (not a named real product):

\[
I=\operatorname{clip}\left(\frac{V-K_eN(\dot\phi-\dot\alpha)}{R},\,-I_{max},I_{max}\right),
\qquad \tau_{wheel}=\eta N K_t I.
\]

It requests voltage to track a PD torque command but saturates to `supply_voltage`, `current_limit` and the back-EMF torque-speed envelope. When a voltage-limited request would create torque in the opposite direction, it coasts. Values of `torque_constant`, `back_emf_constant`, `armature_resistance`, `gear_ratio`, `gear_efficiency`, and `wheel_inertia` are editable in `physics.py`; the CLI exposes some of these.

This captures key **steady-state** electrical limits, but **neglects motor inductance/current-loop bandwidth, temperature, thermal derating, battery internal resistance/sag, motor rotor inertia reflected through the gearbox, backlash, driver limits, and detailed regenerative efficiency**. These must be modeled before making final hardware selections.

Default 48 V, 50 A, 5:1, \(K_t=K_e=0.05\), \(R=0.16\,\Omega\), \(\eta=0.9\) produce an illustrative 11.25 N·m current-limited stall wheel torque. This is **not** a recommendation for a motor model.

### Controller and optimizer

Every candidate hop gets **two control references**: a stance body pitch and a flight body pitch. Reaction-wheel PD torque tracking and voltage/current saturation produce actual torque throughout simulation. The planner compares independent horizons \(N=1,\dots,H\). For each horizon, bounded Powell optimizes \(2N\) reference variables using the same physics as the execution simulator. Only candidates with a foot touchdown, sufficient flight duration/height, and body pitch/pitch-rate inside limits are eligible. Plans can include backward hops; the controller replans after each touchdown. A hop count limit is only a safety cap. Stop after target tolerance or user-defined goal crossings (up to 2 by default); residual energy need not go to zero.

This is **event-based nonlinear trajectory optimization / receding-horizon planning over setpoints**. It is *not* a direct multiple-shooting IPOPT MPC that optimizes every instantaneous motor voltage. Powell is a local derivative-free optimizer. The horizon comparison does not guarantee a globally optimal number of hops.

## Validations included

`python -m unittest -v test_sim.py` checks:
1. positive-definite mass matrix,
2. body corners can collide before the foot,
3. contact impulses remain inside the Coulomb friction cone and permit slip,
4. motor current/voltage and torque-speed limitations,
5. passive-spring numerical energy boundedness,
6. a valid single hop,
7. two-hop arrival near a 3 m target.

The included `demo/` output uses `--target 3 --compression .4 --lookahead 2 --maxfev 12 --dt .004`. It completes two hops near 3.007 m with safe pitch. This is a **nominal model result**, not a statement of hardware capability.

## Suggested validation before selecting hardware

1. Perform time-step and contact-iteration convergence checks (e.g. `dt=0.004, 0.002, 0.001`), comparing landing positions, impact impulses, and total energy drift.
2. Sweeps over mass/inertia, spring `k,c`, body width/height, friction, motor `R,Kt,Ke,N`, voltage/current.
3. Penalize or reject high power/speed/energy designs in a more explicit actuator-sizing outer optimization.
4. Recreate representative candidate jumps in a well-tested multibody physics engine and compare.
5. Add motor inductance/thermal and measured contact properties after hardware candidates are shortlisted.

**No thrusters are modeled in this version.**
