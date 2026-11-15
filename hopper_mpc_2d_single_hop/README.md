# 2D reaction-wheel hopping robot — single-hop NMPC prototype

This is a small **educational**, **runnable** hybrid dynamics simulation for a passive-spring hopping robot with a flywheel. It includes:

- A CasADi/IPOPT **single-hop stance trajectory optimizer**.
- An example **receding-horizon MPC loop** that re-solves the optimization using the latest simulated state during stance.
- An independent `scipy.solve_ivp` simulation with **stance -> takeoff -> flight -> touchdown** events.
- Optional 2D flight thrust-force callback, set to **zero** in the example.
- Position, torque, orientation, and spring-length plots, plus an animated GIF.

## Setup

Use Python 3.10+ and run in this folder:

```bash
python -m pip install -r requirements.txt
```

## Run

### Optimize once, then simulate one hop (open-loop)

```bash
python hopper_2d.py --target 0.20
```

### Re-solve MPC as the robot moves (closed-loop simulation)

```bash
python run_mpc.py --target 0.20 --mpc-period 0.020
```

The result files appear in `results/` or `results_mpc/`. To speed up experimentation, use `--no-gif`. `--target` specifies the **desired foot landing location** in meters relative to the initially planted foot. Negative targets move left. You can change the initial passive spring preload:

```bash
python run_mpc.py --target 0.15 --compression 0.12
```

The 20-cm example was tested with the included parameters: the simulated touchdown was at approximately **0.192 m**, about **8 mm** short of the target. This is a numerical result for a hypothetical robot with simplified dynamics—not a hardware prediction or a guaranteed controller performance.

## Mechanical model

Spring leg is massless, attached to the rigid body and flywheel at the center of mass, with a **no-slip pinned foot** during stance. The flywheel angle `phi` is an **absolute inertial angle** (not body-relative). Leg/body angle `alpha` is measured from upright, and positive angle tips toward +x.

Stance state: `y = [alpha, phi, l, alpha_dot, phi_dot, l_dot]`.

The spring force comes from `0.5*k*(l-l0)^2`. There is **no spring motor**. The model starts with a 12-cm **precompressed spring** that supplies energy to the *first* hop. This initial compression must be created physically, e.g. by a latch, a drop, or later by thrust-based recharging. A passive spring sitting at its equilibrium compression does **not** spontaneously launch the robot.

The stance equations are:

```
(Ib + m*l^2)*alpha_ddot + 2*m*l*l_dot*alpha_dot - m*g*l*sin(alpha) = -tau
Iw*phi_ddot = tau
m*l_ddot - m*l*alpha_dot^2 + m*g*cos(alpha) + k*(l-l0) = 0
```

A motor's torque `tau` acts with opposite signs on the robot body and flywheel. Wheel speed and motor torque have limits.

During flight (ignoring thrusters by default), the COM follows ballistic motion, while the reaction-wheel torque affects body angle and flywheel speed. A simple PD flight controller reduces landing tilt. Touchdown is detected when the foot tip reaches the ground on descent.

The planner *approximates* takeoff as the spring reaching its free length `l0` while extending. It enforces ground-normal and Coulomb friction constraints at shooting-grid endpoints. The simulator also reports maximum residual friction violation, but does **not** model slip or enforce contact feasibility continuously between nodes.

## Optimization setup

`plan_one_hop()` in `hopper_2d.py` uses direct multiple shooting with RK4 steps and IPOPT:

- Decision variables: stance states `X`, motor torque sequence `U`, and stance duration `T`.
- Equality constraints: initial measured state, discrete stance dynamics, spring at free length at takeoff.
- Inequality constraints: torque, flywheel speed, spring length, normal force, friction cone, upward takeoff velocity.
- Main cost: `(predicted_landing_x - desired_landing_x)^2`.
- Small additional penalties for motor effort, control changes, body rotation, and wheel speed.

The planner predicts touchdown analytically by assuming the **leg is upright at touchdown**, so the foot touchdown x coordinate matches the COM x coordinate. The actual simulated landing uses a flight attitude controller and explicit foot-height contact event. Small differences between predicted and simulated landing positions are expected.

`run_mpc.py` makes this into a receding-horizon example by repeatedly:

1. Obtaining the current stance state from the ODE simulation (perfect measurement assumption).
2. Solving a new NLP for the remainder of stance.
3. Applying the optimized motor torque for only 20 ms.
4. Repeating until just before takeoff.

The final short interval is executed without another solve. The flight phase uses separate PD control, not a flight-phase NMPC. IPOPT solving is **not guaranteed real-time** on the example control period; do not assume this can execute on hardware without timing measurements, warm-starting, and optimization.

## Adjust parameters

Edit the `Robot` dataclass near the top of `hopper_2d.py` to change mass, body/wheel inertia, spring stiffness/free length, wheel speed limit, motor torque limit, friction coefficient, and flight attitude PD gains. The `--target`, `--compression`, and `--nodes` CLI flags are also available.

To examine how landing distance depends on target, run the program with several different `--target` values. Large values may be unreachable under the current torque and friction constraints; IPOPT will aim for the closest feasible outcome rather than promise the target.

## Extending to thrust and multiple hops

The **flight dynamics already support externally applied 2D forces** through `simulate_one_hop(..., flight_thrust=your_callback)`, where the callback has signature:

```python
# Thrust force applied through COM: Newtons (Fx, Fz)
def your_callback(time, flight_state):
    return (0.0, 0.0)
```

For example, a **future** horizontal force model could return `(0.5, 0.0)`, giving `0.5 / m` horizontal acceleration while airborne. This is only a forward-simulation capability: **the current optimizer assumes zero thrust**, so it must be expanded before it can *plan* optimal thruster use. Thrusters away from COM also add pitch torque, which would need to be modeled.

**Multiple hops are not yet implemented.** To extend this safely:

1. Add an **impact/reset model** at touchdown to map flight state and foot position to consistent new stance coordinates/velocities, including impact energy loss.
2. Model landing spring compression and damping, ground reaction, friction, and possibly a sliding foot.
3. Model where **energy for the next hop** comes from. Passive spring energy is lost on impact/damping. Add thrust-driven compression or another recharge mechanism; otherwise repeated hops die out.
4. Allow the foot anchor's world `x` position to change after each landing, and plan in global coordinates.
5. Expand the optimizer from stance-only landing prediction to hybrid stance/flight MPC if in-air thrusters are used for navigation.

### Important limitations

- Planar 2D model; no yaw, roll, lateral hopping, 3D inertias, or full body geometry.
- Massless leg/foot with leg axis rigidly aligned to body, a pinned point foot, and a linear spring.
- No leg damping, frictional energy losses, motor electrical dynamics, wheel angular momentum saturation recovery, or impact restitution yet.
- `l=l0` is a **simplified takeoff event**. In a realistic hybrid contact model, loss of support can happen earlier when the normal ground force reaches zero, and slip can occur if the friction cone is violated.
- Optimized contact constraints checked at grid nodes and endpoints only.
- MPC receives exact simulated state; add sensor and state-estimator noise for realistic testing.

This code should be used to verify controller concepts, compare objectives and understand the physics before moving to a more detailed mechanical/hardware simulation.
