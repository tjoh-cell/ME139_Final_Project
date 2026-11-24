# Hardware selection with the twin-pogo simulator

Open **Hopper Setup.pyw** on Windows, or run **python view.py --setup** from this directory using the Python environment with MuJoCo installed. For a fresh environment, install the supplied requirements and use a Python installation with Tk support. The separate settings window stays open while MuJoCo runs.

1. On **Mission**, choose a destination, controller and requested apex. Feedback is fastest for comparing components. Predictive normally prepares the mission before opening playback.
2. Enter dimensions, masses and motor data on the hardware tabs. The lower readout updates the modeled mass, spring preload energy, maximum upward thrust-to-weight ratio and rotor consistency check.
3. Click **Run simulation**. Edit fields during playback to prepare another setup. Clicking Run again closes the previous viewer, builds a fresh model and starts from the specified preload. Closing only the viewer leaves the setup window open.
4. Use **Save settings** and **Load settings** for reusable JSON presets. **Export completed runs** writes the input settings and measurements for every finished trial to a comparison JSON. Stopped runs have no completed outcome. Last touchdown x is the foot location; the live x readout tracks the trunk.

## Hardware selection notes

- **Springs**. The two catalog rates add in parallel. Two 0.225 N/mm springs give a combined 450 N/m. Initial energy is ½ k_total × preload². Compare preload with usable stroke and the manufacturer's allowable deflection. Enter the moving portion of the legs separately from the fixed structure.
- **Mass and inertia**. Use the assembled fixed trunk mass and its principal inertias about its center of mass, aligned with the modeled axes. Add the rotating wheel assembly and sliding feet in their own fields. The model fixes the trunk center of mass at the box origin and the wheel shaft along body y; it does not fit a full CAD inertia tensor or arbitrary shaft placement.
- **Flywheel**. Enter axial and transverse moments from CAD or measurements. For a uniform solid disk, spin inertia is ½ m r². For mass confined within outer radius r, m r² is the upper bound. The historical defaults specify 0.003 kg·m², 0.03 kg and 0.05 m; the bound is 0.000075 kg·m². The panel displays this discrepancy and preserves the reference configuration for comparisons. Changing the drawn radius alone does not change the inertia.
- **Motor and gearbox**. Enter shaft-side Kt and Ke with consistent winding/current conventions. The ideal conversion from speed constant in rpm/V is Ke = 60/(2π Kv) in V·s/rad when the voltage definitions match. Enter reduction only once. Wheel speed is output-side; multiply by the gear ratio for motor shaft speed. The results show peak motor current, voltage and relative wheel speed. Evaluate continuous torque, thermal loading and battery sag separately.
- **Thrusters**. Use thrust per installed nozzle and include nozzle, gimbal, driver and energy-source mass in the trunk. The budget is the integral of both force magnitudes over time. It does not model electrical energy, fuel mass or temperature. The vertical thrust-to-weight readout assumes both nozzles can point straight up at full thrust; actual thrust depends on angle, ascent gates, slew and remaining impulse.
- **Hardware comparisons**. Hold goal, controller, preload and environment fixed while changing components. Track touchdown outcome, height, used impulse and actuator peaks. Arrival means a qualified foot touchdown inside tolerance; a stable stop is a separate requirement.

## Editable parameters and starting values

The following are simulator defaults, not hardware ratings. Angles labeled deg are entered in degrees. The setup window also has switches for thruster use, feedback/predictive control and live predictive planning.

### Mission

| Parameter | Default | Meaning |
|---|---:|---|
| Destination x | 15 m | Touchdown target along the ground. |
| Requested apex | 3 m | Target trunk height during flight. |
| Playback speed | 1 × | Display speed relative to real time. |
| Hop limit | 100 hops | End the mission at this many landings. |
| Time limit | 300 s | Maximum simulated mission duration. |
| Predictive search budget | 8 actions | Maximum first-hop candidates per search. |
| Steady crosswind | 0 m/s | Signed wind velocity along world y. |
| Crosswind gust amplitude | 0 m/s | Amplitude of the sinusoidal gust. |

### Springs & feet

| Parameter | Default | Meaning |
|---|---:|---|
| Free leg length | 1 m | Distance from trunk pivot to toe center at rest. |
| Minimum leg length | 0.18 m | Length at full compression. |
| Launch preload | 0.5 m | Initial shortening of each spring, applied once. |
| Combined spring stiffness | 450 N/m | For two identical springs, enter twice the catalog rate in N/m. |
| Combined damping | 1 N·s/m | Sum of both legs’ axial damping coefficients. |
| Combined moving leg mass | 0.035 kg | Total sliding mass lumped at the two feet. |
| Toe radius | 0.012 m | Radius of each spherical contact surface. |
| Leg mount offset | 0.085 m | Lateral distance from centerline to one leg. |
| Sliding friction | 0.85  | Toe and ground friction coefficient. |

### Body & wheel

| Parameter | Default | Meaning |
|---|---:|---|
| Trunk mass | 1 kg | Include fixed hardware and payload. Exclude rotor and sliding legs. |
| Trunk roll inertia | 0.011 kg·m² | About body x through the trunk center of mass. |
| Trunk pitch inertia | 0.015 kg·m² | About body y through the trunk center of mass. |
| Trunk yaw inertia | 0.011 kg·m² | About body z through the trunk center of mass. |
| Trunk length along x | 0.18 m | Front-to-back collision-box dimension. |
| Trunk width along y | 0.24 m | Side-to-side collision-box dimension. |
| Trunk height | 0.22 m | Vertical collision-box dimension. |
| Rotor mass | 0.03 kg | Rotating assembly only. |
| Rotor outer radius | 0.05 m | Sets the drawing and inertia check; enter inertia separately. |
| Rotor spin inertia | 0.003 kg·m² | About the wheel shaft (body y). Use measured or CAD inertia. |
| Rotor transverse inertia | 0.0016 kg·m² | About either axis across the shaft, assuming axial symmetry. |

### Motor & gearbox

| Parameter | Default | Meaning |
|---|---:|---|
| Available voltage | 48 V | Available motor-drive bus voltage under load. |
| Current limit | 50 A | Controller current cap used by the motor model. |
| Motor resistance | 0.16 Ω | Effective winding resistance in the DC-equivalent model. |
| Torque constant | 0.05 N·m/A | Motor shaft torque per ampere before gearing. |
| Back-EMF constant | 0.05 V·s/rad | Motor-side generated voltage per rad/s. |
| Gear reduction | 5 motor / wheel | 5 means five motor revolutions per wheel revolution. |
| Gear efficiency | 0.9 fraction | 0.9 means 90 percent. |
| Wheel speed warning | 190 rad/s | Flags peak speed in results; the motor can exceed it. |

### Thrusters

| Parameter | Default | Meaning |
|---|---:|---|
| Peak thrust per nozzle | 8 N | Maximum magnitude from one nozzle. |
| Mission impulse budget | 100 N·s | Shared by both nozzles across all hops. |
| Thrust slew per nozzle | 120 N/s | Maximum magnitude change while the flight gate is open. |
| Gimbal slew | 9 rad/s | Maximum angular rate on each gimbal axis. |
| Minimum elevation | 0 deg | Body-frame angle from forward toward up. |
| Maximum elevation | 160 deg | Maximum angle in the forward/up plane. |
| Lateral deflection | 45 deg | Symmetric side-to-side angular limit. |
| Nozzle mount offset | 0.16 m | Distance from centerline to one nozzle. |
| Thrust cutoff height | 7 m | Disable thrust above this trunk height. |
| Mission height ceiling | 8 m | Crossing this trunk height ends the run. |

### Guidance

| Parameter | Default | Meaning |
|---|---:|---|
| Cruise speed | 8 m/s | Desired horizontal travel speed. |
| Braking acceleration | 2 m/s² | Guidance assumption for slowing near the goal. |
| Foot-placement gain | 0.55  | Scales landing lean. The stance-swing default is 0.55. |
| Forward thrust gain | 0.25 1/s | Scales thrust used to correct horizontal ground speed. |
| Vertical thrust request | 10 N | Desired total vertical assist below the requested apex. |
| Goal tolerance along x | 0.05 m | Maximum foot-position error accepted at touchdown. |
