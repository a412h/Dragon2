# Dragon2 - Test Case Documentation

This document provides detailed information about each example test case included with Dragon2.

---

## 1. 2D Cylinder Flow (cylinder-2d.cfg)

### Physical Problem

Mach 3 supersonic flow around a circular cylinder is a classic validation case for compressible flow solvers. The flow features a strong detached bow shock upstream of the cylinder and complex wake structures downstream.

### Flow Conditions

| Parameter | Value |
|-----------|-------|
| Mach number | 3.0 |
| Reynolds number | Inviscid (Euler) |
| Cylinder diameter | 0.5 units |
| Domain size | 4.0 x 2.0 units |
| Flow direction | Left to right (+x) |

### Initial State

```
rho (density) = 1.4
velocity = 3.0 (Mach 3)
pressure = 1.0
gamma = 1.4
```

### Boundary Conditions

| Boundary | Location | Type |
|----------|----------|------|
| Inflow | Left | Dirichlet |
| Outflow | Right | Do-nothing |
| Walls | Top/Bottom | Slip |
| Cylinder | Center | Slip |

### Mesh Details

- Cells: 36,864
- Vertices: 37,376
- Type: Structured with refinement near cylinder
- File: `Meshes/cylinder-2d.msh`

### Expected Results

- Detached bow shock upstream of cylinder
- Shock standoff distance ~0.2 diameters
- Expansion fans at cylinder shoulders
- Wake recirculation region

### Running the Case

```bash
./solver_ns examples/cylinder-2d.cfg
```

Simulation time: ~6 seconds for 5.0 time units (2420 steps)

---

## 2. 3D Cylinder Flow (cylinder-3d.cfg)

### Physical Problem

Three-dimensional extension of the classic Mach 3 cylinder flow. The 2D cylinder mesh is extruded in the z-direction to create a 3D domain with a cylindrical obstacle.

### Flow Conditions

| Parameter | Value |
|-----------|-------|
| Mach number | 3.0 |
| Reynolds number | Inviscid (Euler) |
| Cylinder diameter | 0.5 units |
| Domain size | 4.0 x 2.0 x 2.0 units |
| Flow direction | Left to right (+x) |

### Initial State

```
rho (density) = 1.4
velocity = 3.0 (Mach 3)
pressure = 1.0
gamma = 1.4
```

### Boundary Conditions

| Boundary | Location | Type |
|----------|----------|------|
| Inflow | Left | Dirichlet |
| Outflow | Right | Do-nothing |
| Walls | Top/Bottom/Front/Back | Slip |
| Cylinder | Center | Slip |

### Mesh Details

- Cells: 6,912
- Vertices: 9,728
- Type: Extruded 2D cylinder mesh (4 layers in z)
- File: `Meshes/cylinder-3d.msh`

### Expected Results

- 3D detached bow shock
- Similar flow features to 2D case with spanwise uniformity
- Expansion fans at cylinder shoulders

### Running the Case

**Note**: Dragon must be compiled with `dim = 3` to run this case.

```bash
./solver_ns examples/cylinder-3d.cfg
```

---

## 3. 3D Sphere in Channel (sphere-channel-3d.cfg)

### Physical Problem

Three-dimensional Mach 3 viscous flow around a sphere in a rectangular channel. This case demonstrates 3D shock-body interaction with viscous effects.

### Flow Conditions

| Parameter | Value |
|-----------|-------|
| Mach number | 3.0 |
| Equation | Navier-Stokes |
| Dynamic viscosity | 1.0e-3 |
| Sphere radius | 0.5 units |
| Channel dimensions | 6.0 x 2.0 x 1.0 units |

### Initial State

```
rho (density) = 1.4
velocity = 3.0 (Mach 3)
pressure = 1.0
gamma = 1.4
mu = 1.0e-3
```

### Boundary Conditions

| Boundary | Type |
|----------|------|
| Inflow | Dirichlet |
| Outflow | Do-nothing |
| Channel walls | Slip |
| Sphere surface | No-slip (viscous wall) |

### Mesh Details

- Cells: 4,736
- Vertices: 5,658
- Type: Unstructured with shell mesh around sphere
- File: `Meshes/sphere-channel-3d.msh`

### Expected Results

- 3D detached bow shock
- Viscous boundary layer on sphere
- Wake structure with 3D vortical features

### Running the Case

```bash
./solver_ns examples/sphere-channel-3d.cfg
```

---

## 4. 2D Reentry Capsule (capsule-2d.cfg)

### Physical Problem

Supersonic flow around a NASA-style atmospheric reentry capsule. The capsule geometry consists of a spherical nose cap and a conical afterbody. This is a simplified 2D axisymmetric-style simulation.

### Flow Conditions

| Parameter | Value |
|-----------|-------|
| Mach number | 3.0 |
| Equation | Euler |
| Flow direction | Right to left (-x) |
| Capsule diameter | 198 units |

### Initial State

```
rho (density) = 1.4
velocity = 3.0 (Mach 3)
pressure = 1.0
direction = (-1, 0) - reentry direction
```

### Boundary Conditions

| Boundary | Physical ID | Type |
|----------|-------------|------|
| Heat shield (capsule) | 3 | No-slip |
| Far-field | 5 | Dynamic (characteristic) |

### Mesh Details

- Cells: 112,640
- Vertices: 113,344
- Type: O-grid mesh for good shock capturing
- File: `Meshes/capsule-2d.msh`

### Expected Results

- Strong bow shock ahead of capsule
- Stagnation region at nose
- Expansion around shoulders
- Wake recirculation

### Running the Case

```bash
./solver_ns examples/capsule-2d.cfg
```

---

## 5. 3D Hypersonic Capsule (capsule-3d.cfg)

### Physical Problem

Full 3D simulation of a NASA-style capsule at Mach 8 hypersonic conditions. This represents atmospheric reentry conditions (simplified, without real gas effects).

### Flow Conditions

| Parameter | Value |
|-----------|-------|
| Mach number | 8.0 (hypersonic) |
| Equation | Euler |
| Flow direction | Right to left (-x) |
| Capsule diameter | 198 units |

### Initial State

```
rho (density) = 1.4
velocity = 8.0 (Mach 8)
pressure = 1.0
direction = (-1, 0, 0)
```

### Boundary Conditions

| Boundary | Physical ID | Type |
|----------|-------------|------|
| Heat shield | 3 | No-slip |
| Far-field | 5 | Dynamic |

### Mesh Details

- Cells: 450,560
- Vertices: 461,906
- Type: 3D O-grid with 32 azimuthal divisions
- File: `Meshes/capsule-3d.msh`

**Note**: This is a large mesh requiring significant GPU memory (4+ GB recommended)

### Expected Results

- Very strong detached bow shock
- High temperature stagnation region
- 3D wake structure

### Running the Case

```bash
./solver_ns examples/capsule-3d.cfg
```

---

## 6. ONERA OAT15a Transonic Airfoil (oat15a-2d.cfg)

### Physical Problem

Transonic viscous flow over the ONERA OAT15a supercritical airfoil at Mach 0.73 with 3.5 degree angle of attack. This is a standard validation case for compressible Navier-Stokes solvers featuring shock-boundary layer interaction.

### Flow Conditions

| Parameter | Value | Units |
|-----------|-------|-------|
| Mach number | 0.73 | - |
| Angle of attack | 3.5 | degrees |
| Reynolds number | ~3e6 | - |
| Chord length | 0.23 | m |
| Freestream density | 1.225 | kg/m^3 |
| Freestream velocity | 248.42 | m/s |
| Freestream pressure | 101,300 | Pa |

### Initial State

```
rho = 1.225 kg/m^3 (sea level air)
velocity = 248.42 m/s (Mach 0.73)
pressure = 1.013e5 Pa (1 atm)
direction = (0.99813, 0.061049) - 3.5 deg AoA
gamma = 1.401
mu = 1.789e-5 Pa.s
kappa = 3.616e-5
```

### Boundary Conditions

| Boundary | Physical ID | Type |
|----------|-------------|------|
| Airfoil surface | 3 | No-slip |
| Far-field | 5 | Dynamic |

### Mesh Details

- File: `Meshes/oat15a_mesh.msh`
- High resolution near airfoil surface
- Boundary layer mesh for viscous effects

### Expected Results

- Weak normal shock on upper surface (~60-70% chord)
- Shock-induced boundary layer separation
- Turbulent wake downstream

### References

1. S. Deck, N. Renard. "Towards an enhanced protection of attached boundary layers in hybrid RANS/LES methods." J. Comput. Phys. 400:108970, 2020.

2. L. Jacquin et al. "Experimental Study of Shock Oscillation over a Transonic Supercritical Profile." AIAA J. 47:1985-1994, 2009.

### Running the Case

```bash
./solver_ns examples/oat15a-2d.cfg
```

---

## Creating Your Own Cases

### Step 1: Prepare Your Mesh

1. Create geometry in Gmsh (or your preferred meshing tool)
2. Define physical groups for boundaries
3. Generate mesh
4. Export as `.msh` file to `Meshes/` directory

### Step 2: Create Configuration File

Copy an existing example and modify:

```bash
cp examples/cylinder-2d.cfg examples/my_case.cfg
```

### Step 3: Configure Boundary Mapping

Map your mesh's physical IDs to boundary conditions:

```
set boundary mapping = 1:dirichlet, 2:slip, 3:no_slip
```

### Step 4: Set Initial Conditions

Specify the uniform initial state:

```
set primitive state = density, velocity, pressure
set direction = vx, vy  # (or vx, vy, vz for 3D)
```

### Step 5: Run and Validate

```bash
./solver_ns examples/my_case.cfg
```

Check output in ParaView for correctness.

---

## Tips for Successful Simulations

### CFL Number Selection

- Start with CFL = 0.5 for complex geometries
- Increase gradually to find stable maximum
- Typical stable range: 0.5 - 1.0 for explicit schemes

### Mesh Quality

- Aspect ratio < 10:1 preferred
- Smooth mesh transitions
- Adequate refinement near shocks and walls

### Boundary Condition Selection

| Scenario | Recommended BC |
|----------|----------------|
| Supersonic inflow | Dirichlet |
| Subsonic inflow | Dirichlet or Dynamic |
| Supersonic outflow | Do-nothing |
| Subsonic outflow | Do-nothing or Dynamic |
| Solid wall (inviscid) | Slip |
| Solid wall (viscous) | No-slip |
| Far-field | Dynamic |
