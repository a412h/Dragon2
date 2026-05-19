# Dragon2 Solver - User Manual

## Overview

Dragon2 is a GPU-accelerated NS, compressible, transient, solver. At the moment it uses
only Q1 continuous elements, with conformal meshes.

### System Requirements

- NVIDIA GPU with CUDA support (RTX 4000 / 5000 series tested)
- CUDA 12.0 or later
- deal.II 9.3+ or later
- Linux (Ubuntu 24.04+)

---

## Building

```bash
mkdir build && cd build
cmake -DDEAL_II_DIR=/path/to/dealii/install ..
make
```

You must provide the path to your deal.II installation via DEAL_II_DIR.

---

## Quick Start

Run from inside the `build/` directory so VTU output files land next to the
executable:

```bash
cd build
./solver_ns ../cases/cylinder-2d.prm
./solver_ns --help
```

## Configuration File Format

Dragon2 uses `.prm` parameter files (deal.II `ParameterHandler` format).
For example:

```
subsection A - TimeLoop
    set basename          = my_simulation
    set final time        = 1.0
    set timer granularity = 0.01
end

subsection B - Equation
    set dimension = 2
    set equation  = navier_stokes
    set gamma     = 1.4
end

subsection C - Discretization
    subsection mesh_file
        set file path                  = ../cases/my_mesh.msh
        set boundary mapping           = 0:do_nothing, 2:slip, 4:dirichlet
        set default boundary condition = do_nothing
    end
end

subsection E - InitialValues
    set direction = 1, 0

    subsection uniform
        set primitive state = 1.4, 3.0, 1.0
    end
end

subsection H - TimeIntegrator
    set cfl min = 0.9
    set cfl max = 0.9
end
```

---

## Configuration Reference

### A - TimeLoop Section

| Parameter | Description | Default |
|-----------|-------------|---------|
| basename | Output file prefix | simulation |
| final time | Simulation end time | 1.0 |
| timer granularity | Output interval | 0.01 |

### B - Equation Section

| Parameter | Description | Default |
|-----------|-------------|---------|
| dimension | 2 or 3 | 2 |
| equation | navier stokes |
| gamma | 1.4 |
| mu | 0.0 |
| lambda | 0.0 |
| kappa | 0.0 |

### C - Discretization Section

Subsection `mesh_file`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| file path | Path to `.msh` file (relative to cwd) | - |
| boundary mapping | Comma-separated list of `phys_id:bc_type` pairs (see below) | (empty) |
| default boundary condition | BC type for any physical ID not listed in the mapping | `do_nothing` |

Available BC types: `do_nothing`, `periodic`, `slip`, `no_slip`, `dirichlet`, `dynamic`.

### E - InitialValues Section

| Parameter | Description | Default |
|-----------|-------------|---------|
| direction | Flow direction (2D: x,y; 3D: x,y,z) | 1, 0 |
| primitive state | Initial state: rho, velocity, pressure | 1.4, 3.0, 1.0 |

### H - TimeIntegrator Section

| Parameter | Description | Default |
|-----------|-------------|---------|
| cfl min | Minimum CFL number | 0.9 |
| cfl max | Maximum CFL number | 0.9 |

---

## Boundary Conditions

| Type | Description | Use Case |
|------|-------------|----------|
| do_nothing, slip, no_slip, dirichlet, dynamic |

### Boundary mapping syntax

In the `.prm` file, map each Gmsh physical ID to a BC type:

```
set boundary mapping = phys_id1:bc_type1, phys_id2:bc_type2, ...
```

Example:

```
set boundary mapping = 1:dirichlet, 2:slip, 3:no_slip, 5:dynamic
```

Physical IDs that don't appear in the mapping take the `default boundary
condition` value.

---

## Mesh Requirements

Dragon2 reads Gmsh MSH format files.

### Creating Meshes in Gmsh

1. Create your geometry in Gmsh
2. Define Physical Groups for boundaries — use the numeric IDs above
3. Generate the mesh
4. Export as `.msh` file

### Physical IDs

- Assign each boundary face its numeric ID from the table above
- Unmapped IDs are treated as the `do_nothing` default

### Mesh Requirements

- Quadrilaterals (2D) or hexahedra (3D), conformal

---

## Example Cases

Dragon2 includes several example parameter files in `cases/`. All assume you
run from inside the `build/` directory so VTU output lands next to the
executable.

### 1. cylinder-2d.prm — 2D Cylinder Flow
`./solver_ns ../cases/cylinder-2d.prm`

### 2. cylinder-3d.prm — 3D Cylinder Flow
`./solver_ns ../cases/cylinder-3d.prm`

### 3. sphere-channel-3d.prm — 3D Sphere in Channel
`./solver_ns ../cases/sphere-channel-3d.prm`

### 4. capsule-2d.prm — 2D Reentry Capsule
`./solver_ns ../cases/capsule-2d.prm`

### 5. capsule-3d.prm — 3D Hypersonic Capsule
`./solver_ns ../cases/capsule-3d.prm`

### 6. oat15a-2d.prm — Transonic Airfoil
`./solver_ns ../cases/oat15a-2d.prm`

### Note on 3D cases

`main.cpp` is compiled with `constexpr int dim = 2`. To run a 3D case
(`cylinder-3d.prm`, `capsule-3d.prm`, `sphere-channel-3d.prm`), edit that
line to `constexpr int dim = 3` and rebuild.

---

## Output Files

Produces VTU output files for visualization.

### Output Naming Convention

```
{basename}_{timestep:06d}.vtu
```

Example: cylinder-2d_000050.vtu

### Visualizing Results

Open VTU files in ParaView:

1. Open ParaView
2. File -> Open -> Select VTU files
3. Apply
4. Color by density, pressure, or Mach number

### Available Fields

- rho
- momentum
- energy
- pressure
- schlieren

---

## Troubleshooting

Contact us

---

## License

Dragon2 is licensed under the Apache License 2.0. See LICENSE for details.

---

## Sponsor - Consulting
Consider contacting us if you need new functionalities.