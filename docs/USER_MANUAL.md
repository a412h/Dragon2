# Dragon2 Solver - User Manual

## Overview

Dragon2 is a GPU-accelerated NS, direct, compressible, transient, solver. At the moment it uses
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

```bash
./solver_ns config.cfg
./solver_ns --help
```

## Configuration File Format

Dragon uses .cfg configuration files. For example:

```
subsection A - TimeLoop
    set basename          = my_simulation
    set final time        = 1.0
    set timer granularity = 0.01
end

subsection B - Equation
    set dimension = 2
    set equation  = navier stokes
    set gamma     = 1.4
end

subsection C - Discretization
    subsection mesh_file
        set file path                  = cases/my_mesh.msh
        set boundary mapping           = 1:dirichlet, 2:slip, 3:no_slip
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

### C - Discretization Section (mesh_file)

| Parameter | Description | Default |
|-----------|-------------|---------|
| file path | Path to .msh file | - |
| boundary mapping | Physical ID to BC mapping | - |
| default boundary condition | Default BC type | do_nothing |

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

### Boundary Mapping Syntax

In the .cfg file, map Gmsh physical IDs to boundary conditions:

```
set boundary mapping = phys_id1:bc_type, phys_id2:bc_type, ...
```

Example:
```
set boundary mapping = 1:dirichlet, 2:slip, 3:no_slip, 5:dynamic
```

---

## Mesh Requirements

Dragon2 reads Gmsh MSH format files

### Creating Meshes in Gmsh

1. Create your geometry in Gmsh
2. Define Physical Groups for boundaries
3. Generate mesh
4. Export as .msh file

### Physical IDs

- Assign unique physical IDs to each boundary surface
- Map these IDs in your configuration file
- Unmapped IDs use the default boundary condition

### Mesh Requirements

- Only rectangle or hexahedrons with conformal mesh

---

## Example Cases

Dragon2 includes several example configurations in the cases/ directory:

### 1. cylinder-2d.cfg - 2D Cylinder Flow
Run: ./solver_ns cases/cylinder-2d.cfg

### 2. cylinder-3d.cfg - 3D Cylinder Flow
Run: ./solver_ns cases/cylinder-3d.cfg

### 3. sphere-channel-3d.cfg - 3D Sphere in Channel
Run: ./solver_ns cases/sphere-channel-3d.cfg

### 4. capsule-2d.cfg - 2D Reentry Capsule
Run: ./solver_ns cases/capsule-2d.cfg

### 5. capsule-3d.cfg - 3D Hypersonic Capsule
Run: ./solver_ns cases/capsule-3d.cfg

### 6. oat15a-2d.cfg - Transonic Airfoil
Run: ./solver_ns cases/oat15a-2d.cfg

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