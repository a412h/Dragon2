# Dragon2 Solver

Transient Direct Navier-Stokes solver, written in CUDA / C++

## Features

- Fast implementation
- 2D and 3D simulations
- Simple use: just load Gmsh mesh (.msh format), set boundary conditions, run

## Requirements

- NVIDIA GPU (RTX 3000/4000/5000 series or compatible)
- CUDA Toolkit 12.0+
- deal.II 9.3+ (finite element library)
- CMake 3.18+
- Linux (Ubuntu 22.04+ recommended)
- OpenMP

## Building

```bash
mkdir build && cd build
cmake -DDEAL_II_DIR=/path/to/dealii/install ..
make -j$(nproc)
```

Set DEAL_II_DIR to the path of deal.II installation

The build step detects GPU architecture (on CMake 3.24+.)
On older CMake versions it compiles for sm_86, sm_89, and sm_100.

## Quick Start

```bash
./solver_ns examples/cylinder-2d.cfg
./solver_ns --help
```

## Example Cases

| cylinder-2d.cfg | Cylinder flow | 2D |
| cylinder-3d.cfg | Cylinder flow | 3D |
| sphere-channel-3d.cfg | Sphere in channel | 3D |
| capsule-2d.cfg | Nasa capsule 120-CA Reentry | 2D |
| capsule-3d.cfg | Nasa capsule 120-CA Reentry | 3D |
| oat15a-2d.cfg | Onera OAT15a Transonic airfoil | 2D |

## Documentation

- [User Manual](docs/USER_MANUAL.md)
- [Test Cases](docs/TEST_CASES.md)

## License

Licensed under the Apache License 2.0. See LICENSE for details.

### - These are extracts of transcient Navier-Stokes simulations -

#### Cylinder Mach3, resp. in 2d and 3d:

![Cylinder 2D](ns_mach3_2d_7.png)
![Cylinder 3D](ns_mach3_3d_5.png)

#### Sphere flying at Mach3 inside a uniform channel (contour of density, 1.6 millions of cells):

![Sphere 3D](ns_sphere_channel_mach3.png)


#### Flow transonic with wing Onera OAT15a (contour of density, 0.5 million of points):

![Flow 2D](ns_oat15a.png)


#### Atmospheric-entry of Nasa Capsule 120-CA

##### mesh with dealii:

![Flow 2D](nasa_120_ca_mesh.png)

##### solution as contour of density, 4.5 millions of points, Mach 8:

![Flow 2D](nasa_120_ca_reentry_4M_mesh.png)

##### solution as contour of density with isolines, 0.4 millions of points, Mach 0.8:

![Flow 2D](nasa_120_ca_reentry_mach_0.8.png)

##### solution as density, 0.4 millions of points, Mach 3:

![Flow 2D](nasa_120_ca_reentry_mach_3.png)

