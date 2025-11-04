<div align="center">
<pre>
██████╗ ██████╗  █████╗  ██████╗  ██████╗ ███╗   ██╗    ██╗██╗
██╔══██╗██╔══██╗██╔══██╗██╔════╝ ██╔═══██╗████╗  ██║    ██║██║
██║  ██║██████╔╝███████║██║  ███╗██║   ██║██╔██╗ ██║    ██║██║
██║  ██║██╔══██╗██╔══██║██║   ██║██║   ██║██║╚██╗██║    ██║██║
██████╔╝██║  ██║██║  ██║╚██████╔╝╚██████╔╝██║ ╚████║    ██║██║
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝    ╚═╝╚═╝
</pre>
</div>

# DRAGON II - an experimental CUDA port of the solver ryujin - in development

## Description
This project is a CUDA port of the solver ryujin (https://github.com/conservation-laws/ryujin).
It is not a fork as most of the original code have been entirely rewritten for CUDA, and it is not an official contribution.

## Key Concepts
The approach followed here is to transfer the entire computation on the GPU to offer maximal speed.

## Remarks
- Version II (Navier-Stokes)  
  - Navier-Stokes Mach3 cylinder cases in 2d and 3d have been tested  
- Single GPU computation
- Except for the generation of output files, libraries OpenMP, MPI and SIMD have been removed, as all computations are done on the GPU

## Supported OS
Tested on Ubuntu 24.04

## Prerequisites
Tested with a Nvidia RTX 4000 generation card

## Build and run

### Build with:
```bash
cd build/
cmake ..
make
```

### Then run with:
```bash
./solver_ns
```

### - These are extracts of transcient Navier-Stokes simulations (run on a laptop) -

#### Cylinder Mach3, resp. in 2d and 3d:

![Cylinder 2D](ns_mach3_2d_7.png)
![Cylinder 3D](ns_mach3_3d_5.png)

#### Sphere flying at Mach3 inside a uniform channel (contour of density, 11 millions of cells):

![Sphere 3D](ns_sphere_channel_mach3.png)

