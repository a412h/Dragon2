#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"

// Device function for applying boundary conditions
template<int dim, typename Number>
__device__ void apply_boundary_conditions_device(
    State<dim, Number>& U,
    const BoundaryData<dim, Number>& boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int idx)
{
    // Test if DOF has a boundary condition
    for (int b = 0; b < boundary_data.n_boundary_dofs; ++b) {
        if (boundary_data.boundary_dofs[b] == idx) {
            const int bid = boundary_data.boundary_ids[b];
            
            if (bid == 4) {  // Dirichlet (inflow)
                U.rho[idx] = inflow_rho;
                U.momentum_x[idx] = inflow_momentum_x;
                if constexpr (dim >= 2) U.momentum_y[idx] = inflow_momentum_y;
                if constexpr (dim == 3) U.momentum_z[idx] = inflow_momentum_z;
                U.energy[idx] = inflow_energy;
            } 
            else if (bid == 2) {  // Slip boundary
                // Get the normal vector for this boundary DOF
                Number normal[dim];
                for (int d = 0; d < dim; ++d) {
                    normal[d] = boundary_data.boundary_normals[b * dim + d];
                }
                
                // Extract momentum components
                Number momentum[dim];
                momentum[0] = U.momentum_x[idx];
                if constexpr (dim >= 2) momentum[1] = U.momentum_y[idx];
                if constexpr (dim == 3) momentum[2] = U.momentum_z[idx];
                
                // Compute dot product (m, n)
                Number m_dot_n = Number(0);
                for (int d = 0; d < dim; ++d) {
                    m_dot_n += momentum[d] * normal[d];
                }
                
                // Remove normal component: m = m - (m, n) * n
                U.momentum_x[idx] = momentum[0] - m_dot_n * normal[0];
                if constexpr (dim >= 2) U.momentum_y[idx] = momentum[1] - m_dot_n * normal[1];
                if constexpr (dim == 3) U.momentum_z[idx] = momentum[2] - m_dot_n * normal[2];
            }
            else if (bid == 3) {  // No-slip (corner nodes)
                // Set all momentum components to zero
                U.momentum_x[idx] = Number(0);
                if constexpr (dim >= 2) U.momentum_y[idx] = Number(0);
                if constexpr (dim == 3) U.momentum_z[idx] = Number(0);
            }
            // bid == 0 is do_nothing
        }
    }
}

// Device function for computing precomputed values
template<int dim, typename Number>
__device__ void compute_precomputed_values_device(
    const State<dim, Number>& U,
    Number* precomputed,
    int idx)
{
    using PF = PhysicsFunctions<dim, Number>;
    precomputed[idx * 2 + 0] = PF::specific_entropy(U, idx);
    precomputed[idx * 2 + 1] = PF::harten_entropy(U, idx);
}

// Kernel for initial state preparation
template<int dim, typename Number>
__global__ void prepare_state_vector(
    State<dim, Number> d_U,
    Number* d_precomputed,
    const BoundaryData<dim, Number> boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_dofs,
    int n_precomputation_cycles = 1)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_dofs) return;
    
    // Apply boundary conditions
    apply_boundary_conditions_device<dim, Number>(
        d_U, boundary_data, 
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        idx);
    
    // Synchronize to ensure all boundary conditions are applied
    __syncthreads();
    
    // Perform precomputation cycles
    for (int cycle = 0; cycle < n_precomputation_cycles; ++cycle) {
        compute_precomputed_values_device<dim, Number>(d_U, d_precomputed, idx);
        
        if (cycle < n_precomputation_cycles - 1) {
            __syncthreads();
        }
    }
}

// Host function to launch the prepare_state_vector kernel
template<int dim, typename Number>
void launch_prepare_state_vector(
    State<dim, Number>& d_U,
    Number* d_precomputed,
    const BoundaryData<dim, Number>& d_boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_dofs,
    int n_precomputation_cycles,
    cudaStream_t stream = 0)
{
    const int block_size = 256;
    const int grid_size = (n_dofs + block_size - 1) / block_size;
    
    prepare_state_vector<dim, Number><<<grid_size, block_size, 0, stream>>>(
        d_U, d_precomputed, d_boundary_data,
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        n_dofs, n_precomputation_cycles
    );
    
    CUDA_CHECK(cudaGetLastError());
}

#endif // BOUNDARY_CONDITIONS_CUH