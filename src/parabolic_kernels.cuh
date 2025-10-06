// parabolic_kernels.cuh - Kernel definitions for parabolic step computations
#ifndef PARABOLIC_KERNELS_CUH
#define PARABOLIC_KERNELS_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "boundary_conditions.cuh"
#include "atomic_operations.cuh"


// ============================================================================
// Step 1: Velocity Kernels
// ============================================================================

template<int dim, typename Number>
__global__ void build_velocity_rhs_kernel(
    const State<dim, Number> old_U,
    const State<dim, Number> init_U,
    const Number* lumped_mass_matrix,
    Number* density,
    Number* velocity,
    Number* velocity_rhs,
    Number* internal_energy,
    const BoundaryData<dim, Number> boundary_data,
    int n_dofs)
{
    using PF = PhysicsFunctions<dim, Number>;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const Number rho_i = PF::density(old_U, i);
    const auto M_i = PF::momentum(old_U, i);
    const Number rho_e_i = PF::internal_energy(old_U, i);
    const Number m_i = lumped_mass_matrix[i];

    density[i] = rho_i;
    
    // velocity = momentum / density
    velocity[i * dim + 0] = M_i[0] / rho_i;
    // velocity_rhs = m_i * momentum (mass-weighted momentum)
    velocity_rhs[i * dim + 0] = m_i * M_i[0];
    
    if constexpr (dim >= 2) {
        velocity[i * dim + 1] = M_i[1] / rho_i;
        velocity_rhs[i * dim + 1] = m_i * M_i[1];
    }
    if constexpr (dim == 3) {
        velocity[i * dim + 2] = M_i[2] / rho_i;
        velocity_rhs[i * dim + 2] = m_i * M_i[2];
    }

    // Internal energy per unit mass
    internal_energy[i] = rho_e_i / rho_i;

    // Apply boundary conditions
    for (int b = 0; b < boundary_data.n_boundary_dofs; ++b) {
        if (boundary_data.boundary_dofs[b] == i) {
            const auto id = boundary_data.boundary_ids[b];

            if (id == 2) {  // Slip boundary
                Number V_i[dim];
                Number RHS_i[dim];
                for (int d = 0; d < dim; ++d) {
                    V_i[d] = velocity[i * dim + d];
                    RHS_i[d] = velocity_rhs[i * dim + d];
                }

                // Remove normal component
                Number V_i_dot_n = Number(0);
                Number RHS_i_dot_n = Number(0);  
                for (int d = 0; d < dim; ++d) {
                    V_i_dot_n += V_i[d] * boundary_data.boundary_normals[b * dim + d];
                    RHS_i_dot_n += RHS_i[d] * boundary_data.boundary_normals[b * dim + d];
                }
                for (int d = 0; d < dim; ++d){
                    V_i[d] -= V_i_dot_n * boundary_data.boundary_normals[b * dim + d];
                    RHS_i[d] -= RHS_i_dot_n * boundary_data.boundary_normals[b * dim + d];
                }
                for (int d = 0; d < dim; ++d) {
                    velocity[i * dim + d] = V_i[d];
                    velocity_rhs[i * dim + d] = RHS_i[d];
                }
            }
            else if (id == 1) {  // No-slip boundary
                for (int d = 0; d < dim; ++d) {
                    velocity[i * dim + d] = Number(0);
                    velocity_rhs[i * dim + d] = Number(0);
                }            
            }
            else if (id == 4) {  // Dirichlet inflow
                const Number rho_init_i = PF::density(init_U, i);
                const Number rho_init_inv = Number(1) / rho_init_i;
                const auto M_init_i = PF::momentum(init_U, i);
                const Number e_init = PF::internal_energy(init_U, i) * rho_init_inv;
                
                for (int d = 0; d < dim; ++d) {
                    velocity[i * dim + d] = M_init_i[d] * rho_init_inv;
                    velocity_rhs[i * dim + d] = m_i * M_init_i[d];
                }
                internal_energy[i] = e_init;
            }
            break;
        }
    }
}

template<int dim, typename Number>
__global__ void update_momentum_from_velocity_kernel(
    State<dim, Number> new_U,
    const State<dim, Number> old_U,
    const Number* velocity_solution,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const Number rho_i = old_U.rho[i];
    
    new_U.momentum_x[i] = rho_i * velocity_solution[i * dim + 0];
    if constexpr (dim >= 2) {
        new_U.momentum_y[i] = rho_i * velocity_solution[i * dim + 1];
    }
    if constexpr (dim == 3) {
        new_U.momentum_z[i] = rho_i * velocity_solution[i * dim + 2];
    }
}

// ============================================================================
// Step 2: Internal Energy System Kernels
// ============================================================================

template<int dim, typename Number>
__global__ void complete_internal_energy_rhs_kernel(
    const State<dim, Number> old_U,
    const State<dim, Number> init_U,
    const Number* velocity,
    const Number* velocity_new,
    const Number* density,
    const Number* internal_energy,
    Number* internal_energy_rhs,
    const Number* lumped_mass_matrix,
    const BoundaryData<dim, Number> boundary_data,
    Number tau,
    bool crank_nicolson_extrapolation,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const Number m_i = lumped_mass_matrix[i];
    const Number rho_i = density[i];
    const Number e_i = internal_energy[i];
    const Number i_e_rhs = internal_energy_rhs[i];  // Record before overwriting
    
    Number V_i[dim];
    Number V_i_new[dim];
    for (int d = 0; d < dim; ++d) {
        V_i[d] = velocity[i * dim + d];
        V_i_new[d] = velocity_new[i * dim + d];
    }
    
    // Kinetic energy correction: 0.5 * |V_old - V_new|^2
    Number correction;
    if (crank_nicolson_extrapolation) {
        correction = Number(0);
    } else {
        Number norm_square = Number(0);
        for (int d = 0; d < dim; ++d) {
            Number diff = V_i[d] - V_i_new[d];
            norm_square += diff * diff;
        }
        correction = Number(0.5) * norm_square;
    }
    
    internal_energy_rhs[i] = m_i * rho_i * (e_i + correction) + tau * i_e_rhs;

    // Apply boundary conditions
    for (int b = 0; b < boundary_data.n_boundary_dofs; ++b) {
        if (boundary_data.boundary_dofs[b] == i) {
            const auto id = boundary_data.boundary_ids[b];
            
            if (id == 4) {  // Dirichlet inflow
                const Number rho_init = init_U.rho[i];
                const Number rho_init_inv = Number(1) / rho_init;
                
                Number M_init[dim];
                M_init[0] = init_U.momentum_x[i];
                if constexpr (dim >= 2) M_init[1] = init_U.momentum_y[i];
                if constexpr (dim == 3) M_init[2] = init_U.momentum_z[i];
                
                Number kinetic = Number(0);
                for (int d = 0; d < dim; ++d) {
                    kinetic += M_init[d] * M_init[d];
                }
                kinetic *= Number(0.5) * rho_init_inv;
                
                const Number e_init = (init_U.energy[i] * rho_init_inv) - kinetic;
                internal_energy_rhs[i] = m_i * rho_i * e_init;
            }
            break;
        }
    }
}

template<int dim, typename Number>
__global__ void update_total_energy_from_internal_energy_kernel(
    State<dim, Number> new_U,
    const State<dim, Number> old_U,
    const Number* internal_energy_solution,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const Number rho_i = old_U.rho[i];
    const Number rho_i_inv = Number(1) / rho_i;
    const Number e_i = internal_energy_solution[i];
    
    // Kinetic energy from updated momentum
    Number kinetic = Number(0);
    kinetic += new_U.momentum_x[i] * new_U.momentum_x[i];
    if constexpr (dim >= 2) {
        kinetic += new_U.momentum_y[i] * new_U.momentum_y[i];
    }
    if constexpr (dim == 3) {
        kinetic += new_U.momentum_z[i] * new_U.momentum_z[i];
    }
    kinetic *= Number(0.5) * rho_i_inv;
    
    // Total energy = rho * e + kinetic
    new_U.energy[i] = rho_i * e_i + kinetic;
}

// ============================================================================
// Step 3: Kernel for update
// ============================================================================

template<int dim, typename Number>
__global__ void copy_density_kernel(
    State<dim, Number> new_U,
    const State<dim, Number> old_U,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    new_U.rho[i] = old_U.rho[i];
}

#endif // PARABOLIC_KERNELS_CUH