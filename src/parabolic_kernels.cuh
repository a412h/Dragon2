#ifndef PARABOLIC_KERNELS_CUH
#define PARABOLIC_KERNELS_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "boundary_conditions.cuh"
#include "atomic_operations.cuh"



template<int dim, typename Number>
__global__ void __launch_bounds__(256, 4) build_velocity_rhs_kernel(
    const State<dim, Number> old_U,
    const State<dim, Number> init_U,
    const Number* __restrict__ lumped_mass_matrix,
    Number* __restrict__ density,
    Number* __restrict__ velocity,
    Number* __restrict__ velocity_rhs,
    Number* __restrict__ internal_energy,
    const BoundaryData<dim, Number> boundary_data,
    int n_dofs)
{
    using PF = PhysicsFunctions<dim, Number>;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number rho_i = __ldg(&old_U.rho[i]);
    Number M_i[dim];
    M_i[0] = __ldg(&old_U.momentum_x[i]);
    if constexpr (dim >= 2) M_i[1] = __ldg(&old_U.momentum_y[i]);
    if constexpr (dim == 3) M_i[2] = __ldg(&old_U.momentum_z[i]);
    const Number E_i = __ldg(&old_U.energy[i]);
    const Number rho_i_inv = Number(1) / rho_i;

    Number m_sq = M_i[0] * M_i[0];
    if constexpr (dim >= 2) m_sq += M_i[1] * M_i[1];
    if constexpr (dim == 3) m_sq += M_i[2] * M_i[2];
    const Number rho_e_i = E_i - Number(0.5) * m_sq * rho_i_inv;

    const Number m_i = __ldg(&lumped_mass_matrix[i]);

    density[i] = rho_i;

    velocity[i * dim + 0] = M_i[0] * rho_i_inv;
    velocity_rhs[i * dim + 0] = m_i * M_i[0];

    if constexpr (dim >= 2) {
        velocity[i * dim + 1] = M_i[1] * rho_i_inv;
        velocity_rhs[i * dim + 1] = m_i * M_i[1];
    }
    if constexpr (dim == 3) {
        velocity[i * dim + 2] = M_i[2] * rho_i_inv;
        velocity_rhs[i * dim + 2] = m_i * M_i[2];
    }

    internal_energy[i] = rho_e_i * rho_i_inv;

    const int id = __ldg(&boundary_data.bc_type[i]);
    if (id < 0) return;

    const int b = __ldg(&boundary_data.bc_index[i]);

    if (id == 2) {
        Number V_i[dim];
        Number RHS_i[dim];
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            V_i[d] = velocity[i * dim + d];
            RHS_i[d] = velocity_rhs[i * dim + d];
        }

        Number V_i_dot_n = Number(0);
        Number RHS_i_dot_n = Number(0);
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            const Number n_d = __ldg(&boundary_data.boundary_normals[b * dim + d]);
            V_i_dot_n += V_i[d] * n_d;
            RHS_i_dot_n += RHS_i[d] * n_d;
        }
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            const Number n_d = __ldg(&boundary_data.boundary_normals[b * dim + d]);
            V_i[d] -= V_i_dot_n * n_d;
            RHS_i[d] -= RHS_i_dot_n * n_d;
        }
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            velocity[i * dim + d] = V_i[d];
            velocity_rhs[i * dim + d] = RHS_i[d];
        }
    }
    else if (id == 1) {
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            velocity[i * dim + d] = Number(0);
            velocity_rhs[i * dim + d] = Number(0);
        }
    }
    else if (id == 4) {
        const Number rho_init_i = __ldg(&init_U.rho[i]);
        const Number rho_init_inv = Number(1) / rho_init_i;
        Number M_init_i[dim];
        M_init_i[0] = __ldg(&init_U.momentum_x[i]);
        if constexpr (dim >= 2) M_init_i[1] = __ldg(&init_U.momentum_y[i]);
        if constexpr (dim == 3) M_init_i[2] = __ldg(&init_U.momentum_z[i]);
        const Number E_init_i = __ldg(&init_U.energy[i]);

        Number m_sq_init = M_init_i[0] * M_init_i[0];
        if constexpr (dim >= 2) m_sq_init += M_init_i[1] * M_init_i[1];
        if constexpr (dim == 3) m_sq_init += M_init_i[2] * M_init_i[2];
        const Number e_init = (E_init_i - Number(0.5) * m_sq_init * rho_init_inv) * rho_init_inv;

        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            velocity[i * dim + d] = M_init_i[d] * rho_init_inv;
            velocity_rhs[i * dim + d] = m_i * M_init_i[d];
        }
        internal_energy[i] = e_init;
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


template<int dim, typename Number>
__global__ void __launch_bounds__(256, 4) complete_internal_energy_rhs_kernel(
    const State<dim, Number> old_U,
    const State<dim, Number> init_U,
    const Number* __restrict__ velocity,
    const Number* __restrict__ velocity_new,
    const Number* __restrict__ density,
    const Number* __restrict__ internal_energy,
    Number* __restrict__ internal_energy_rhs,
    const Number* __restrict__ lumped_mass_matrix,
    const BoundaryData<dim, Number> boundary_data,
    Number tau,
    bool crank_nicolson_extrapolation,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number m_i = __ldg(&lumped_mass_matrix[i]);
    const Number rho_i = __ldg(&density[i]);
    const Number e_i = __ldg(&internal_energy[i]);
    const Number i_e_rhs = __ldg(&internal_energy_rhs[i]);

    Number V_i[dim];
    Number V_i_new[dim];
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        V_i[d] = __ldg(&velocity[i * dim + d]);
        V_i_new[d] = __ldg(&velocity_new[i * dim + d]);
    }

    Number correction;
    if (crank_nicolson_extrapolation) {
        correction = Number(0);
    } else {
        Number norm_square = Number(0);
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            Number diff = V_i[d] - V_i_new[d];
            norm_square += diff * diff;
        }
        correction = Number(0.5) * norm_square;
    }

    Number result = m_i * rho_i * (e_i + correction) + tau * i_e_rhs;

    const int id = __ldg(&boundary_data.bc_type[i]);
    if (id == 4) {
        const int b = __ldg(&boundary_data.bc_index[i]);
        (void)b;

        const Number rho_init = __ldg(&init_U.rho[i]);
        const Number rho_init_inv = Number(1) / rho_init;

        Number M_init[dim];
        M_init[0] = __ldg(&init_U.momentum_x[i]);
        if constexpr (dim >= 2) M_init[1] = __ldg(&init_U.momentum_y[i]);
        if constexpr (dim == 3) M_init[2] = __ldg(&init_U.momentum_z[i]);

        Number kinetic = Number(0);
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            kinetic += M_init[d] * M_init[d];
        }
        kinetic *= Number(0.5) * rho_init_inv;

        const Number e_init = (__ldg(&init_U.energy[i]) * rho_init_inv) - kinetic;
        result = m_i * rho_i * e_init;
    }

    internal_energy_rhs[i] = result;
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

    Number kinetic = Number(0);
    kinetic += new_U.momentum_x[i] * new_U.momentum_x[i];
    if constexpr (dim >= 2) {
        kinetic += new_U.momentum_y[i] * new_U.momentum_y[i];
    }
    if constexpr (dim == 3) {
        kinetic += new_U.momentum_z[i] * new_U.momentum_z[i];
    }
    kinetic *= Number(0.5) * rho_i_inv;

    new_U.energy[i] = rho_i * e_i + kinetic;
}


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

#endif