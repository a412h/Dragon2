#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"


template<int dim, typename Number, int component>
__device__ void prescribe_riemann_characteristic(

    Number rho_first,
    Number momentum_first[dim],
    Number energy_first,

    Number rho_second,
    Number momentum_second[dim],
    Number energy_second,
    const Number normal[dim],
    Number result[dim + 2])
{
    static_assert(component == 1 || component == 2, "component must be 1 or 2");
    
    using PF = PhysicsFunctions<dim, Number>;

    const Number rho_first_inv = Number(1.0) / rho_first;
    Number m_sq_first = momentum_first[0] * momentum_first[0];
    if constexpr (dim >= 2) m_sq_first += momentum_first[1] * momentum_first[1];
    if constexpr (dim == 3) m_sq_first += momentum_first[2] * momentum_first[2];
    const Number e_first = energy_first - Number(0.5) * m_sq_first * rho_first_inv;
    const Number p_first = PF::gamma_minus_one * e_first;
    const Number a_first = sqrt(PF::gamma * p_first * rho_first_inv);

    Number vn_first = Number(0);
    for (int d = 0; d < dim; ++d) {
        vn_first += momentum_first[d] * normal[d];
    }
    vn_first *= rho_first_inv;

    const Number rho_second_inv = Number(1.0) / rho_second;
    Number m_sq_second = momentum_second[0] * momentum_second[0];
    if constexpr (dim >= 2) m_sq_second += momentum_second[1] * momentum_second[1];
    if constexpr (dim == 3) m_sq_second += momentum_second[2] * momentum_second[2];
    const Number e_second = energy_second - Number(0.5) * m_sq_second * rho_second_inv;
    const Number p_second = PF::gamma_minus_one * e_second;
    const Number a_second = sqrt(PF::gamma * p_second * rho_second_inv);

    Number vn_second = Number(0);
    for (int d = 0; d < dim; ++d) {
        vn_second += momentum_second[d] * normal[d];
    }
    vn_second *= rho_second_inv;

    const Number R_1 = (component == 1)
        ? (vn_second - Number(2.0) * a_second / PF::gamma_minus_one)  
        : (vn_first - Number(2.0) * a_first / PF::gamma_minus_one);    
    
    const Number R_2 = (component == 2)
        ? (vn_second + Number(2.0) * a_second / PF::gamma_minus_one)  
        : (vn_first + Number(2.0) * a_first / PF::gamma_minus_one);    

    const Number s = p_first / pow(rho_first, PF::gamma);
    
    Number v_perp[dim];
    for (int d = 0; d < dim; ++d) {
        v_perp[d] = momentum_first[d] * rho_first_inv - vn_first * normal[d];
    }

    const Number vn_new = Number(0.5) * (R_1 + R_2);

    const Number factor = (PF::gamma_minus_one / Number(4.0)) * (R_2 - R_1);
    const Number rho_new_inner = (Number(1.0) / (PF::gamma * s)) * factor * factor;
    const Number rho_new = pow(rho_new_inner, Number(1.0) / PF::gamma_minus_one);

    const Number p_new = s * pow(rho_new, PF::gamma);

    Number v_perp_sq = Number(0);
    for (int d = 0; d < dim; ++d) {
        v_perp_sq += v_perp[d] * v_perp[d];
    }

    result[0] = rho_new;
    for (int d = 0; d < dim; ++d) {
        result[1 + d] = rho_new * (vn_new * normal[d] + v_perp[d]);
    }
    result[dim + 1] = p_new / PF::gamma_minus_one + 
                      Number(0.5) * rho_new * (vn_new * vn_new + v_perp_sq);
}




template<int dim, typename Number>
__global__ void apply_dirichlet_bc_kernel(
    State<dim, Number> U,
    const int* boundary_dofs,
    const int* boundary_ids,
    int n_boundary_dofs,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_boundary_dofs) return;
    if (boundary_ids[eid] != 4) return;  

    const int idx = boundary_dofs[eid];
    U.rho[idx] = inflow_rho;
    U.momentum_x[idx] = inflow_momentum_x;
    if constexpr (dim >= 2) U.momentum_y[idx] = inflow_momentum_y;
    if constexpr (dim == 3) U.momentum_z[idx] = inflow_momentum_z;
    U.energy[idx] = inflow_energy;
}


template<int dim, typename Number>
__global__ void apply_no_slip_bc_kernel(
    State<dim, Number> U,
    const int* boundary_dofs,
    const int* boundary_ids,
    int n_boundary_dofs)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_boundary_dofs) return;
    if (boundary_ids[eid] != 3) return;

    const int idx = boundary_dofs[eid];
    U.momentum_x[idx] = Number(0);
    if constexpr (dim >= 2) U.momentum_y[idx] = Number(0);
    if constexpr (dim == 3) U.momentum_z[idx] = Number(0);
}


template<int dim, typename Number>
__global__ void apply_slip_bc_kernel(
    State<dim, Number> U,
    const int* boundary_dofs,
    const int* boundary_ids,
    const Number* boundary_normals,
    int n_boundary_dofs)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_boundary_dofs) return;
    if (boundary_ids[eid] != 2) return;

    const int idx = boundary_dofs[eid];

    Number normal[dim];
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        normal[d] = boundary_normals[eid * dim + d];
    }

    Number momentum[dim];
    momentum[0] = U.momentum_x[idx];
    if constexpr (dim >= 2) momentum[1] = U.momentum_y[idx];
    if constexpr (dim == 3) momentum[2] = U.momentum_z[idx];

    Number m_dot_n = Number(0);
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        m_dot_n += momentum[d] * normal[d];
    }

    U.momentum_x[idx] = momentum[0] - m_dot_n * normal[0];
    if constexpr (dim >= 2) U.momentum_y[idx] = momentum[1] - m_dot_n * normal[1];
    if constexpr (dim == 3) U.momentum_z[idx] = momentum[2] - m_dot_n * normal[2];
}


template<int dim, typename Number>
__global__ void apply_dynamic_bc_kernel(
    State<dim, Number> U,
    const int* boundary_dofs,
    const int* boundary_ids,
    const Number* boundary_normals,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_boundary_dofs)
{
    using PF = PhysicsFunctions<dim, Number>;
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_boundary_dofs) return;
    if (boundary_ids[eid] != 5) return;

    const int idx = boundary_dofs[eid];

    Number normal[dim];
    #pragma unroll
    for (int d = 0; d < dim; ++d) {
        normal[d] = boundary_normals[eid * dim + d];
    }

    const Number rho_curr = U.rho[idx];
    Number m_curr[dim];
    m_curr[0] = U.momentum_x[idx];
    if constexpr (dim >= 2) m_curr[1] = U.momentum_y[idx];
    if constexpr (dim == 3) m_curr[2] = U.momentum_z[idx];
    const Number E_curr = U.energy[idx];

    const Number rho_inv = Number(1.0) / rho_curr;
    const Number a = PF::speed_of_sound(U, idx);

    Number vn = Number(0);
    #pragma unroll
    for (int d = 0; d < dim; ++d) vn += m_curr[d] * normal[d];
    vn *= rho_inv;

    Number momentum_bar[dim];
    momentum_bar[0] = inflow_momentum_x;
    if constexpr (dim >= 2) momentum_bar[1] = inflow_momentum_y;
    if constexpr (dim == 3) momentum_bar[2] = inflow_momentum_z;

    Number result[dim + 2];

    if (vn < -a) {
        U.rho[idx] = inflow_rho;
        U.momentum_x[idx] = inflow_momentum_x;
        if constexpr (dim >= 2) U.momentum_y[idx] = inflow_momentum_y;
        if constexpr (dim == 3) U.momentum_z[idx] = inflow_momentum_z;
        U.energy[idx] = inflow_energy;
    } else if (vn >= -a && vn <= Number(0)) {
        prescribe_riemann_characteristic<dim, Number, 2>(
            inflow_rho, momentum_bar, inflow_energy,
            rho_curr, m_curr, E_curr,
            normal, result);
        U.rho[idx] = result[0];
        U.momentum_x[idx] = result[1];
        if constexpr (dim >= 2) U.momentum_y[idx] = result[2];
        if constexpr (dim == 3) U.momentum_z[idx] = result[3];
        U.energy[idx] = result[dim + 1];
    } else if (vn > Number(0) && vn <= a) {
        prescribe_riemann_characteristic<dim, Number, 1>(
            rho_curr, m_curr, E_curr,
            inflow_rho, momentum_bar, inflow_energy,
            normal, result);
        U.rho[idx] = result[0];
        U.momentum_x[idx] = result[1];
        if constexpr (dim >= 2) U.momentum_y[idx] = result[2];
        if constexpr (dim == 3) U.momentum_z[idx] = result[3];
        U.energy[idx] = result[dim + 1];
    }

}


template<int dim, typename Number>
void apply_boundary_conditions(
    State<dim, Number>& U,
    const BoundaryData<dim, Number>& bd,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    cudaStream_t stream,
    bool apply_dirichlet = true)
{
    const int n = bd.n_boundary_dofs;
    if (n <= 0) return;
    const int blocks = (n + 255) / 256;

    if (apply_dirichlet)
        apply_dirichlet_bc_kernel<dim, Number><<<blocks, 256, 0, stream>>>(
            U, bd.boundary_dofs, bd.boundary_ids, n,
            inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy);

    apply_no_slip_bc_kernel<dim, Number><<<blocks, 256, 0, stream>>>(
        U, bd.boundary_dofs, bd.boundary_ids, n);

    apply_slip_bc_kernel<dim, Number><<<blocks, 256, 0, stream>>>(
        U, bd.boundary_dofs, bd.boundary_ids, bd.boundary_normals, n);

    apply_dynamic_bc_kernel<dim, Number><<<blocks, 256, 0, stream>>>(
        U, bd.boundary_dofs, bd.boundary_ids, bd.boundary_normals,
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy, n);
}

#endif 
