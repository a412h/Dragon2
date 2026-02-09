#ifndef BOUNDARY_CONDITIONS_CUH
#define BOUNDARY_CONDITIONS_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"

template<int dim, typename Number, int component>
__device__ void prescribe_riemann_characteristic(
    const State<dim, Number>& U,
    int idx,
    Number rho_bar,
    Number momentum_bar[dim],
    Number energy_bar,
    const Number normal[dim],
    Number result[dim + 2])
{
    static_assert(component == 1 || component == 2, "component must be 1 or 2");

    using PF = PhysicsFunctions<dim, Number>;

    const Number rho = U.rho[idx];
    Number m[dim];
    m[0] = U.momentum_x[idx];
    if constexpr (dim >= 2) m[1] = U.momentum_y[idx];
    if constexpr (dim == 3) m[2] = U.momentum_z[idx];
    const Number E = U.energy[idx];

    const Number rho_inv = Number(1.0) / rho;
    const Number p = PF::pressure(U, idx);
    const Number a = PF::speed_of_sound(U, idx);

    Number vn = Number(0);
    for (int d = 0; d < dim; ++d) {
        vn += m[d] * normal[d];
    }
    vn *= rho_inv;

    const Number rho_bar_inv = Number(1.0) / rho_bar;
    Number m_sq_bar = momentum_bar[0] * momentum_bar[0];
    if constexpr (dim >= 2) m_sq_bar += momentum_bar[1] * momentum_bar[1];
    if constexpr (dim == 3) m_sq_bar += momentum_bar[2] * momentum_bar[2];
    const Number e_bar = energy_bar - Number(0.5) * m_sq_bar * rho_bar_inv;
    const Number p_bar = PF::gamma_minus_one * e_bar;
    const Number a_bar = sqrt(PF::gamma * p_bar * rho_bar_inv);

    Number vn_bar = Number(0);
    for (int d = 0; d < dim; ++d) {
        vn_bar += momentum_bar[d] * normal[d];
    }
    vn_bar *= rho_bar_inv;

    const Number R_1 = (component == 1)
        ? (vn_bar - Number(2.0) * a_bar / PF::gamma_minus_one)
        : (vn - Number(2.0) * a / PF::gamma_minus_one);

    const Number R_2 = (component == 2)
        ? (vn_bar + Number(2.0) * a_bar / PF::gamma_minus_one)
        : (vn + Number(2.0) * a / PF::gamma_minus_one);

    const Number s = p / pow(rho, PF::gamma);

    Number v_perp[dim];
    for (int d = 0; d < dim; ++d) {
        v_perp[d] = m[d] * rho_inv - vn * normal[d];
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
    using PF = PhysicsFunctions<dim, Number>;

    for (int b = 0; b < boundary_data.n_boundary_dofs; ++b) {
        if (boundary_data.boundary_dofs[b] == idx) {
            const int bid = boundary_data.boundary_ids[b];

            if (bid == 4) {
                U.rho[idx] = inflow_rho;
                U.momentum_x[idx] = inflow_momentum_x;
                if constexpr (dim >= 2) U.momentum_y[idx] = inflow_momentum_y;
                if constexpr (dim == 3) U.momentum_z[idx] = inflow_momentum_z;
                U.energy[idx] = inflow_energy;
            }
            else if (bid == 2) {
                Number normal[dim];
                for (int d = 0; d < dim; ++d) {
                    normal[d] = boundary_data.boundary_normals[b * dim + d];
                }

                Number momentum[dim];
                momentum[0] = U.momentum_x[idx];
                if constexpr (dim >= 2) momentum[1] = U.momentum_y[idx];
                if constexpr (dim == 3) momentum[2] = U.momentum_z[idx];

                Number m_dot_n = Number(0);
                for (int d = 0; d < dim; ++d) {
                    m_dot_n += momentum[d] * normal[d];
                }

                U.momentum_x[idx] = momentum[0] - m_dot_n * normal[0];
                if constexpr (dim >= 2) U.momentum_y[idx] = momentum[1] - m_dot_n * normal[1];
                if constexpr (dim == 3) U.momentum_z[idx] = momentum[2] - m_dot_n * normal[2];
            }
            else if (bid == 3) {
                U.momentum_x[idx] = Number(0);
                if constexpr (dim >= 2) U.momentum_y[idx] = Number(0);
                if constexpr (dim == 3) U.momentum_z[idx] = Number(0);
            }
            else if (bid == 5) {
                Number normal[dim];
                for (int d = 0; d < dim; ++d) {
                    normal[d] = boundary_data.boundary_normals[b * dim + d];
                }

                Number m[dim];
                m[0] = U.momentum_x[idx];
                if constexpr (dim >= 2) m[1] = U.momentum_y[idx];
                if constexpr (dim == 3) m[2] = U.momentum_z[idx];

                const Number rho = U.rho[idx];
                const Number rho_inv = Number(1.0) / rho;
                const Number a = PF::speed_of_sound(U, idx);

                Number vn = Number(0);
                for (int d = 0; d < dim; ++d) {
                    vn += m[d] * normal[d];
                }
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
                }
                else if (vn >= -a && vn <= Number(0)) {
                    prescribe_riemann_characteristic<dim, Number, 2>(
                        U, idx, inflow_rho, momentum_bar, inflow_energy, normal, result);

                    U.rho[idx] = result[0];
                    U.momentum_x[idx] = result[1];
                    if constexpr (dim >= 2) U.momentum_y[idx] = result[2];
                    if constexpr (dim == 3) U.momentum_z[idx] = result[3];
                    U.energy[idx] = result[dim + 1];
                }
                else if (vn > Number(0) && vn <= a) {
                    prescribe_riemann_characteristic<dim, Number, 1>(
                        U, idx, inflow_rho, momentum_bar, inflow_energy, normal, result);

                    U.rho[idx] = result[0];
                    U.momentum_x[idx] = result[1];
                    if constexpr (dim >= 2) U.momentum_y[idx] = result[2];
                    if constexpr (dim == 3) U.momentum_z[idx] = result[3];
                    U.energy[idx] = result[dim + 1];
                }
            }
        }
    }
}

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

template<int dim, typename Number>
__global__ void prepare_state_kernel(
    State<dim, Number> d_U,
    Number* d_precomputed,
    const BoundaryData<dim, Number> boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_dofs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_dofs) return;

    apply_boundary_conditions_device<dim, Number>(
        d_U, boundary_data,
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        idx);

    __syncthreads();

    compute_precomputed_values_device<dim, Number>(d_U, d_precomputed, idx);
}

template<int dim, typename Number>
void prepare_state(
    State<dim, Number>& d_U,
    Number* d_precomputed,
    const BoundaryData<dim, Number>& d_boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_dofs,
    cudaStream_t stream)
{
    const int blockSize = 256;
    const int numBlocks = (n_dofs + blockSize - 1) / blockSize;

    prepare_state_kernel<dim, Number><<<numBlocks, blockSize, 0, stream>>>(
        d_U, d_precomputed, d_boundary_data,
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        n_dofs);
}

template void prepare_state<2, float>(State<2, float>&, float*, const BoundaryData<2, float>&,
    float, float, float, float, float, int, cudaStream_t);
template void prepare_state<2, double>(State<2, double>&, double*, const BoundaryData<2, double>&,
    double, double, double, double, double, int, cudaStream_t);
template void prepare_state<3, float>(State<3, float>&, float*, const BoundaryData<3, float>&,
    float, float, float, float, float, int, cudaStream_t);
template void prepare_state<3, double>(State<3, double>&, double*, const BoundaryData<3, double>&,
    double, double, double, double, double, int, cudaStream_t);

#endif
