#ifndef INTERNAL_ENERGY_SOLVER_CUH
#define INTERNAL_ENERGY_SOLVER_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

template<int dim, typename Number>
__global__ void apply_energy_mass_kernel(
    const Number* energy_in,
    Number* energy_out,
    const Number* density,
    const Number* lumped_mass,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    energy_out[i] = lumped_mass[i] * density[i] * energy_in[i];
}

template<int dim, typename Number, int nodes_per_elem = (dim == 2) ? 4 : 8>
__global__ void apply_energy_laplacian_kernel(
    const Number* energy_in,
    Number* energy_out,
    const int* element_connectivity,
    const Number* jacobian_data,
    Number kappa,
    Number tau,
    int n_elements)
{
    const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_elements) return;

    int nodes[nodes_per_elem];
    for (int n = 0; n < nodes_per_elem; ++n) {
        nodes[n] = element_connectivity[elem_id * nodes_per_elem + n];
    }

    Number e_elem[nodes_per_elem];
    for (int n = 0; n < nodes_per_elem; ++n) {
        e_elem[n] = energy_in[nodes[n]];
    }

    Number node_contrib[nodes_per_elem];
    for (int n = 0; n < nodes_per_elem; ++n) node_contrib[n] = Number(0);

    if constexpr (dim == 2) {
        constexpr int n_q_points = 4;
        constexpr int jac_data_per_quad = 8;
        constexpr double gp = 0.5773502691896258;
        const Number qpts[4][2] = {
            {Number(-gp), Number(-gp)},
            {Number( gp), Number(-gp)},
            {Number( gp), Number( gp)},
            {Number(-gp), Number( gp)}
        };

        for (int q = 0; q < n_q_points; ++q) {
            const Number xi  = qpts[q][0];
            const Number eta = qpts[q][1];

            const int jac_offset = elem_id * n_q_points * jac_data_per_quad + q * jac_data_per_quad;
            Number J_inv[2][2];
            J_inv[0][0] = jacobian_data[jac_offset + 4];
            J_inv[0][1] = jacobian_data[jac_offset + 5];
            J_inv[1][0] = jacobian_data[jac_offset + 6];
            J_inv[1][1] = jacobian_data[jac_offset + 7];
            const Number J00 = jacobian_data[jac_offset + 0];
            const Number J01 = jacobian_data[jac_offset + 1];
            const Number J10 = jacobian_data[jac_offset + 2];
            const Number J11 = jacobian_data[jac_offset + 3];
            const Number det_J = J00 * J11 - J01 * J10;
            const Number weight = fabs(det_J);

            Number grad_phi_ref[4][2];
            grad_phi_ref[0][0] = -Number(0.25) * (1 - eta);  grad_phi_ref[0][1] = -Number(0.25) * (1 - xi);
            grad_phi_ref[1][0] =  Number(0.25) * (1 - eta);  grad_phi_ref[1][1] = -Number(0.25) * (1 + xi);
            grad_phi_ref[2][0] = -Number(0.25) * (1 + eta);  grad_phi_ref[2][1] =  Number(0.25) * (1 - xi);
            grad_phi_ref[3][0] =  Number(0.25) * (1 + eta);  grad_phi_ref[3][1] =  Number(0.25) * (1 + xi);

            Number grad_phi[4][2];
            for (int n = 0; n < 4; ++n) {
                for (int i = 0; i < 2; ++i) {
                    grad_phi[n][i] = J_inv[0][i] * grad_phi_ref[n][0] +
                                     J_inv[1][i] * grad_phi_ref[n][1];
                }
            }

            Number grad_e[2] = {0, 0};
            for (int n = 0; n < 4; ++n) {
                for (int i = 0; i < 2; ++i) {
                    grad_e[i] += e_elem[n] * grad_phi[n][i];
                }
            }

            for (int n = 0; n < 4; ++n) {
                Number c = grad_phi[n][0] * grad_e[0] + grad_phi[n][1] * grad_e[1];
                node_contrib[n] += weight * c;
            }
        }
    }

    for (int n = 0; n < nodes_per_elem; ++n) {
        atomicAdd(&energy_out[nodes[n]], tau * kappa * node_contrib[n]);
    }
}

template<int dim, typename Number>
__global__ void energy_operator_bc_kernel(
    const Number* energy_in,
    Number* energy_out,
    const int* bc_type,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    if (bc_type[i] == 4) {
        energy_out[i] = energy_in[i];
    }
}

template<int dim, typename Number>
__global__ void apply_energy_operator_kernel(
    const Number* energy_in,
    Number* energy_out,
    const Number* density,
    const Number* lumped_mass,
    const int* cij_row_offsets,
    const int* cij_col_indices,
    const Number* cij_values,
    const Number* mij_values,
    const int* bc_type,
    Number kappa,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number rho_i = density[i];
    const Number m_i = lumped_mass[i];
    const Number e_i = energy_in[i];

    Number result = m_i * rho_i * e_i;

    const int bid_i = bc_type[i];
    if (bid_i == 4) {
        energy_out[i] = result;
        return;
    }

    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];

    for (int idx = row_start; idx < row_end; ++idx) {
        const int j = cij_col_indices[idx];
        const Number m_ij = mij_values[idx];
        const Number e_j = energy_in[j];

        Number c_ij_norm_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            const Number c_d = cij_values[idx * dim + d];
            c_ij_norm_sq += c_d * c_d;
        }

        result += tau * kappa * (c_ij_norm_sq / m_ij) * (e_i - e_j);
    }

    energy_out[i] = result;
}

template<int dim, typename Number>
__global__ void compute_energy_diagonal_preconditioner_kernel(
    Number* diagonal_inv,
    const Number* density,
    const Number* lumped_mass,
    const int* cij_row_offsets,
    const Number* cij_values,
    const Number* mij_values,
    Number kappa,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number rho_i = density[i];
    const Number m_i = lumped_mass[i];

    Number diag_value = m_i * rho_i;

    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];

    Number laplacian_diag = Number(0);
    for (int idx = row_start; idx < row_end; ++idx) {
        const Number m_ij = mij_values[idx];

        Number c_ij_norm_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            const Number c_d = cij_values[idx * dim + d];
            c_ij_norm_sq += c_d * c_d;
        }

        if (m_ij > Number(1e-30)) {
            laplacian_diag += c_ij_norm_sq / m_ij;
        }
    }

    diag_value += tau * kappa * laplacian_diag;

    diagonal_inv[i] = (diag_value > Number(1e-30)) ? Number(1) / diag_value : Number(0);
}

template<int dim, typename Number>
class InternalEnergySolver {
private:
    cudaStream_t stream;
    cublasHandle_t cublas_handle;

    const int n_dofs;

    Number* d_r;
    Number* d_z;
    Number* d_p;
    Number* d_Ap;
    Number* d_diag_inv;

    const Number* d_density;
    const Number* d_lumped_mass;
    const int* d_cij_row_offsets;
    const int* d_cij_col_indices;
    const Number* d_cij_values;
    const Number* d_mij_values;
    const int* d_bc_type;

    const int* d_element_nodes;
    const Number* d_jacobian_data;
    int n_elements;

    Number kappa;

public:
    InternalEnergySolver(int _n_dofs, cudaStream_t _stream = 0)
        : n_dofs(_n_dofs), stream(_stream)
    {
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream);

        cudaMalloc(&d_r, n_dofs * sizeof(Number));
        cudaMalloc(&d_z, n_dofs * sizeof(Number));
        cudaMalloc(&d_p, n_dofs * sizeof(Number));
        cudaMalloc(&d_Ap, n_dofs * sizeof(Number));
        cudaMalloc(&d_diag_inv, n_dofs * sizeof(Number));
    }

    ~InternalEnergySolver() {
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_Ap);
        cudaFree(d_diag_inv);
        cublasDestroy(cublas_handle);
    }

    void set_system_matrices(
        const Number* density,
        const Number* lumped_mass,
        const int* cij_row_offsets,
        const int* cij_col_indices,
        const Number* cij_values,
        const Number* mij_values,
        Number _kappa)
    {
        d_density = density;
        d_lumped_mass = lumped_mass;
        d_cij_row_offsets = cij_row_offsets;
        d_cij_col_indices = cij_col_indices;
        d_cij_values = cij_values;
        d_mij_values = mij_values;
        kappa = _kappa;
    }

    void set_bc_type(const int* bc_type) {
        d_bc_type = bc_type;
    }

    void set_element_data(const int* element_nodes, const Number* jacobian_data, int _n_elements) {
        d_element_nodes = element_nodes;
        d_jacobian_data = jacobian_data;
        n_elements = _n_elements;
    }

    void apply_operator(Number* dst, const Number* src, Number tau) {
        const int threads = 256;
        const int blocks_dofs = (n_dofs + threads - 1) / threads;
        const int blocks_elem = (n_elements + threads - 1) / threads;

        apply_energy_mass_kernel<dim, Number><<<blocks_dofs, threads, 0, stream>>>(
            src, dst, d_density, d_lumped_mass, n_dofs);

        if (kappa != Number(0)) {
            apply_energy_laplacian_kernel<dim, Number><<<blocks_elem, threads, 0, stream>>>(
                src, dst, d_element_nodes, d_jacobian_data, kappa, tau, n_elements);
        }

        energy_operator_bc_kernel<dim, Number><<<blocks_dofs, threads, 0, stream>>>(
            src, dst, d_bc_type, n_dofs);
    }

    int solve(Number* energy_solution,
              const Number* energy_rhs,
              Number tau,
              Number tolerance,
              int max_iterations)
    {
        const int threads = 256;
        const int blocks = (n_dofs + threads - 1) / threads;

        Number rhs_norm;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSnrm2(cublas_handle, n_dofs, energy_rhs, 1, &rhs_norm);
        } else {
            cublasDnrm2(cublas_handle, n_dofs, energy_rhs, 1, &rhs_norm);
        }

        Number init_norm;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSnrm2(cublas_handle, n_dofs, energy_solution, 1, &init_norm);
        } else {
            cublasDnrm2(cublas_handle, n_dofs, energy_solution, 1, &init_norm);
        }

        compute_energy_diagonal_preconditioner_kernel<dim><<<blocks, threads, 0, stream>>>(
            d_diag_inv, d_density, d_lumped_mass, d_cij_row_offsets,
            d_cij_values, d_mij_values, kappa, tau, n_dofs);

        cudaMemcpyAsync(d_r, energy_rhs, n_dofs * sizeof(Number),
                       cudaMemcpyDeviceToDevice, stream);

        apply_operator(d_Ap, energy_solution, tau);

        Number alpha = Number(-1);
        if constexpr (std::is_same_v<Number, float>) {
            cublasSaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
        } else {
            cublasDaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
        }

        Number r_norm_init;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm_init);
        } else {
            cublasDnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm_init);
        }

        const Number abs_tol = std::is_same_v<Number, float> ? Number(1e-8) : Number(1e-15);

        if (r_norm_init < abs_tol) {
            return 0;
        }

        const Number conv_target = (tolerance * r_norm_init < abs_tol) ? abs_tol : tolerance * r_norm_init;

        apply_diagonal_preconditioner_kernel<<<blocks, threads, 0, stream>>>(
            d_diag_inv, d_r, d_z, n_dofs);

        cudaMemcpyAsync(d_p, d_z, n_dofs * sizeof(Number),
                       cudaMemcpyDeviceToDevice, stream);

        Number rho;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
        } else {
            cublasDdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
        }

        for (int iter = 0; iter < max_iterations; ++iter) {

            apply_operator(d_Ap, d_p, tau);

            Number pAp;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, n_dofs, d_p, 1, d_Ap, 1, &pAp);
            } else {
                cublasDdot(cublas_handle, n_dofs, d_p, 1, d_Ap, 1, &pAp);
            }

            alpha = rho / pAp;

            if constexpr (std::is_same_v<Number, float>) {
                cublasSaxpy(cublas_handle, n_dofs, &alpha, d_p, 1, energy_solution, 1);
            } else {
                cublasDaxpy(cublas_handle, n_dofs, &alpha, d_p, 1, energy_solution, 1);
            }

            alpha = -alpha;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
            } else {
                cublasDaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
            }

            Number r_norm;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm);
            } else {
                cublasDnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm);
            }

            if (r_norm < conv_target) {
                return iter + 1;
            }

            apply_diagonal_preconditioner_kernel<<<blocks, threads, 0, stream>>>(
                d_diag_inv, d_r, d_z, n_dofs);

            Number rho_old = rho;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
            } else {
                cublasDdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
            }

            Number beta = rho / rho_old;

            if constexpr (std::is_same_v<Number, float>) {
                cublasSscal(cublas_handle, n_dofs, &beta, d_p, 1);
                alpha = Number(1);
                cublasSaxpy(cublas_handle, n_dofs, &alpha, d_z, 1, d_p, 1);
            } else {
                cublasDscal(cublas_handle, n_dofs, &beta, d_p, 1);
                alpha = Number(1);
                cublasDaxpy(cublas_handle, n_dofs, &alpha, d_z, 1, d_p, 1);
            }
        }

        return -1;
    }
};

#endif