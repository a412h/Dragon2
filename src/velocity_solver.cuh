#ifndef VELOCITY_SOLVER_CUH
#define VELOCITY_SOLVER_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>














template<int dim, typename Number>
__global__ void apply_velocity_operator_kernel(
    const Number* velocity_in,
    Number* velocity_out,
    const Number* density,
    const Number* lumped_mass_matrix,
    const int* cij_row_offsets,
    const int* cij_col_indices,
    const Number* cij_values,
    const Number* mij_values,
    Number mu,
    Number lambda,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number rho_i = density[i];
    const Number m_i = lumped_mass_matrix[i];

    Number V_i[dim];
    for (int d = 0; d < dim; ++d) {
        V_i[d] = velocity_in[i * dim + d];
    }

    Number result[dim];
    for (int d = 0; d < dim; ++d) {
        result[d] = m_i * rho_i * V_i[d];
    }


    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];

    const Number lambda_bar = lambda - Number(2.0 / 3.0) * mu;
    const Number two_mu = Number(2.0) * mu;

    for (int idx = row_start; idx < row_end; ++idx) {
        const int j = cij_col_indices[idx];
        const Number m_ij = mij_values[idx];

        Number c_ij[dim];
        for (int d = 0; d < dim; ++d) {
            c_ij[d] = cij_values[idx * dim + d];
        }

        Number V_j[dim];
        for (int d = 0; d < dim; ++d) {
            V_j[d] = velocity_in[j * dim + d];
        }

        Number dV[dim];
        for (int d = 0; d < dim; ++d) {
            dV[d] = V_j[d] - V_i[d];
        }

        Number c_dot_c = Number(0);
        for (int d = 0; d < dim; ++d) {
            c_dot_c += c_ij[d] * c_ij[d];
        }


        for (int d = 0; d < dim; ++d) {
            Number contrib = Number(0);

            for (int e = 0; e < dim; ++e) {
                Number S_de = two_mu * m_ij * c_ij[d] * c_ij[e];

                if (d == e) {
                    S_de += lambda_bar * m_ij * c_dot_c;
                }

                contrib += S_de * dV[e];
            }

            result[d] += tau * contrib;
        }
    }

    for (int d = 0; d < dim; ++d) {
        velocity_out[i * dim + d] = result[d];
    }
}

template<int dim, typename Number>
__global__ void compute_diagonal_preconditioner_kernel(
    Number* diagonal_inv,
    const Number* density,
    const Number* lumped_mass_matrix,
    const int* cij_row_offsets,
    const Number* cij_values,
    Number mu,
    Number lambda,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number rho_i = density[i];
    const Number m_i = lumped_mass_matrix[i];

    Number diag_value = m_i * rho_i;

    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];

    Number visc_diag = Number(0);
    for (int idx = row_start; idx < row_end; ++idx) {
        for (int d = 0; d < dim; ++d) {
            Number c_d = cij_values[idx * dim + d];
            visc_diag += c_d * c_d;
        }
    }

    diag_value += tau * (Number(2) * mu + lambda) * visc_diag;

    for (int d = 0; d < dim; ++d) {
        diagonal_inv[i * dim + d] = Number(1) / diag_value;
    }
}

template<typename Number>
__global__ void apply_diagonal_preconditioner_kernel(
    const Number* diagonal_inv,
    const Number* residual,
    Number* z,
    int vector_size)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vector_size) {
        z[i] = residual[i] * diagonal_inv[i];
    }
}

template<int dim, typename Number>
class VelocitySolver {
private:
    cudaStream_t stream;
    cublasHandle_t cublas_handle;

    const int n_dofs;
    const int vector_size;

    Number* d_r;
    Number* d_z;
    Number* d_p;
    Number* d_Ap;
    Number* d_diag_inv;

    const Number* d_density;
    const Number* d_lumped_mass_matrix;
    const int* d_cij_row_offsets;
    const int* d_cij_col_indices;
    const Number* d_cij_values;
    const Number* d_mij_values;

    Number mu;
    Number lambda;

public:
    VelocitySolver(int _n_dofs, cudaStream_t _stream = 0)
        : n_dofs(_n_dofs),
          vector_size(_n_dofs * dim),
          stream(_stream)
    {
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream);

        cudaMalloc(&d_r, vector_size * sizeof(Number));
        cudaMalloc(&d_z, vector_size * sizeof(Number));
        cudaMalloc(&d_p, vector_size * sizeof(Number));
        cudaMalloc(&d_Ap, vector_size * sizeof(Number));
        cudaMalloc(&d_diag_inv, vector_size * sizeof(Number));
    }

    ~VelocitySolver() {
        cudaFree(d_r);
        cudaFree(d_z);
        cudaFree(d_p);
        cudaFree(d_Ap);
        cudaFree(d_diag_inv);
        cublasDestroy(cublas_handle);
    }

    void set_system_matrices(
        const Number* density,
        const Number* lumped_mass_matrix,
        const int* cij_row_offsets,
        const int* cij_col_indices,
        const Number* cij_values,
        const Number* mij_values,
        Number _mu,
        Number _lambda)
    {
        d_density = density;
        d_lumped_mass_matrix = lumped_mass_matrix;
        d_cij_row_offsets = cij_row_offsets;
        d_cij_col_indices = cij_col_indices;
        d_cij_values = cij_values;
        d_mij_values = mij_values;
        mu = _mu;
        lambda = _lambda;
    }

    int solve(Number* velocity_solution,
              const Number* velocity_rhs,
              Number tau,
              Number tolerance,
              int max_iterations)
    {
        const int threads = 256;
        const int blocks = (n_dofs + threads - 1) / threads;

        compute_diagonal_preconditioner_kernel<dim><<<blocks, threads, 0, stream>>>(
            d_diag_inv, d_density, d_lumped_mass_matrix, d_cij_row_offsets,
            d_cij_values, mu, lambda, tau, n_dofs);

        cudaMemcpyAsync(d_r, velocity_rhs, vector_size * sizeof(Number),
                       cudaMemcpyDeviceToDevice, stream);

        apply_velocity_operator_kernel<dim><<<blocks, threads, 0, stream>>>(
            velocity_solution, d_Ap, d_density, d_lumped_mass_matrix,
            d_cij_row_offsets, d_cij_col_indices, d_cij_values, d_mij_values,
            mu, lambda, tau, n_dofs);

        Number alpha = Number(-1);
        if constexpr (std::is_same_v<Number, float>) {
            cublasSaxpy(cublas_handle, vector_size, &alpha, d_Ap, 1, d_r, 1);
        } else {
            cublasDaxpy(cublas_handle, vector_size, &alpha, d_Ap, 1, d_r, 1);
        }

        Number r_norm_init;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSnrm2(cublas_handle, vector_size, d_r, 1, &r_norm_init);
        } else {
            cublasDnrm2(cublas_handle, vector_size, d_r, 1, &r_norm_init);
        }

        if (r_norm_init < tolerance) {
            return 0;
        }

        apply_diagonal_preconditioner_kernel<<<blocks * dim, threads, 0, stream>>>(
            d_diag_inv, d_r, d_z, vector_size);

        cudaMemcpyAsync(d_p, d_z, vector_size * sizeof(Number),
                       cudaMemcpyDeviceToDevice, stream);

        Number rho;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSdot(cublas_handle, vector_size, d_r, 1, d_z, 1, &rho);
        } else {
            cublasDdot(cublas_handle, vector_size, d_r, 1, d_z, 1, &rho);
        }

        for (int iter = 0; iter < max_iterations; ++iter) {
            apply_velocity_operator_kernel<dim><<<blocks, threads, 0, stream>>>(
                d_p, d_Ap, d_density, d_lumped_mass_matrix,
                d_cij_row_offsets, d_cij_col_indices, d_cij_values, d_mij_values,
                mu, lambda, tau, n_dofs);

            Number pAp;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, vector_size, d_p, 1, d_Ap, 1, &pAp);
            } else {
                cublasDdot(cublas_handle, vector_size, d_p, 1, d_Ap, 1, &pAp);
            }

            alpha = rho / pAp;

            if constexpr (std::is_same_v<Number, float>) {
                cublasSaxpy(cublas_handle, vector_size, &alpha, d_p, 1, velocity_solution, 1);
            } else {
                cublasDaxpy(cublas_handle, vector_size, &alpha, d_p, 1, velocity_solution, 1);
            }

            alpha = -alpha;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSaxpy(cublas_handle, vector_size, &alpha, d_Ap, 1, d_r, 1);
            } else {
                cublasDaxpy(cublas_handle, vector_size, &alpha, d_Ap, 1, d_r, 1);
            }

            Number r_norm;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSnrm2(cublas_handle, vector_size, d_r, 1, &r_norm);
            } else {
                cublasDnrm2(cublas_handle, vector_size, d_r, 1, &r_norm);
            }

            if (r_norm < tolerance * r_norm_init) {
                return iter + 1;
            }

            apply_diagonal_preconditioner_kernel<<<blocks * dim, threads, 0, stream>>>(
                d_diag_inv, d_r, d_z, vector_size);

            Number rho_old = rho;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, vector_size, d_r, 1, d_z, 1, &rho);
            } else {
                cublasDdot(cublas_handle, vector_size, d_r, 1, d_z, 1, &rho);
            }

            Number beta = rho / rho_old;

            if constexpr (std::is_same_v<Number, float>) {
                cublasSscal(cublas_handle, vector_size, &beta, d_p, 1);
                alpha = Number(1);
                cublasSaxpy(cublas_handle, vector_size, &alpha, d_z, 1, d_p, 1);
            } else {
                cublasDscal(cublas_handle, vector_size, &beta, d_p, 1);
                alpha = Number(1);
                cublasDaxpy(cublas_handle, vector_size, &alpha, d_z, 1, d_p, 1);
            }
        }

        return -1;
    }
};

#endif


