#ifndef VELOCITY_SOLVER_CUH
#define VELOCITY_SOLVER_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>


template<int dim, typename Number>
__global__ void apply_velocity_mass_kernel(
    const Number* velocity_in,
    Number* velocity_out,
    const Number* density,
    const Number* lumped_mass_matrix,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const Number m_rho = lumped_mass_matrix[i] * density[i];
    for (int d = 0; d < dim; ++d) {
        velocity_out[i * dim + d] = m_rho * velocity_in[i * dim + d];
    }
}

template<int dim, typename Number, int nodes_per_elem = (dim == 2) ? 4 : 8>
__global__ void apply_velocity_stress_kernel(
    const Number* velocity_in,
    Number* velocity_out,
    const int* element_connectivity,
    const Number* jacobian_data,
    Number mu,
    Number lambda,
    Number tau,
    int n_elements)
{
    const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_elements) return;

    int nodes[nodes_per_elem];
    for (int n = 0; n < nodes_per_elem; ++n) {
        nodes[n] = element_connectivity[elem_id * nodes_per_elem + n];
    }

    Number V_elem[nodes_per_elem][dim];
    for (int n = 0; n < nodes_per_elem; ++n) {
        for (int d = 0; d < dim; ++d) {
            V_elem[n][d] = velocity_in[nodes[n] * dim + d];
        }
    }

    Number node_contrib[nodes_per_elem][dim];
    for (int n = 0; n < nodes_per_elem; ++n)
        for (int d = 0; d < dim; ++d)
            node_contrib[n][d] = Number(0);

    if constexpr (dim == 2) {
        constexpr int n_q_points = 4;
        constexpr int jac_data_per_quad = 8;
        const Number two_mu = Number(2) * mu;
        const Number lambda_bar = lambda - Number(2.0/3.0) * mu;

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

            Number grad_V[2][2] = {{0,0},{0,0}};
            for (int n = 0; n < 4; ++n) {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        grad_V[i][j] += V_elem[n][i] * grad_phi[n][j];
                    }
                }
            }

            Number eps[2][2];
            eps[0][0] = grad_V[0][0];
            eps[1][1] = grad_V[1][1];
            eps[0][1] = Number(0.5) * (grad_V[0][1] + grad_V[1][0]);
            eps[1][0] = eps[0][1];

            const Number div_V = eps[0][0] + eps[1][1];

            Number S[2][2];
            S[0][0] = two_mu * eps[0][0] + lambda_bar * div_V;
            S[1][1] = two_mu * eps[1][1] + lambda_bar * div_V;
            S[0][1] = two_mu * eps[0][1];
            S[1][0] = S[0][1];

            for (int n = 0; n < 4; ++n) {
                for (int i = 0; i < 2; ++i) {
                    Number c = Number(0);
                    for (int j = 0; j < 2; ++j) {
                        c += S[i][j] * grad_phi[n][j];
                    }
                    node_contrib[n][i] += weight * c;
                }
            }
        }
    } else {

    }

    for (int n = 0; n < nodes_per_elem; ++n) {
        for (int d = 0; d < dim; ++d) {
            atomicAdd(&velocity_out[nodes[n] * dim + d], tau * node_contrib[n][d]);
        }
    }
}

template<int dim, typename Number>
__global__ void velocity_operator_bc_kernel(
    const Number* velocity_in,
    Number* velocity_out,
    const int* bc_type,
    const int* bc_index,
    const Number* boundary_normals,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const int id = bc_type[i];

    if (id == 4) {
        for (int d = 0; d < dim; ++d) {
            velocity_out[i * dim + d] = velocity_in[i * dim + d];
        }
    }
    else if (id == 3) {
        for (int d = 0; d < dim; ++d) {
            velocity_out[i * dim + d] = velocity_in[i * dim + d];
        }
    }
    else if (id == 2) {
        const int b = bc_index[i];
        Number n_vec[dim];
        for (int d = 0; d < dim; ++d) n_vec[d] = boundary_normals[b * dim + d];

        Number dst_n = Number(0), src_n = Number(0);
        for (int d = 0; d < dim; ++d) {
            dst_n += velocity_out[i * dim + d] * n_vec[d];
            src_n += velocity_in[i * dim + d] * n_vec[d];
        }
        for (int d = 0; d < dim; ++d) {
            velocity_out[i * dim + d] += (src_n - dst_n) * n_vec[d];
        }
    }
}

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
    const int* bc_type,
    const int* bc_index,
    const Number* boundary_normals,
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

    const int bid_i = bc_type[i];

    if (bid_i == 4) {
        for (int d = 0; d < dim; ++d) {
            velocity_out[i * dim + d] = result[d];
        }
        return;
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
            dV[d] = V_i[d] - V_j[d];
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

    if (bid_i == 2) {
        const int b = bc_index[i];
        Number normal[dim];
        for (int d = 0; d < dim; ++d) {
            normal[d] = boundary_normals[b * dim + d];
        }
        Number res_n = Number(0);
        for (int d = 0; d < dim; ++d) {
            res_n += result[d] * normal[d];
        }
        for (int d = 0; d < dim; ++d) {
            result[d] -= res_n * normal[d];

        }

        Number V_n = Number(0);
        for (int d = 0; d < dim; ++d) {
            V_n += V_i[d] * normal[d];
        }
        for (int d = 0; d < dim; ++d) {
            result[d] += m_i * rho_i * V_n * normal[d];
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

    const Number inv = (diag_value > Number(1e-30)) ? Number(1) / diag_value : Number(0);
    for (int d = 0; d < dim; ++d) {
        diagonal_inv[i * dim + d] = inv;
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

    const int* d_bc_type;
    const int* d_bc_index;
    const Number* d_boundary_normals;

    const int* d_element_nodes;
    const Number* d_jacobian_data;
    int n_elements;

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

    void set_boundary_data(
        const int* bc_type,
        const int* bc_index,
        const Number* boundary_normals)
    {
        d_bc_type = bc_type;
        d_bc_index = bc_index;
        d_boundary_normals = boundary_normals;
    }

    void set_element_data(
        const int* element_nodes,
        const Number* jacobian_data,
        int _n_elements)
    {
        d_element_nodes = element_nodes;
        d_jacobian_data = jacobian_data;
        n_elements = _n_elements;
    }

    void apply_operator(Number* dst, const Number* src, Number tau)
    {
        const int threads = 256;
        const int blocks_dofs = (n_dofs + threads - 1) / threads;
        const int blocks_elem = (n_elements + threads - 1) / threads;

        apply_velocity_mass_kernel<dim, Number><<<blocks_dofs, threads, 0, stream>>>(
            src, dst, d_density, d_lumped_mass_matrix, n_dofs);

        if (mu != Number(0) || lambda != Number(0)) {
            apply_velocity_stress_kernel<dim, Number><<<blocks_elem, threads, 0, stream>>>(
                src, dst, d_element_nodes, d_jacobian_data,
                mu, lambda, tau, n_elements);
        }

        velocity_operator_bc_kernel<dim, Number><<<blocks_dofs, threads, 0, stream>>>(
            src, dst, d_bc_type, d_bc_index, d_boundary_normals, n_dofs);
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

        apply_operator(d_Ap, velocity_solution, tau);

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

        const Number abs_tol = std::is_same_v<Number, float> ? Number(1e-8) : Number(1e-15);

        if (r_norm_init < abs_tol) {
            return 0;
        }

        const Number conv_target = (tolerance * r_norm_init < abs_tol) ? abs_tol : tolerance * r_norm_init;

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

            apply_operator(d_Ap, d_p, tau);

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

            if (r_norm < conv_target) {
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

