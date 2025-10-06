#ifndef INTERNAL_ENERGY_SOLVER_CUH
#define INTERNAL_ENERGY_SOLVER_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

// Implements: (m_i * rho_i + tau * kappa * L_ij) * e
// where L_ij is the scalar Laplacian operator
template<int dim, typename Number>
__global__ void apply_energy_operator_kernel(
    const Number* energy_in,         // Input internal energy [n_dofs]
    Number* energy_out,              // Output [n_dofs]
    const Number* density,           // Density at each node
    const Number* lumped_mass,       // Lumped mass matrix diagonal
    const int* cij_row_offsets,      // CSR row offsets for C_ij
    const int* cij_col_indices,      // CSR column indices for C_ij
    const Number* cij_values,        // C_ij matrix values [nnz * dim]
    const Number* mij_values,        // M_ij matrix values [nnz]
    Number kappa,                    // Thermal diffusivity
    Number tau,                      // Time step
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const Number rho_i = density[i];
    const Number m_i = lumped_mass[i];
    const Number e_i = energy_in[i];
    
    // Initialize result with mass matrix contribution: m_i * rho_i * e_i
    Number result = m_i * rho_i * e_i;
    
    // Add thermal diffusion contribution: tau * kappa * L * e
    // L_ij = sum_j c_ij . c_ij^T / m_ij (scalar Laplacian)
    
    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];
    
    for (int idx = row_start; idx < row_end; ++idx) {
        const int j = cij_col_indices[idx];
        const Number m_ij = mij_values[idx];
        const Number e_j = energy_in[j];
        
        // Compute c_ij . c_ij^T (scalar product)
        Number c_ij_norm_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            const Number c_d = cij_values[idx * dim + d];
            c_ij_norm_sq += c_d * c_d;
        }
        
        // Apply Laplacian: sum_j (c_ij . c_ij^T / m_ij) * (e_j - e_i)
        result += tau * kappa * (c_ij_norm_sq / m_ij) * (e_j - e_i);
    }
    
    energy_out[i] = result;
}

// Diagonal preconditioner for energy system
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
    
    // Diagonal from mass matrix
    Number diag_value = m_i * rho_i;
    
    // Add diagonal contribution from Laplacian
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
        
        // Diagonal gets negative contribution from Laplacian
        laplacian_diag -= c_ij_norm_sq / m_ij;
    }
    
    diag_value += tau * kappa * laplacian_diag;
    
    // Store inverse
    diagonal_inv[i] = Number(1) / diag_value;
}

// PCG solver for internal energy system
template<int dim, typename Number>
class InternalEnergySolver {
private:
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    
    const int n_dofs;
    
    // Working vectors for PCG
    Number* d_r;        // Residual
    Number* d_z;        // Preconditioned residual
    Number* d_p;        // Search direction
    Number* d_Ap;       // A * p
    Number* d_diag_inv; // Diagonal preconditioner
    
    // System matrices (references to external data)
    const Number* d_density;
    const Number* d_lumped_mass;
    const int* d_cij_row_offsets;
    const int* d_cij_col_indices;
    const Number* d_cij_values;
    const Number* d_mij_values;
    
    Number kappa;
    
public:
    InternalEnergySolver(int _n_dofs, cudaStream_t _stream = 0)
        : n_dofs(_n_dofs), stream(_stream)
    {
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream);
        
        // Allocate working vectors
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
    
    int solve(Number* energy_solution, 
              const Number* energy_rhs, 
              Number tau,
              Number tolerance, 
              int max_iterations)
    {
        const int threads = 256;
        const int blocks = (n_dofs + threads - 1) / threads;
        
        // Diagnostic - compute norms on GPU
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
        
        // Build diagonal preconditioner
        compute_energy_diagonal_preconditioner_kernel<dim><<<blocks, threads, 0, stream>>>(
            d_diag_inv, d_density, d_lumped_mass, d_cij_row_offsets,
            d_cij_values, d_mij_values, kappa, tau, n_dofs);
        
        // r = r - A*x
        Number alpha = Number(-1);
        if constexpr (std::is_same_v<Number, float>) {
            cublasSaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
        } else {
            cublasDaxpy(cublas_handle, n_dofs, &alpha, d_Ap, 1, d_r, 1);
        }
        
        // Check initial residual
        Number r_norm_init;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm_init);
        } else {
            cublasDnrm2(cublas_handle, n_dofs, d_r, 1, &r_norm_init);
        }
        
        if (r_norm_init < tolerance) {
            return 0;
        }
        
        // Apply preconditioner: z = M^{-1} * r
        apply_diagonal_preconditioner_kernel<<<blocks, threads, 0, stream>>>(
            d_diag_inv, d_r, d_z, n_dofs);
        
        // p = z
        cudaMemcpyAsync(d_p, d_z, n_dofs * sizeof(Number), 
                       cudaMemcpyDeviceToDevice, stream);
        
        Number rho;
        if constexpr (std::is_same_v<Number, float>) {
            cublasSdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
        } else {
            cublasDdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
        }
        
        // PCG iterations
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Ap = A * p
            apply_energy_operator_kernel<dim><<<blocks, threads, 0, stream>>>(
                d_p, d_Ap, d_density, d_lumped_mass,
                d_cij_row_offsets, d_cij_col_indices, d_cij_values, d_mij_values,
                kappa, tau, n_dofs);
            
            Number pAp;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, n_dofs, d_p, 1, d_Ap, 1, &pAp);
            } else {
                cublasDdot(cublas_handle, n_dofs, d_p, 1, d_Ap, 1, &pAp);
            }
            
            alpha = rho / pAp;
            
            // x = x + alpha * p
            if constexpr (std::is_same_v<Number, float>) {
                cublasSaxpy(cublas_handle, n_dofs, &alpha, d_p, 1, energy_solution, 1);
            } else {
                cublasDaxpy(cublas_handle, n_dofs, &alpha, d_p, 1, energy_solution, 1);
            }
            
            // r = r - alpha * Ap
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
            
            if (r_norm < tolerance * r_norm_init) {
                return iter + 1;
            }
            
            // z = M^{-1} * r
            apply_diagonal_preconditioner_kernel<<<blocks, threads, 0, stream>>>(
                d_diag_inv, d_r, d_z, n_dofs);
            
            Number rho_old = rho;
            if constexpr (std::is_same_v<Number, float>) {
                cublasSdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
            } else {
                cublasDdot(cublas_handle, n_dofs, d_r, 1, d_z, 1, &rho);
            }
            
            Number beta = rho / rho_old;
            
            // p = z + beta * p
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
        
        return -1;  // Failed to converge
    }
};

#endif