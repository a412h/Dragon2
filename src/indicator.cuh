#ifndef INDICATOR_CUH
#define INDICATOR_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"

template<int dim, typename Number>
struct Indicator {
    using FluxType = Flux<dim, Number>;
    using PF = PhysicsFunctions<dim, Number>;
    
    // Member variables that can be reset/accumulated
    Number rho_i_inverse;
    Number eta_i;
    FluxType f_i;
    Number d_eta_i[dim + 2];
    Number left;
    Number right[dim + 2];
    static constexpr int problem_dimension = dim + 2;
    
    __device__
    void reset(int i,
               const Number U_i_local[dim+2],
               const Number* precomputed)
    {
        const Number s_i = precomputed[i * 2];
        
        const Number rho_i = U_i_local[0];
        rho_i_inverse = Number(1.) / rho_i;
        eta_i = precomputed[i * 2 + 1];
        
        // Compute Harten entropy derivative
        Number d_eta_tmp[dim + 2];
        PF::harten_entropy_derivative_local(U_i_local, d_eta_tmp);
        d_eta_i[0] = d_eta_tmp[0] - eta_i * rho_i_inverse;
        for (int k = 1; k < problem_dimension; ++k) {
            d_eta_i[k] = d_eta_tmp[k];
        }
        
        f_i = PF::f_local(U_i_local);
        
        // Initialize accumulators
        left = Number(0.);
        for (int k = 0; k < problem_dimension; ++k) {
            right[k] = Number(0.);
        }
    }
    
    __device__
    void accumulate(int j,
                   const Number U_j_local[dim+2],
                   const Number* c_ij,
                   const Number* precomputed)
    {
        // Extract precomputed variables for neighbor
        const Number s_j = precomputed[j * 2];
        const Number eta_j = precomputed[j * 2 + 1];
        
        const Number rho_j = U_j_local[0];
        const Number rho_j_inverse = Number(1.) / rho_j;
        
        // Compute momentum and entropy flux
        Number m_j_dot_c_ij = Number(0.);
        for (int d = 0; d < dim; ++d)
            m_j_dot_c_ij += U_j_local[1+d] * c_ij[d];
        
        const auto f_j = PF::f_local(U_j_local);
        
        const Number entropy_flux = 
            (eta_j * rho_j_inverse - eta_i * rho_i_inverse) * m_j_dot_c_ij;
        left += entropy_flux;
        
        for (int k = 0; k < problem_dimension; ++k) {
            Number component = Number(0.);
            for (int d = 0; d < dim; ++d) {
                component += (f_j(k, d) - f_i(k, d)) * c_ij[d];
            }
            right[k] += component;
        }
    }
    
    __device__
    Number alpha(const Number hd_i, const Number evc_factor) const
    {
        Number numerator = left;
        Number denominator = fabs(left);
        
        for (int k = 0; k < problem_dimension; ++k) {
            numerator -= d_eta_i[k] * right[k];
            denominator += fabs(d_eta_i[k] * right[k]);
        }
        
        const Number quotient = 
            fabs(numerator) / (denominator + hd_i * fabs(eta_i));
        
        return fmin(Number(1.), evc_factor * quotient);
    }
};

// Kernel for computation of alpha_i for all nodes
template<int dim, typename Number>
__global__ void compute_alpha_kernel(
    const State<dim, Number> U,
    const Sparsity& sparsity,
    const CijMatrix<dim, Number>& cij,
    const Number* precomputed,
    const MiMatrix<Number>& mi,
    Number* alpha_i,
    Number evc_factor,
    Number measure_of_omega_inverse,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    // Row bounds
    const int row_start = sparsity.row_offsets[i];
    const int row_end = sparsity.row_offsets[i + 1];
    
    // Skip constrained DoFs
    if (row_end - row_start == 1) {
        alpha_i[i] = Number(0.);
        return;
    }
    
    // Load U_i into local array
    Number U_i_local[dim + 2];
    U_i_local[0] = U.rho[i];
    U_i_local[1] = U.momentum_x[i];
    if constexpr (dim >= 2) U_i_local[2] = U.momentum_y[i];
    if constexpr (dim == 3) U_i_local[3] = U.momentum_z[i];
    U_i_local[dim + 1] = U.energy[i];
    
    // Initialize indicator for node i
    Indicator<dim, Number> indicator;
    indicator.reset(i, U_i_local, precomputed);
    
    // Accumulate contributions from neighbors (skip diagonal at idx = row_start)
    for (int idx = row_start + 1; idx < row_end; ++idx) {
        const int j = sparsity.col_indices[idx];
        const Number* c_ij = &cij.values[idx * dim];
        
        // Load U_j into local array
        Number U_j_local[dim + 2];
        U_j_local[0] = U.rho[j];
        U_j_local[1] = U.momentum_x[j];
        if constexpr (dim >= 2) U_j_local[2] = U.momentum_y[j];
        if constexpr (dim == 3) U_j_local[3] = U.momentum_z[j];
        U_j_local[dim + 1] = U.energy[j];
        
        indicator.accumulate(j, U_j_local, c_ij, precomputed);
    }
    
    // Compute alpha
    const Number hd_i = mi.values[i] * measure_of_omega_inverse;
    alpha_i[i] = indicator.alpha(hd_i, evc_factor);
}

#endif // INDICATOR_CUH