#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include "data_struct.cuh"
#include "phy_func.cuh"
#include "boundary_conditions.cuh"
#include "riemann_solver.cuh"
#include "indicator.cuh"
#include "limiter.cuh"
#include "atomic_operations.cuh"


// Designed for coalescence of data

// ============================================================================
// KERNEL 1: Prepare state
// ============================================================================
template<int dim, typename Number>
__global__ void prepare_state_kernel(
    State<dim, Number> d_U,
    Number* d_pressure,
    Number* d_speed_of_sound,
    Number* d_precomputed,
    const BoundaryData<dim, Number> d_boundary_data,
    Number inflow_rho,
    Number inflow_momentum_x,
    Number inflow_momentum_y,
    Number inflow_momentum_z,
    Number inflow_energy,
    int n_dofs)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_dofs) return;
    
    using PF = PhysicsFunctions<dim, Number>;
    
    apply_boundary_conditions_device<dim, Number>(
        d_U, d_boundary_data, 
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        tid);
    
    compute_precomputed_values_device<dim, Number>(d_U, d_precomputed, tid);
    
    d_pressure[tid] = PF::pressure(d_U, tid);
    const Number rho = d_U.rho[tid];
    d_speed_of_sound[tid] = sqrt(PF::gamma * d_pressure[tid] / rho);
}

// ============================================================================
// KERNEL 2: Compute off-diagonal d_ij and entropy viscosity alpha_i
// ============================================================================
template<int dim, typename Number>
__global__ void compute_off_diag_d_ij_and_alpha_i_kernel(
    const State<dim, Number> U,
    const Number* pressure,
    const Number* speed_of_sound,
    Number* alpha_i,
    Number* dij_matrix,
    const Sparsity& sparsity,
    const CijMatrix<dim, Number>& cij_matrix,
    const MiMatrix<Number>& mi_matrix,
    const Number* precomputed,
    Number evc_factor,
    Number measure_of_omega,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    using PF = PhysicsFunctions<dim, Number>;
    const Number measure_of_omega_inverse = Number(1) / measure_of_omega;
    
    const int row_start = sparsity.row_offsets[i];
    const int row_end = sparsity.row_offsets[i + 1];
    const int stencil_size = row_end - row_start;
    
    if (stencil_size == 1) {
        alpha_i[i] = Number(0);
        return;
    }
    
    // Load U_i into local array
    Number U_i_local[dim + 2];
    U_i_local[0] = U.rho[i];
    U_i_local[1] = U.momentum_x[i];
    if constexpr (dim >= 2) U_i_local[2] = U.momentum_y[i];
    if constexpr (dim == 3) U_i_local[3] = U.momentum_z[i];
    U_i_local[dim + 1] = U.energy[i];
    
    RiemannSolver<dim, Number> riemann_solver;
    Indicator<dim, Number> indicator;
    
    indicator.reset(i, U_i_local, precomputed);
    
    // Process off-diagonal entries
    for (int idx = row_start; idx < row_end; ++idx) {
        const int j = sparsity.col_indices[idx];
        const Number* c_ij = &cij_matrix.values[idx * dim];
        
        // Load U_j into local array
        Number U_j_local[dim + 2];
        U_j_local[0] = U.rho[j];
        U_j_local[1] = U.momentum_x[j];
        if constexpr (dim >= 2) U_j_local[2] = U.momentum_y[j];
        if constexpr (dim == 3) U_j_local[3] = U.momentum_z[j];
        U_j_local[dim + 1] = U.energy[j];
        
        indicator.accumulate(j, U_j_local, c_ij, precomputed);
        
        if (idx == row_start) continue; // Skip diagonal
        if (j < i) continue; // Upper triangular only
        
        Number norm = PF::norm_dim(c_ij);
        if (norm > Number(1e-14)) {
            Number n_ij[dim];
            #pragma unroll
            for (int k = 0; k < dim; ++k) {
                n_ij[k] = c_ij[k] / norm;
            }
            const Number lambda_max = riemann_solver.compute_local(U_i_local, U_j_local, n_ij);
            dij_matrix[idx] = norm * lambda_max;
        } else {
            dij_matrix[idx] = Number(0);
        }
    }
    
    const Number mass = mi_matrix.values[i];
    const Number hd_i = mass * measure_of_omega_inverse;
    alpha_i[i] = indicator.alpha(hd_i, evc_factor);
}

// ============================================================================
// KERNEL 3a: Complete d_ij matrix at boundaries
// ============================================================================
template<int dim, typename Number>
__global__ void complete_boundaries_kernel(
    const State<dim, Number> U,
    Number* dij_matrix,
    const MijMatrix<Number>& mij_matrix,
    const CijMatrix<dim, Number>& cij_matrix,
    const CouplingPairs& coupling_pairs)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= coupling_pairs.n_boundary_pairs) return;
    
    const int i = coupling_pairs.boundary_pairs[tid * 3 + 0];
    const int col_idx = coupling_pairs.boundary_pairs[tid * 3 + 1];
    const int j = coupling_pairs.boundary_pairs[tid * 3 + 2];
    
    if (j <= i) return;
    
    RiemannSolver<dim, Number> riemann_solver;
    
    Number c_ji[dim];
    bool found = false;
    for (int idx = cij_matrix.row_offsets[j]; idx < cij_matrix.row_offsets[j + 1]; ++idx) {
        if (cij_matrix.col_indices[idx] == i) {
            #pragma unroll
            for (int d = 0; d < dim; ++d) {
                c_ji[d] = cij_matrix.values[idx * dim + d];
            }
            found = true;
            break;
        }
    }
    
    if (found) {
        Number norm_ji = Number(0);
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            norm_ji += c_ji[d] * c_ji[d];
        }
        norm_ji = sqrt(norm_ji);
        
        if (norm_ji > Number(1e-12)) {
            Number n_ji[dim];
            #pragma unroll
            for (int d = 0; d < dim; ++d) {
                n_ji[d] = c_ji[d] / norm_ji;
            }
            
            // Load states into local arrays
            Number U_i_local[dim + 2];
            U_i_local[0] = U.rho[i];
            U_i_local[1] = U.momentum_x[i];
            if constexpr (dim >= 2) U_i_local[2] = U.momentum_y[i];
            if constexpr (dim == 3) U_i_local[3] = U.momentum_z[i];
            U_i_local[dim + 1] = U.energy[i];
            
            Number U_j_local[dim + 2];
            U_j_local[0] = U.rho[j];
            U_j_local[1] = U.momentum_x[j];
            if constexpr (dim >= 2) U_j_local[2] = U.momentum_y[j];
            if constexpr (dim == 3) U_j_local[3] = U.momentum_z[j];
            U_j_local[dim + 1] = U.energy[j];
            
            const int offset = mij_matrix.row_offsets[i] + col_idx;
            const Number d_ij = dij_matrix[offset];
            const Number lambda_max = riemann_solver.compute_local(U_j_local, U_i_local, n_ji);
            const Number d_ji = norm_ji * lambda_max;
            dij_matrix[offset] = fmax(d_ij, d_ji);
        }
    }
}

// ============================================================================
// KERNEL 3b: Compute diagonal and tau (shared memory)
// ============================================================================
template<int dim, typename Number>
__global__ void compute_diagonal_and_tau_kernel(
    Number* dij_matrix,
    Number* d_tau,
    const MijMatrix<Number>& mij_matrix,
    const MiMatrix<Number>& mi_matrix,
    Number cfl,
    int n_dofs)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    extern __shared__ __align__(sizeof(Number)) unsigned char shared_mem_raw[];
    Number* shared_tau = reinterpret_cast<Number*>(shared_mem_raw);
    
    Number local_tau = Number(1e20);
    
    if (tid < n_dofs) {
        const int row_start = mij_matrix.row_offsets[tid];
        const int row_end = mij_matrix.row_offsets[tid + 1];
        const int row_length = row_end - row_start;
        
        if (row_length > 1) {
            Number d_sum = Number(0);
            
            #pragma unroll 4
            for (int idx = row_start + 1; idx < row_end; ++idx) {
                const int j = mij_matrix.col_indices[idx];
                
                if (j < tid) {
                    for (int jdx = mij_matrix.row_offsets[j]; 
                        jdx < mij_matrix.row_offsets[j + 1]; ++jdx) {
                        if (mij_matrix.col_indices[jdx] == tid) {
                            dij_matrix[idx] = dij_matrix[jdx];
                            break;
                        }
                    }
                }
                d_sum -= dij_matrix[idx];
            }
            
            d_sum = fmin(d_sum, Number(-1e-6));
            dij_matrix[row_start] = d_sum;
            
            const Number mass = mi_matrix.values[tid];
            local_tau = cfl * mass / (Number(-2.0) * d_sum);
        }
    }
    
    shared_tau[threadIdx.x] = local_tau;
    __syncthreads();
    
    if (threadIdx.x < 32) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_tau = fmin(local_tau, __shfl_down_sync(0xFFFFFFFF, local_tau, offset));
        }
        if (threadIdx.x == 0) {
            shared_tau[threadIdx.x / 32] = local_tau;
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        local_tau = shared_tau[0];
        for (int i = 1; i < (blockDim.x + 31) / 32; ++i) {
            local_tau = fmin(local_tau, shared_tau[i]);
        }
        atomicMinNumber(d_tau, local_tau);
    }
}  // Vérifier ceci (réduction seulement sur 32 threads) ?

// ============================================================================
// KERNEL 4: Low-order update
// ============================================================================
template<int dim, typename Number>
__global__ void __launch_bounds__(128, 4) low_order_update_kernel(
    const State<dim, Number> U,
    State<dim, Number> new_U,
    const Number* __restrict__ pressure,
    const Number* __restrict__ alpha_i,
    const Number* __restrict__ dij_matrix,
    Pij<dim, Number> pij_matrix,
    Ri<dim, Number> ri,
    Number* __restrict__ bounds,
    const Sparsity& sparsity,
    const MijMatrix<Number>& mij_matrix,
    const MiMatrix<Number>& mi_matrix,
    const MiMatrixInverse<Number>& mi_inv_matrix,
    const CijMatrix<dim, Number>& cij_matrix,
    const Number* __restrict__ precomputed,
    Number tau,
    Number measure_of_omega,
    const State<dim, Number> stage_U_0,
    const State<dim, Number> stage_U_1,
    Number weight_0,
    Number weight_1,
    Number weight_main,
    int stage,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    using PF = PhysicsFunctions<dim, Number>;
    using LimiterType = Limiter<dim, Number>;
    
    const int row_start = __ldg(&sparsity.row_offsets[i]);
    const int row_end = __ldg(&sparsity.row_offsets[i + 1]);
    const int row_length = row_end - row_start;
    
    if (row_length == 1) return;
    
    const Number alpha_i_local = __ldg(&alpha_i[i]);
    const Number m_i_local = __ldg(&mi_matrix.values[i]);
    const Number m_i_inv_local = __ldg(&mi_inv_matrix.values[i]);
    const Number pressure_i_local = __ldg(&pressure[i]);
    
    Number U_i_array[dim + 2];
    U_i_array[0] = __ldg(&U.rho[i]);
    U_i_array[1] = __ldg(&U.momentum_x[i]);
    if constexpr (dim >= 2) U_i_array[2] = __ldg(&U.momentum_y[i]);
    if constexpr (dim == 3) U_i_array[3] = __ldg(&U.momentum_z[i]);
    U_i_array[dim + 1] = __ldg(&U.energy[i]);
    
    Number U_i_new_array[dim + 2];
    #pragma unroll
    for (int k = 0; k < dim + 2; ++k) {
        U_i_new_array[k] = U_i_array[k];
    }
    
    const auto flux_i = PF::f_local(U_i_array);
    
    Flux<dim, Number> flux_iHs[3];
    if (stage >= 1 && stage_U_0.rho != nullptr) {
        Number U_stage[dim + 2];
        U_stage[0] = __ldg(&stage_U_0.rho[i]);
        U_stage[1] = __ldg(&stage_U_0.momentum_x[i]);
        if constexpr (dim >= 2) U_stage[2] = __ldg(&stage_U_0.momentum_y[i]);
        if constexpr (dim == 3) U_stage[3] = __ldg(&stage_U_0.momentum_z[i]);
        U_stage[dim + 1] = __ldg(&stage_U_0.energy[i]);
        flux_iHs[0] = PF::f_local(U_stage);
    }
    if (stage >= 2 && stage_U_1.rho != nullptr) {
        Number U_stage[dim + 2];
        U_stage[0] = __ldg(&stage_U_1.rho[i]);
        U_stage[1] = __ldg(&stage_U_1.momentum_x[i]);
        if constexpr (dim >= 2) U_stage[2] = __ldg(&stage_U_1.momentum_y[i]);
        if constexpr (dim == 3) U_stage[3] = __ldg(&stage_U_1.momentum_z[i]);
        U_stage[dim + 1] = __ldg(&stage_U_1.energy[i]);
        flux_iHs[1] = PF::f_local(U_stage);
    }
    
    Number F_iH_array[dim + 2] = {0};
    
    LimiterType limiter;
    limiter.reset(U_i_array, pressure_i_local, __ldg(&precomputed[i * 2]));
    
    for (int idx = row_start; idx < row_end; ++idx) {
        const int j = __ldg(&sparsity.col_indices[idx]);
        
        Number U_j_array[dim + 2];
        U_j_array[0] = __ldg(&U.rho[j]);
        U_j_array[1] = __ldg(&U.momentum_x[j]);
        if constexpr (dim >= 2) U_j_array[2] = __ldg(&U.momentum_y[j]);
        if constexpr (dim == 3) U_j_array[3] = __ldg(&U.momentum_z[j]);
        U_j_array[dim + 1] = __ldg(&U.energy[j]);
        
        const Number alpha_j_local = __ldg(&alpha_i[j]);
        const Number d_ij_local = __ldg(&dij_matrix[idx]);
        const Number factor = (alpha_i_local + alpha_j_local) * Number(0.5);
        const Number d_ijH = d_ij_local * factor;
        
        Number c_ij_local[dim];
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            c_ij_local[d] = __ldg(&cij_matrix.values[idx * dim + d]);
        }
        
        constexpr Number eps = Number(1e-14);
        const Number scale = (abs(d_ij_local) < eps * eps) ? Number(0) : Number(1) / d_ij_local;
        Number scaled_c_ij[dim];
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            scaled_c_ij[d] = c_ij_local[d] * scale;
        }
        
        const auto flux_j = PF::f_local(U_j_array);
        const Number m_ij = __ldg(&mij_matrix.values[idx]);
        
        Number flux_ij[dim + 2];
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            flux_ij[k] = Number(0);
            #pragma unroll
            for (int d = 0; d < dim; ++d) {
                flux_ij[k] -= (flux_i(k, d) + flux_j(k, d)) * c_ij_local[d];
            }
        }
        
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            U_i_new_array[k] += tau * m_i_inv_local * flux_ij[k];
            U_i_new_array[k] += tau * m_i_inv_local * d_ij_local * (U_j_array[k] - U_i_array[k]);
            F_iH_array[k] += d_ijH * (U_j_array[k] - U_i_array[k]);
            F_iH_array[k] += weight_main * flux_ij[k];
        }
        
        Number P_ij_tmp[dim + 2];
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            P_ij_tmp[k] = -flux_ij[k];
            P_ij_tmp[k] += (d_ijH - d_ij_local) * (U_j_array[k] - U_i_array[k]);
            P_ij_tmp[k] += weight_main * flux_ij[k];
        }

        if (stage >= 1 && stage_U_0.rho != nullptr) {
            Number U_j_stage[dim + 2];
            U_j_stage[0] = __ldg(&stage_U_0.rho[j]);
            U_j_stage[1] = __ldg(&stage_U_0.momentum_x[j]);
            if constexpr (dim >= 2) U_j_stage[2] = __ldg(&stage_U_0.momentum_y[j]);
            if constexpr (dim == 3) U_j_stage[3] = __ldg(&stage_U_0.momentum_z[j]);
            U_j_stage[dim + 1] = __ldg(&stage_U_0.energy[j]);
            
            const auto flux_jHs0 = PF::f_local(U_j_stage);
            Number flux_ij_stage[dim + 2];
            #pragma unroll
            for (int k = 0; k < dim + 2; ++k) {
                flux_ij_stage[k] = Number(0);
                #pragma unroll
                for (int d = 0; d < dim; ++d) {
                    flux_ij_stage[k] -= (flux_iHs[0](k, d) + flux_jHs0(k, d)) * c_ij_local[d];
                }
                F_iH_array[k] += weight_0 * flux_ij_stage[k];
                P_ij_tmp[k] += weight_0 * flux_ij_stage[k];
            }
        }
        
        if (stage >= 2 && stage_U_1.rho != nullptr) {
            Number U_j_stage[dim + 2];
            U_j_stage[0] = __ldg(&stage_U_1.rho[j]);
            U_j_stage[1] = __ldg(&stage_U_1.momentum_x[j]);
            if constexpr (dim >= 2) U_j_stage[2] = __ldg(&stage_U_1.momentum_y[j]);
            if constexpr (dim == 3) U_j_stage[3] = __ldg(&stage_U_1.momentum_z[j]);
            U_j_stage[dim + 1] = __ldg(&stage_U_1.energy[j]);
            
            const auto flux_jHs1 = PF::f_local(U_j_stage);
            Number flux_ij_stage[dim + 2];
            #pragma unroll
            for (int k = 0; k < dim + 2; ++k) {
                flux_ij_stage[k] = Number(0);
                #pragma unroll
                for (int d = 0; d < dim; ++d) {
                    flux_ij_stage[k] -= (flux_iHs[1](k, d) + flux_jHs1(k, d)) * c_ij_local[d];
                }
                F_iH_array[k] += weight_1 * flux_ij_stage[k];
                P_ij_tmp[k] += weight_1 * flux_ij_stage[k];
            }
        }
        
        pij_matrix.p_rho[idx] = P_ij_tmp[0];
        pij_matrix.p_momentum_x[idx] = P_ij_tmp[1];
        if constexpr (dim >= 2) pij_matrix.p_momentum_y[idx] = P_ij_tmp[2];
        if constexpr (dim == 3) pij_matrix.p_momentum_z[idx] = P_ij_tmp[3];
        pij_matrix.p_energy[idx] = P_ij_tmp[dim + 1];
        
        limiter.accumulate(U_j_array, scaled_c_ij, __ldg(&pressure[j]));
    }
    
    new_U.rho[i] = U_i_new_array[0];
    new_U.momentum_x[i] = U_i_new_array[1];
    if constexpr (dim >= 2) new_U.momentum_y[i] = U_i_new_array[2];
    if constexpr (dim == 3) new_U.momentum_z[i] = U_i_new_array[3];
    new_U.energy[i] = U_i_new_array[dim + 1];
    
    ri.r_rho[i] = F_iH_array[0];
    ri.r_momentum_x[i] = F_iH_array[1];
    if constexpr (dim >= 2) ri.r_momentum_y[i] = F_iH_array[2];
    if constexpr (dim == 3) ri.r_momentum_z[i] = F_iH_array[3];
    ri.r_energy[i] = F_iH_array[dim + 1];
    
    const Number measure_of_omega_inverse = Number(1) / measure_of_omega;
    const Number hd_i = m_i_local * measure_of_omega_inverse;
    typename LimiterType::Bounds relaxed_bounds = limiter.get_bounds(hd_i);
    bounds[i * 3 + 0] = relaxed_bounds.rho_min;
    bounds[i * 3 + 1] = relaxed_bounds.rho_max;
    bounds[i * 3 + 2] = relaxed_bounds.s_min;
}

// ============================================================================
// KERNEL 5: Compute limiter
// ============================================================================
template<int dim, typename Number>
__global__ void compute_limiter_kernel(
    const State<dim, Number> new_U,
    Pij<dim, Number> pij_matrix,
    const Ri<dim, Number> ri,
    Number* lij_matrix,
    const Number* bounds,
    const Sparsity& sparsity,
    const MijMatrix<Number>& mij_matrix,
    const MiMatrixInverse<Number>& mi_inv_matrix,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    using LimiterType = Limiter<dim, Number>;
    
    const int row_start = sparsity.row_offsets[i];
    const int row_end = sparsity.row_offsets[i + 1];
    const int row_length = row_end - row_start;
    
    if (row_length == 1) return;
    
    const Number m_i_inv = mi_inv_matrix.values[i];
    
    // Load U_i_new and F_iH into array (optimization)
    Number U_i_new_array[dim + 2];
    U_i_new_array[0] = new_U.rho[i];
    U_i_new_array[1] = new_U.momentum_x[i];
    if constexpr (dim >= 2) U_i_new_array[2] = new_U.momentum_y[i];
    if constexpr (dim == 3) U_i_new_array[3] = new_U.momentum_z[i];
    U_i_new_array[dim + 1] = new_U.energy[i];
    
    Number F_iH_array[dim + 2];
    F_iH_array[0] = ri.r_rho[i];
    F_iH_array[1] = ri.r_momentum_x[i];
    if constexpr (dim >= 2) F_iH_array[2] = ri.r_momentum_y[i];
    if constexpr (dim == 3) F_iH_array[3] = ri.r_momentum_z[i];
    F_iH_array[dim + 1] = ri.r_energy[i];
    
    const Number lambda_inv = Number(row_length - 1);
    const Number factor = tau * m_i_inv * lambda_inv;
    
    typename LimiterType::Bounds bounds_i;
    bounds_i.rho_min = bounds[i * 3 + 0];
    bounds_i.rho_max = bounds[i * 3 + 1];
    bounds_i.s_min = bounds[i * 3 + 2];
    
    LimiterType limiter;
    
    for (int col_idx = 1; col_idx < row_length; ++col_idx) {
        const int idx = row_start + col_idx;
        
        // Load P_ij
        Number P_ij_array[dim + 2];
        P_ij_array[0] = pij_matrix.p_rho[idx];
        P_ij_array[1] = pij_matrix.p_momentum_x[idx];
        if constexpr (dim >= 2) P_ij_array[2] = pij_matrix.p_momentum_y[idx];
        if constexpr (dim == 3) P_ij_array[3] = pij_matrix.p_momentum_z[idx];
        P_ij_array[dim + 1] = pij_matrix.p_energy[idx];
        
        const unsigned int j = sparsity.col_indices[idx];
        
        // Load F_jH
        Number F_jH_array[dim + 2];
        F_jH_array[0] = ri.r_rho[j];
        F_jH_array[1] = ri.r_momentum_x[j];
        if constexpr (dim >= 2) F_jH_array[2] = ri.r_momentum_y[j];
        if constexpr (dim == 3) F_jH_array[3] = ri.r_momentum_z[j];
        F_jH_array[dim + 1] = ri.r_energy[j];
        
        // Mass matrix correction
        const Number m_j_inv = mi_inv_matrix.values[j];
        const Number m_ij = mij_matrix.values[idx];
        const Number b_ij = -m_ij * m_j_inv;
        const Number b_ji = -m_ij * m_i_inv;
        
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            P_ij_array[k] += b_ij * F_jH_array[k] - b_ji * F_iH_array[k];
            P_ij_array[k] *= factor;
        }
        
        // Update modified P_ij
        pij_matrix.p_rho[idx] = P_ij_array[0];
        pij_matrix.p_momentum_x[idx] = P_ij_array[1];
        if constexpr (dim >= 2) pij_matrix.p_momentum_y[idx] = P_ij_array[2];
        if constexpr (dim == 3) pij_matrix.p_momentum_z[idx] = P_ij_array[3];
        pij_matrix.p_energy[idx] = P_ij_array[dim + 1];
        
        lij_matrix[idx] = limiter.limit(bounds_i, U_i_new_array, P_ij_array, Number(0), Number(1));
    }
}

// ============================================================================
// KERNEL 6: High-order update iteration 1
// ============================================================================
template<int dim, typename Number>
__global__ void high_order_update_iter1_kernel(
    State<dim, Number> new_U,
    const Pij<dim, Number> pij_matrix,
    const Number* lij_matrix,
    Number* lij_next,
    const Number* bounds,
    const Sparsity& sparsity,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    using LimiterType = Limiter<dim, Number>;
    
    const int row_start = sparsity.row_offsets[i];
    const int row_end = sparsity.row_offsets[i + 1];
    const int row_length = row_end - row_start;
    
    if (row_length <= 1) return;
    
    // Load U_i_new into array (optimization)
    Number U_i_new_array[dim + 2];
    U_i_new_array[0] = new_U.rho[i];
    U_i_new_array[1] = new_U.momentum_x[i];
    if constexpr (dim >= 2) U_i_new_array[2] = new_U.momentum_y[i];
    if constexpr (dim == 3) U_i_new_array[3] = new_U.momentum_z[i];
    U_i_new_array[dim + 1] = new_U.energy[i];
    
    const Number lambda = Number(1) / Number(row_length - 1);
    
    // First pass: apply limiters
    for (int col_idx = 1; col_idx < row_length; ++col_idx) {
        const int idx = row_start + col_idx;
    
        Number l_ij_ = lij_matrix[idx];    
        const int idx_ji = sparsity.transpose_indices[idx];
        if (idx_ji >= 0) {
            l_ij_ = fmin(l_ij_, lij_matrix[idx_ji]);
        }
        
        // Load P_ij and apply contribution
        Number P_ij[dim + 2];
        P_ij[0] = pij_matrix.p_rho[idx];
        P_ij[1] = pij_matrix.p_momentum_x[idx];
        if constexpr (dim >= 2) P_ij[2] = pij_matrix.p_momentum_y[idx];
        if constexpr (dim == 3) P_ij[3] = pij_matrix.p_momentum_z[idx];
        P_ij[dim + 1] = pij_matrix.p_energy[idx];
        
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            U_i_new_array[k] += l_ij_ * lambda * P_ij[k];
        }
        
        lij_next[idx] = l_ij_;
    }
    
    // Write new_U
    new_U.rho[i] = U_i_new_array[0];
    new_U.momentum_x[i] = U_i_new_array[1];
    if constexpr (dim >= 2) new_U.momentum_y[i] = U_i_new_array[2];
    if constexpr (dim == 3) new_U.momentum_z[i] = U_i_new_array[3];
    new_U.energy[i] = U_i_new_array[dim + 1];
    
    // Prepare bounds
    typename LimiterType::Bounds bounds_i;
    bounds_i.rho_min = bounds[i * 3 + 0];
    bounds_i.rho_max = bounds[i * 3 + 1];
    bounds_i.s_min = bounds[i * 3 + 2];
    
    LimiterType limiter;
    
    // Second pass: compute new limiting coefficients
    for (int col_idx = 1; col_idx < row_length; ++col_idx) {
        const int idx = row_start + col_idx;
        
        const Number old_l_ij = lij_next[idx];
        
        Number P_ij[dim + 2];
        P_ij[0] = pij_matrix.p_rho[idx];
        P_ij[1] = pij_matrix.p_momentum_x[idx];
        if constexpr (dim >= 2) P_ij[2] = pij_matrix.p_momentum_y[idx];
        if constexpr (dim == 3) P_ij[3] = pij_matrix.p_momentum_z[idx];
        P_ij[dim + 1] = pij_matrix.p_energy[idx];
        
        Number new_p_ij[dim + 2];
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            new_p_ij[k] = (Number(1) - old_l_ij) * P_ij[k];
        }
        
        const Number new_l_ij = limiter.limit(bounds_i, U_i_new_array, new_p_ij, Number(0), Number(1));
        
        lij_next[idx] = (Number(1) - old_l_ij) * new_l_ij;
    }
}

// ============================================================================
// KERNEL 7: High-order update iteration 2
// ============================================================================
template<int dim, typename Number>
__global__ void high_order_update_iter2_kernel(
    State<dim, Number> new_U,
    const Pij<dim, Number> pij_matrix,
    const Number* lij_matrix,
    const Sparsity& sparsity,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    const int row_start = sparsity.row_offsets[i];
    const int row_end = sparsity.row_offsets[i + 1];
    const int row_length = row_end - row_start;
    
    if (row_length <= 1) return;
    
    // Load U_i_new into array (optimization)
    Number U_i_new_array[dim + 2];
    U_i_new_array[0] = new_U.rho[i];
    U_i_new_array[1] = new_U.momentum_x[i];
    if constexpr (dim >= 2) U_i_new_array[2] = new_U.momentum_y[i];
    if constexpr (dim == 3) U_i_new_array[3] = new_U.momentum_z[i];
    U_i_new_array[dim + 1] = new_U.energy[i];
    
    const Number lambda = Number(1) / Number(row_length - 1);
    
    // Apply final limiting
    for (int col_idx = 1; col_idx < row_length; ++col_idx) {
        const int idx = row_start + col_idx;
        
        Number l_ij_ = lij_matrix[idx];
        const int idx_ji = sparsity.transpose_indices[idx];
        if (idx_ji >= 0) {
            l_ij_ = fmin(l_ij_, lij_matrix[idx_ji]);
        }
        
        // Load P_ij and apply contribution
        Number P_ij[dim + 2];
        P_ij[0] = pij_matrix.p_rho[idx];
        P_ij[1] = pij_matrix.p_momentum_x[idx];
        if constexpr (dim >= 2) P_ij[2] = pij_matrix.p_momentum_y[idx];
        if constexpr (dim == 3) P_ij[3] = pij_matrix.p_momentum_z[idx];
        P_ij[dim + 1] = pij_matrix.p_energy[idx];
        
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            U_i_new_array[k] += l_ij_ * lambda * P_ij[k];
        }
    }
    
    // Write final result
    new_U.rho[i] = U_i_new_array[0];
    new_U.momentum_x[i] = U_i_new_array[1];
    if constexpr (dim >= 2) new_U.momentum_y[i] = U_i_new_array[2];
    if constexpr (dim == 3) new_U.momentum_z[i] = U_i_new_array[3];
    new_U.energy[i] = U_i_new_array[dim + 1];
}


// Kernel for operation sadd
template<int dim, typename Number>
__global__ void sadd_kernel(
    State<dim, Number> dst,
    State<dim, Number> src,
    Number s, Number b,
    int n_dofs)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    
    dst.rho[i] = s * dst.rho[i] + b * src.rho[i];
    dst.momentum_x[i] = s * dst.momentum_x[i] + b * src.momentum_x[i];
    if constexpr (dim >= 2) {
        dst.momentum_y[i] = s * dst.momentum_y[i] + b * src.momentum_y[i];
    }
    if constexpr (dim == 3) {
        dst.momentum_z[i] = s * dst.momentum_z[i] + b * src.momentum_z[i];
    }
    dst.energy[i] = s * dst.energy[i] + b * src.energy[i];
}