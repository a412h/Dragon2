// cuda_time_loop.cu
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

#include "data_struct.cuh"
#include "phy_func.cuh"
#include "hyperbolic_kernels.cu"
#include "configuration.h"
#include "output.h"
#include "boundary_conditions.cuh"
#include "offline_data.h"
#include "parabolic_solver.cuh"

// ============================================================================
// Functions for SoA memory operations
// ============================================================================
template<int dim, typename Number>
void copy_state(State<dim, Number>& dst, const State<dim, Number>& src, 
                int n_dofs, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(dst.rho, src.rho, n_dofs * sizeof(Number), 
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(dst.momentum_x, src.momentum_x, n_dofs * sizeof(Number), 
                               cudaMemcpyDeviceToDevice, stream));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMemcpyAsync(dst.momentum_y, src.momentum_y, n_dofs * sizeof(Number), 
                                   cudaMemcpyDeviceToDevice, stream));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMemcpyAsync(dst.momentum_z, src.momentum_z, n_dofs * sizeof(Number), 
                                   cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaMemcpyAsync(dst.energy, src.energy, n_dofs * sizeof(Number), 
                               cudaMemcpyDeviceToDevice, stream));
}

// ============================================================================
// Stage Executor
// ============================================================================
template<int dim, typename Number>
class StageExecutor {
private:
    // Device pointers
    State<dim, Number> d_U;
    State<dim, Number> d_temp_0;
    State<dim, Number> d_temp_1;
    State<dim, Number> d_temp_2;
    State<dim, Number> d_temp_3;
    State<dim, Number> d_new_U;
    Number* d_pressure;
    Number* d_speed_of_sound;
    Number* d_precomputed;
    Number* d_alpha_i;
    Number* d_dij;
    Pij<dim, Number> d_pij;
    Ri<dim, Number> d_ri;
    Number* d_bounds;
    Number* d_lij;
    Number* d_lij_next;
    Number* d_tau;
    
    // Matrices and data structures
    const MijMatrix<Number>& d_mij;
    const MiMatrix<Number>& d_mi;
    const MiMatrixInverse<Number>& d_mi_inv;
    const CijMatrix<dim, Number>& d_cij;
    const Sparsity& d_sparsity;
    const BoundaryData<dim, Number>& d_boundary_data;
    const CouplingPairs& d_coupling_pairs;
    
    // Parameters - individual components for inflow
    Number inflow_rho;
    Number inflow_momentum_x;
    Number inflow_momentum_y;
    Number inflow_momentum_z;
    Number inflow_energy;
    const Number measure_of_omega;
    const Number cfl;
    const Number evc_factor;
    const int n_dofs;
    const int nnz;
    const Number mu;
    const Number cv_inverse_kappa;
    
    cudaStream_t stream;  // Variable stream declared before parabolic_solver instanciation (debug)
    const OfflineData<dim, double>& offline_data;
    
    ParabolicSolver<dim, Number> parabolic_solver;  // Instanciation for parabolic part (after stream is declared) (debug)
    
    // Kernel configuration
    struct KernelConfig {
        int threadsPerBlock;
        int sharedMemorySize;
        int blocksPerGrid;
    };
    
    KernelConfig prepareConfig;
    KernelConfig viscosityConfig;
    KernelConfig lowOrderConfig;
    KernelConfig limiterConfig;
    KernelConfig highOrderConfig;
    KernelConfig diagonalConfig;
    
public:
    StageExecutor(
        State<dim, Number>& _d_U,
        State<dim, Number>& _d_temp_0,
        State<dim, Number>& _d_temp_1,
        State<dim, Number>& _d_temp_2,
        State<dim, Number>& _d_temp_3,
        State<dim, Number>& _d_new_U,
        Number* _d_pressure,
        Number* _d_speed_of_sound,
        Number* _d_precomputed,
        Number* _d_alpha_i,
        Number* _d_dij,
        Pij<dim, Number>& _d_pij,
        Ri<dim, Number>& _d_ri,
        Number* _d_bounds,
        Number* _d_lij,
        Number* _d_lij_next,
        const MijMatrix<Number>& _d_mij,
        const MiMatrix<Number>& _d_mi,
        const MiMatrixInverse<Number>& _d_mi_inv,
        const CijMatrix<dim, Number>& _d_cij,
        const Sparsity& _d_sparsity,
        const BoundaryData<dim, Number>& _d_boundary_data,
        const CouplingPairs& _d_coupling_pairs,
        Number _inflow_rho,
        Number _inflow_momentum_x,
        Number _inflow_momentum_y,
        Number _inflow_momentum_z,
        Number _inflow_energy,
        Number _measure_of_omega,
        Number _cfl,
        Number _evc_factor,
        int _n_dofs,
        int _nnz,
        Number _mu,
        Number _cv_inverse_kappa,        
        const OfflineData<dim, double>& _offline_data,
        cudaStream_t _stream)
        : d_U(_d_U), d_temp_0(_d_temp_0), d_temp_1(_d_temp_1), d_temp_2(_d_temp_2), d_temp_3(_d_temp_3),
          d_new_U(_d_new_U), d_pressure(_d_pressure), d_speed_of_sound(_d_speed_of_sound),
          d_precomputed(_d_precomputed), d_alpha_i(_d_alpha_i), d_dij(_d_dij),
          d_pij(_d_pij), d_ri(_d_ri), d_bounds(_d_bounds), d_lij(_d_lij),
          d_lij_next(_d_lij_next), d_mij(_d_mij), d_mi(_d_mi), d_mi_inv(_d_mi_inv),
          d_cij(_d_cij), d_sparsity(_d_sparsity), d_boundary_data(_d_boundary_data),
          d_coupling_pairs(_d_coupling_pairs),
          inflow_rho(_inflow_rho), inflow_momentum_x(_inflow_momentum_x),
          inflow_momentum_y(_inflow_momentum_y), inflow_momentum_z(_inflow_momentum_z),
          inflow_energy(_inflow_energy),
          measure_of_omega(_measure_of_omega), cfl(_cfl), evc_factor(_evc_factor),
          n_dofs(_n_dofs), nnz(_nnz),
          mu(_mu), cv_inverse_kappa(_cv_inverse_kappa),
          stream(_stream),
          offline_data(_offline_data),
          parabolic_solver(n_dofs, _mu, _cv_inverse_kappa, _stream)
    {
        CUDA_CHECK(cudaMalloc(&d_tau, sizeof(Number)));

        // Initialize parabolic solver with mesh connectivity
        parabolic_solver.set_element_connectivity(offline_data);
        
        // Set init_U with uniform inflow conditions
        std::vector<Number> h_init_rho(n_dofs, inflow_rho);
        std::vector<Number> h_init_mx(n_dofs, inflow_momentum_x);
        std::vector<Number> h_init_my(n_dofs, inflow_momentum_y);
        std::vector<Number> h_init_mz(n_dofs, inflow_momentum_z);
        std::vector<Number> h_init_e(n_dofs, inflow_energy);
        
        CUDA_CHECK(cudaMemcpy(parabolic_solver.init_U.rho, h_init_rho.data(), 
                             n_dofs * sizeof(Number), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(parabolic_solver.init_U.momentum_x, h_init_mx.data(), 
                             n_dofs * sizeof(Number), cudaMemcpyHostToDevice));
        if constexpr (dim >= 2) {
            CUDA_CHECK(cudaMemcpy(parabolic_solver.init_U.momentum_y, h_init_my.data(), 
                                 n_dofs * sizeof(Number), cudaMemcpyHostToDevice));
        }
        if constexpr (dim == 3) {
            CUDA_CHECK(cudaMemcpy(parabolic_solver.init_U.momentum_z, h_init_mz.data(), 
                                 n_dofs * sizeof(Number), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(parabolic_solver.init_U.energy, h_init_e.data(), 
                             n_dofs * sizeof(Number), cudaMemcpyHostToDevice));
        
        parabolic_solver.set_matrices(d_mij, d_mi, d_cij, d_boundary_data);
        
        // Kernel configuration
        prepareConfig.threadsPerBlock = 256;
        prepareConfig.sharedMemorySize = 0;
        prepareConfig.blocksPerGrid = (n_dofs + prepareConfig.threadsPerBlock - 1) / prepareConfig.threadsPerBlock;
        
        viscosityConfig.threadsPerBlock = 256;
        viscosityConfig.sharedMemorySize = 0;
        viscosityConfig.blocksPerGrid = (n_dofs + viscosityConfig.threadsPerBlock - 1) / viscosityConfig.threadsPerBlock;
        
        lowOrderConfig.threadsPerBlock = 128;
        lowOrderConfig.sharedMemorySize = 0;
        lowOrderConfig.blocksPerGrid = (n_dofs + lowOrderConfig.threadsPerBlock - 1) / lowOrderConfig.threadsPerBlock;
        
        limiterConfig.threadsPerBlock = 192;
        limiterConfig.sharedMemorySize = 0;
        limiterConfig.blocksPerGrid = (n_dofs + limiterConfig.threadsPerBlock - 1) / limiterConfig.threadsPerBlock;
        
        highOrderConfig.threadsPerBlock = 256;
        highOrderConfig.sharedMemorySize = 0;
        highOrderConfig.blocksPerGrid = (n_dofs + highOrderConfig.threadsPerBlock - 1) / highOrderConfig.threadsPerBlock;
        
        diagonalConfig.threadsPerBlock = 256;
        diagonalConfig.sharedMemorySize = diagonalConfig.threadsPerBlock * sizeof(Number);
        diagonalConfig.blocksPerGrid = (n_dofs + diagonalConfig.threadsPerBlock - 1) / diagonalConfig.threadsPerBlock;
        
        // Display configuration
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        std::cout << "\nGPU Configuration:" << std::endl;
        std::cout << "  Device: " << prop.name << std::endl;
        std::cout << "  SMs: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Kernel configurations:" << std::endl;
        std::cout << "    Low-order kernel: " << lowOrderConfig.threadsPerBlock << " threads/block" << std::endl;
        std::cout << "    Limiter kernel: " << limiterConfig.threadsPerBlock << " threads/block" << std::endl;
        
        // Configure L1 cache for memory-bound kernels
        cudaFuncSetCacheConfig(low_order_update_kernel<dim, Number>, cudaFuncCachePreferL1);
        cudaFuncSetCacheConfig(compute_limiter_kernel<dim, Number>, cudaFuncCachePreferL1);
    }
    
    ~StageExecutor() {
        CUDA_CHECK(cudaFree(d_tau));
    }

    // Various constants
    static constexpr Number w_0 = Number(0);
    static constexpr Number w_0d75 = Number(0.75);
    static constexpr Number w_1 = Number(1);
    static constexpr Number w_2 = Number(2);
    static constexpr Number w_2d25 = Number(2.25);
    static constexpr Number w_m1 = Number(-1);
    static constexpr Number w_m2 = Number(-2);
    static constexpr Number efficiency_ = Number(3);

    Number execute_timestep(Number tau_max) {
        State<dim, Number> null_state = {nullptr, nullptr, nullptr, nullptr, nullptr};
        
        // First explicit ERK(3,3,1) step with final result in temp_[2]
        
        prepare_state_hyperbolic(d_U);
        Number tau = compute_step_hyperbolic(0, d_U, d_temp_0, null_state, null_state, w_0, w_0, w_1, tau_max / efficiency_);

        prepare_state_hyperbolic(d_temp_0);
        compute_step_hyperbolic(1, d_temp_0, d_temp_1, d_U, null_state, w_m1, w_0, w_2, tau);

        prepare_state_hyperbolic(d_temp_1);
        compute_step_hyperbolic(2, d_temp_1, d_temp_2, d_U, d_temp_0, w_0d75, w_m2, w_2d25, tau);

        // Implicit Crank-Nicolson step with final result in temp_[3]
        parabolic_solver.compute_step_parabolic(d_temp_2, d_temp_3, tau);

        // Second explicit ERK(3,3,1) step with final result in temp_[2]
        
        prepare_state_hyperbolic(d_temp_3);
        compute_step_hyperbolic(0, d_temp_3, d_temp_0, null_state, null_state, w_0, w_0, w_1, tau);

        prepare_state_hyperbolic(d_temp_0);
        compute_step_hyperbolic(1, d_temp_0, d_temp_1, d_temp_3, null_state, w_m1, w_0, w_2, tau);

        prepare_state_hyperbolic(d_temp_1);
        compute_step_hyperbolic(2, d_temp_1, d_temp_2, d_temp_3, d_temp_0, w_0d75, w_m2, w_2d25, tau);

        // Update U for next iteration
        copy_state(d_U, d_temp_2, n_dofs, stream);

        return efficiency_ * tau;
    }
    
private:

    void prepare_state_hyperbolic(State<dim, Number>& state_vector)
    {
        prepare_state_kernel<dim, Number><<<prepareConfig.blocksPerGrid, 
                                            prepareConfig.threadsPerBlock, 
                                            0, stream>>>(
            state_vector, d_pressure, d_speed_of_sound, d_precomputed,
            d_boundary_data, inflow_rho, inflow_momentum_x, inflow_momentum_y, 
            inflow_momentum_z, inflow_energy, n_dofs);        
    }
    
    Number compute_step_hyperbolic(int stage,
        State<dim, Number>& d_old_state_vector,
        State<dim, Number>& d_new_state_vector,
        State<dim, Number>& d_stage_state_vector_0,
        State<dim, Number>& d_stage_state_vector_1,
        Number stage_weight_0,
        Number stage_weight_1,
        Number stage_weight_acc,
        Number tau_max)
    {
        compute_off_diag_d_ij_and_alpha_i_kernel<dim, Number><<<viscosityConfig.blocksPerGrid, 
                                                            viscosityConfig.threadsPerBlock, 
                                                            0, stream>>>(
            d_old_state_vector, d_pressure, d_speed_of_sound, d_alpha_i, d_dij,
            d_sparsity, d_cij, d_mi, d_precomputed,
            evc_factor, measure_of_omega, n_dofs);

        Number tau = tau_max;
        if (stage == 0) {
            int boundary_blocks = (d_coupling_pairs.n_boundary_pairs + 255) / 256;
            if (boundary_blocks > 0) {
                complete_boundaries_kernel<dim, Number><<<boundary_blocks, 256, 0, stream>>>(
                    d_old_state_vector, d_dij, d_mij, d_cij, d_coupling_pairs);
            }
            
            Number initial_tau = Number(1e20);
            CUDA_CHECK(cudaMemcpyAsync(d_tau, &initial_tau, sizeof(Number),
                                    cudaMemcpyHostToDevice, stream));
            
            compute_diagonal_and_tau_kernel<dim, Number><<<diagonalConfig.blocksPerGrid, 
                                                        diagonalConfig.threadsPerBlock, 
                                                        diagonalConfig.sharedMemorySize, 
                                                        stream>>>(
                d_dij, d_tau, d_mij, d_mi, cfl, n_dofs);
            
            CUDA_CHECK(cudaMemcpyAsync(&tau, d_tau, sizeof(Number),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            tau = fmin(tau, tau_max);
        }

        low_order_update_kernel<dim, Number><<<lowOrderConfig.blocksPerGrid,
                                            lowOrderConfig.threadsPerBlock, 
                                            0, stream>>>(
            d_old_state_vector, d_new_state_vector, d_pressure, d_alpha_i, d_dij, d_pij, d_ri, d_bounds,
            d_sparsity, d_mij, d_mi, d_mi_inv, d_cij, d_precomputed,
            tau, measure_of_omega, d_stage_state_vector_0, d_stage_state_vector_1,
            stage_weight_0, stage_weight_1, stage_weight_acc, stage, n_dofs);

        compute_limiter_kernel<dim, Number><<<limiterConfig.blocksPerGrid, 
                                            limiterConfig.threadsPerBlock, 
                                            0, stream>>>(
            d_new_state_vector, d_pij, d_ri, d_lij, d_bounds,
            d_sparsity, d_mij, d_mi_inv, tau, n_dofs);

        high_order_update_iter1_kernel<dim, Number><<<highOrderConfig.blocksPerGrid, 
                                                    highOrderConfig.threadsPerBlock, 
                                                0, stream>>>(
            d_new_state_vector, d_pij, d_lij, d_lij_next, d_bounds, d_sparsity, n_dofs);

        std::swap(d_lij, d_lij_next);

        high_order_update_iter2_kernel<dim, Number><<<highOrderConfig.blocksPerGrid, 
                                                    highOrderConfig.threadsPerBlock, 
                                                    0, stream>>>(
            d_new_state_vector, d_pij, d_lij, d_sparsity, n_dofs);

        return tau;
    }
};

// ============================================================================
// Main time loop function
// ============================================================================
template<int dim, typename Number_cu>
Number_cu cuda_time_loop(
    const MijMatrix<Number_cu>& d_mij_matrix,
    const MiMatrix<Number_cu>& d_mi_matrix,
    const MiMatrixInverse<Number_cu>& d_mi_inv_matrix,
    const CijMatrix<dim, Number_cu>& d_cij_matrix,
    const Sparsity& d_sparsity,
    State<dim, Number_cu>& d_U,
    const BoundaryData<dim, Number_cu>& d_boundary_data,
    const CouplingPairs& d_coupling_pairs,
    Number_cu measure_of_omega,
    int n_dofs,
    int nnz_mij,
    int nnz_cij,
    const Configuration& config,
    const OfflineData<dim, double>& offline_data,
    VTUOutput<dim>* output_handler)
{
    // Create the streams
    cudaStream_t compute_stream, output_stream;
    CUDA_CHECK(cudaStreamCreate(&compute_stream));
    CUDA_CHECK(cudaStreamCreate(&output_stream));
    
    // Create async writer
    AsyncVTUWriter<dim> async_writer(output_handler);
    
    // Allocate device arrays
    State<dim, Number_cu> d_temp_0, d_temp_1, d_temp_2, d_temp_3, d_new_U;
    Pij<dim, Number_cu> d_pij;
    Ri<dim, Number_cu> d_ri;
    
    allocate_state(d_temp_0, n_dofs);
    allocate_state(d_temp_1, n_dofs);
    allocate_state(d_temp_2, n_dofs);
    allocate_state(d_temp_3, n_dofs);
    allocate_state(d_new_U, n_dofs);
    allocate_pij(d_pij, nnz_mij);
    allocate_ri(d_ri, n_dofs);
    
    Number_cu* d_pressure;
    Number_cu* d_speed_of_sound;
    Number_cu* d_precomputed;
    Number_cu* d_alpha_i;
    Number_cu* d_dij;
    Number_cu* d_bounds;
    Number_cu* d_lij;
    Number_cu* d_lij_next;
    
    CUDA_CHECK(cudaMalloc(&d_pressure, n_dofs * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_speed_of_sound, n_dofs * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_precomputed, n_dofs * 2 * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_alpha_i, n_dofs * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_dij, nnz_mij * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_bounds, n_dofs * 3 * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_lij, nnz_mij * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_lij_next, nnz_mij * sizeof(Number_cu)));
    
    // Set up inflow boundary state
    const Number_cu rho_inf = static_cast<Number_cu>(config.primitive_state[0]);
    const Number_cu u_inf = static_cast<Number_cu>(config.primitive_state[1]);
    const Number_cu v_inf = Number_cu(0);
    const Number_cu p_inf = static_cast<Number_cu>(config.primitive_state[2]);
    const Number_cu gamma = static_cast<Number_cu>(config.gamma);
    Number_cu E_inf = p_inf / (gamma - Number_cu(1)) + 
                      Number_cu(0.5) * rho_inf * (u_inf * u_inf + v_inf * v_inf);
    if constexpr (dim == 3) {
        const Number_cu w_inf = Number_cu(0);
        E_inf += Number_cu(0.5) * rho_inf * w_inf * w_inf;
    }
    const Number_cu mu = static_cast<Number_cu>(config.mu_reference);
    const Number_cu cv_inverse_kappa = static_cast<Number_cu>(config.cv_inverse_kappa_reference);
    
    Number_cu inflow_rho = rho_inf;
    Number_cu inflow_momentum_x = rho_inf * u_inf;
    Number_cu inflow_momentum_y = rho_inf * v_inf;
    const Number_cu w_inf = (dim == 3) ? Number_cu(0) : Number_cu(0);
    Number_cu inflow_momentum_z = (dim == 3) ? (rho_inf * w_inf) : Number_cu(0);
    Number_cu inflow_energy = E_inf;
    
    // Initial state preparation
    const int n_precomputation_cycles = 1;
    launch_prepare_state_vector<dim, Number_cu>(
        d_U, d_precomputed, d_boundary_data, 
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        n_dofs, n_precomputation_cycles, compute_stream);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    
    // Create stage executor
    StageExecutor<dim, Number_cu> stage_executor(
        d_U, d_temp_0, d_temp_1, d_temp_2, d_temp_3, d_new_U,
        d_pressure, d_speed_of_sound, d_precomputed, d_alpha_i,
        d_dij, d_pij, d_ri, d_bounds, d_lij, d_lij_next,
        d_mij_matrix, d_mi_matrix, d_mi_inv_matrix, d_cij_matrix, d_sparsity,
        d_boundary_data, d_coupling_pairs, 
        inflow_rho, inflow_momentum_x, inflow_momentum_y, inflow_momentum_z, inflow_energy,
        measure_of_omega, static_cast<Number_cu>(config.cfl_number),
        Number_cu(1.0), n_dofs, nnz_mij, mu, cv_inverse_kappa, offline_data, compute_stream);
    
    // Performance monitoring
    cudaEvent_t prof_start, prof_stop;
    CUDA_CHECK(cudaEventCreate(&prof_start));
    CUDA_CHECK(cudaEventCreate(&prof_stop));
    
    // Time loop parameters
    Number_cu t = Number_cu(0);
    unsigned int step = 0;
    unsigned int output_cycle = 0;
    const Number_cu output_interval = config.final_time / Number_cu(100);
    Number_cu next_output_time = output_interval;
    
    // Initial output - Convert SoA to AoS
    std::vector<std::array<double, dim + 2>> output_data(n_dofs);
    std::vector<Number_cu> h_rho(n_dofs);
    std::vector<Number_cu> h_momentum_x(n_dofs);
    std::vector<Number_cu> h_momentum_y(n_dofs);
    std::vector<Number_cu> h_momentum_z(n_dofs);
    std::vector<Number_cu> h_energy(n_dofs);
    
    CUDA_CHECK(cudaMemcpy(h_rho.data(), d_U.rho, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_momentum_x.data(), d_U.momentum_x, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMemcpy(h_momentum_y.data(), d_U.momentum_y, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMemcpy(h_momentum_z.data(), d_U.momentum_z, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaMemcpy(h_energy.data(), d_U.energy, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < n_dofs; ++i) {
        output_data[i][0] = static_cast<double>(h_rho[i]);
        output_data[i][1] = static_cast<double>(h_momentum_x[i]);
        if constexpr (dim >= 2) output_data[i][2] = static_cast<double>(h_momentum_y[i]);
        if constexpr (dim == 3) output_data[i][3] = static_cast<double>(h_momentum_z[i]);
        output_data[i][dim + 1] = static_cast<double>(h_energy[i]);
    }
    //async_writer.enqueue_write(std::move(output_data), output_cycle++, static_cast<double>(t));
    
    // Warmup iterations
    std::cout << "\nWarming up GPU..." << std::endl;
    for (int warmup = 0; warmup < 10; ++warmup) {
        Number_cu tau = stage_executor.execute_timestep(config.final_time);
        t += tau;
        step++;
        if (t >= config.final_time) break;
    }
    
    // Reset after warmup
    t = Number_cu(0);
    step = 0;
    CUDA_CHECK(cudaMemcpy(d_U.rho, h_rho.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_U.momentum_x, h_momentum_x.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMemcpy(d_U.momentum_y, h_momentum_y.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMemcpy(d_U.momentum_z, h_momentum_z.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_U.energy, h_energy.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
    
    std::cout << "\nStarting time integration..." << std::endl;
    std::cout << "  CFL: " << config.cfl_number << std::endl;
    
    const int n_steps_per_batch = 20;
    const int display_every_n_steps = 10000;
    const int measure_interval = 100;
    
    float total_kernel_time = 0;
    int total_steps_measured = 0;
    
    // Main time loop
    auto loop_start = std::chrono::high_resolution_clock::now();
    
    while (t < config.final_time) {
        if (step % measure_interval == 0 && step > 0) {
            cudaEventRecord(prof_start, compute_stream);
        }
        
        Number_cu tau_max_remaining = config.final_time - t;
        
        Number_cu batch_dt = 0;
        for (int batch_step = 0; batch_step < n_steps_per_batch; ++batch_step) {
            Number_cu tau = stage_executor.execute_timestep(tau_max_remaining);
            batch_dt += tau;
            
            if (t + batch_dt >= config.final_time) {
                break;
            }
        }
        
        if (step % measure_interval == 0 && step > 0) {
            cudaEventRecord(prof_stop, compute_stream);
            cudaEventSynchronize(prof_stop);
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, prof_start, prof_stop);
            total_kernel_time += milliseconds;
            total_steps_measured += n_steps_per_batch;
            
            float ms_per_step = milliseconds / n_steps_per_batch;
            float steps_per_second = 1000.0f / ms_per_step;
            
            if (step % display_every_n_steps == 0) {
                std::cout << "Step " << std::setw(6) << step
                          << ", t = " << std::setw(10) << std::fixed << std::setprecision(6) << t
                          << ", progress = " << std::fixed << std::setprecision(1)
                          << (100.0 * t / config.final_time) << "%"
                          << ", perf = " << std::scientific << std::setprecision(2) 
                          << steps_per_second << " steps/s"
                          << ", kernel time = " << std::fixed << std::setprecision(3)
                          << ms_per_step << " ms/step" << std::endl;
            }
        }
        
        t += batch_dt;
        step += n_steps_per_batch;
        
        if (t >= next_output_time || t >= config.final_time - 1e-10) {
            CUDA_CHECK(cudaMemcpyAsync(h_rho.data(), d_U.rho, n_dofs * sizeof(Number_cu), 
                                       cudaMemcpyDeviceToHost, output_stream));
            CUDA_CHECK(cudaMemcpyAsync(h_momentum_x.data(), d_U.momentum_x, n_dofs * sizeof(Number_cu), 
                                       cudaMemcpyDeviceToHost, output_stream));
            if constexpr (dim >= 2) {
                CUDA_CHECK(cudaMemcpyAsync(h_momentum_y.data(), d_U.momentum_y, n_dofs * sizeof(Number_cu), 
                                           cudaMemcpyDeviceToHost, output_stream));
            }
            if constexpr (dim == 3) {
                CUDA_CHECK(cudaMemcpyAsync(h_momentum_z.data(), d_U.momentum_z, n_dofs * sizeof(Number_cu), 
                                           cudaMemcpyDeviceToHost, output_stream));
            }
            CUDA_CHECK(cudaMemcpyAsync(h_energy.data(), d_U.energy, n_dofs * sizeof(Number_cu), 
                                       cudaMemcpyDeviceToHost, output_stream));
            
            CUDA_CHECK(cudaStreamSynchronize(output_stream));
            
            std::vector<std::array<double, dim + 2>> output_data(n_dofs);
            #pragma omp parallel for
            for (int i = 0; i < n_dofs; ++i) {
                output_data[i][0] = static_cast<double>(h_rho[i]);
                output_data[i][1] = static_cast<double>(h_momentum_x[i]);
                if constexpr (dim >= 2) output_data[i][2] = static_cast<double>(h_momentum_y[i]);
                if constexpr (dim == 3) output_data[i][3] = static_cast<double>(h_momentum_z[i]);
                output_data[i][dim + 1] = static_cast<double>(h_energy[i]);
            }    
            
            async_writer.enqueue_write(std::move(output_data), output_cycle++, static_cast<double>(t));
            next_output_time += output_interval;
        }
        
        if (step % display_every_n_steps == 0 && step % measure_interval != 0) {
            auto current_time = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration<double>(current_time - loop_start).count();
            double steps_per_second = step / elapsed;
            std::cout << "Step " << std::setw(6) << step
                      << ", t = " << std::setw(10) << std::fixed << std::setprecision(6) << t
                      << ", progress = " << std::fixed << std::setprecision(1)
                      << (100.0 * t / config.final_time) << "%"
                      << ", perf = " << std::scientific << std::setprecision(2) 
                      << steps_per_second << " steps/s" << std::endl;
        }
        
        if (step > 50000) {
            std::cerr << "\nMaximum iterations exceeded." << std::endl;
            break;
        }
    }
    
    async_writer.wait_for_completion();
    
    auto loop_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration<double>(loop_end - loop_start).count();
    
    std::cout << "\nSimulation complete!" << std::endl;
    std::cout << "  Final time: " << t << std::endl;
    std::cout << "  Total steps: " << step << std::endl;
    std::cout << "  Average dt: " << t/step << std::endl;
    std::cout << "  Wall time: " << total_time << " seconds" << std::endl;
    std::cout << "  Performance: " << step/total_time << " steps/second" << std::endl;
    
    if (total_steps_measured > 0) {
        float avg_ms_per_step = total_kernel_time / total_steps_measured;
        std::cout << "\n=== GPU Performance Summary ===" << std::endl;
        std::cout << "  Average kernel time: " << avg_ms_per_step << " ms/step" << std::endl;
        std::cout << "  Average throughput: " << 1000.0f / avg_ms_per_step << " steps/s" << std::endl;
        std::cout << "  DoFs processed per second: " << (n_dofs * 1000.0f / avg_ms_per_step) 
                  << " DoFs/s" << std::endl;
    }
    
    // Cleanup
    free_state(d_temp_0);
    free_state(d_temp_1);
    free_state(d_temp_2);
    free_state(d_temp_3);
    free_state(d_new_U);
    free_pij(d_pij);
    free_ri(d_ri);
    
    CUDA_CHECK(cudaFree(d_pressure));
    CUDA_CHECK(cudaFree(d_speed_of_sound));
    CUDA_CHECK(cudaFree(d_precomputed));
    CUDA_CHECK(cudaFree(d_alpha_i));
    CUDA_CHECK(cudaFree(d_dij));
    CUDA_CHECK(cudaFree(d_bounds));
    CUDA_CHECK(cudaFree(d_lij));
    CUDA_CHECK(cudaFree(d_lij_next));
    CUDA_CHECK(cudaEventDestroy(prof_start));
    CUDA_CHECK(cudaEventDestroy(prof_stop));
    CUDA_CHECK(cudaStreamDestroy(compute_stream));
    CUDA_CHECK(cudaStreamDestroy(output_stream));
    
    return t;
}

// Explicit instantiations
template float cuda_time_loop<2, float>(
    const MijMatrix<float>&, const MiMatrix<float>&, const MiMatrixInverse<float>&,
    const CijMatrix<2, float>&, const Sparsity&, State<2, float>&, 
    const BoundaryData<2, float>&, const CouplingPairs&, float, int, int, int, 
    const Configuration&, const OfflineData<2, double>&, VTUOutput<2>*);

template double cuda_time_loop<2, double>(
    const MijMatrix<double>&, const MiMatrix<double>&, const MiMatrixInverse<double>&,
    const CijMatrix<2, double>&, const Sparsity&, State<2, double>&, 
    const BoundaryData<2, double>&, const CouplingPairs&, double, int, int, int, 
    const Configuration&, const OfflineData<2, double>&, VTUOutput<2>*);

template float cuda_time_loop<3, float>(
    const MijMatrix<float>&, const MiMatrix<float>&, const MiMatrixInverse<float>&,
    const CijMatrix<3, float>&, const Sparsity&, State<3, float>&, 
    const BoundaryData<3, float>&, const CouplingPairs&, float, int, int, int, 
    const Configuration&, const OfflineData<3, double>&, VTUOutput<3>*);

template double cuda_time_loop<3, double>(
    const MijMatrix<double>&, const MiMatrix<double>&, const MiMatrixInverse<double>&,
    const CijMatrix<3, double>&, const Sparsity&, State<3, double>&, 
    const BoundaryData<3, double>&, const CouplingPairs&, double, int, int, int, 
    const Configuration&, const OfflineData<3, double>&, VTUOutput<3>*);