#ifndef CUDA_PARABOLIC_SOLVER_CUH
#define CUDA_PARABOLIC_SOLVER_CUH

#include <cuda_runtime.h>
#include "velocity_solver.cuh"
#include "data_struct.cuh"
#include "boundary_conditions.cuh"
#include "cell_based_operator.cuh"
#include "internal_energy_solver.cuh"
#include "parabolic_kernels.cuh"

template<int dim, typename Number>
struct ParabolicSolver {
    int n_dofs;
    cudaStream_t stream;
    State<dim, Number> init_U;
    
    VelocitySolver<dim, Number> velocity_solver;
    
    Number* d_velocity_rhs;
    Number* d_velocity_solution;

    Number* d_density;
    Number* d_velocity;
    Number* d_internal_energy;
    Number* d_internal_energy_rhs;
    
    BoundaryData<dim, Number> boundary_data;
    MiMatrix<Number> mi_matrix;    
    
    MijMatrix<Number> mass_matrix;
    CijMatrix<dim, Number> cij_matrix;
    Number mu;
    Number cv_inverse_kappa;

    ElementConnectivity<dim, Number>* element_connectivity;

    InternalEnergySolver<dim, Number> energy_solver;
    Number* d_internal_energy_solution;
    Number* d_internal_energy_low_order;
    Number* d_limiting_coefficients;    
    
    ParabolicSolver(int _n_dofs, Number _mu, Number _cv_inverse_kappa, cudaStream_t _stream):
        n_dofs(_n_dofs), 
        stream(_stream),
        velocity_solver(_n_dofs, _stream),
        energy_solver(_n_dofs, _stream),
        mu(_mu),
        cv_inverse_kappa(_cv_inverse_kappa),
        element_connectivity(nullptr)
    {
        cudaMalloc(&d_velocity_rhs, n_dofs * dim * sizeof(Number));
        cudaMalloc(&d_velocity_solution, n_dofs * dim * sizeof(Number));
        allocate_state(init_U, n_dofs);
        cudaMalloc(&d_density, n_dofs * sizeof(Number));
        cudaMalloc(&d_velocity, n_dofs * dim * sizeof(Number));
        cudaMalloc(&d_internal_energy, n_dofs * sizeof(Number));
        cudaMalloc(&d_internal_energy_rhs, n_dofs * sizeof(Number));
        cudaMalloc(&d_internal_energy_solution, n_dofs * sizeof(Number));
        cudaMalloc(&d_internal_energy_low_order, n_dofs * sizeof(Number));
        cudaMalloc(&d_limiting_coefficients, n_dofs * sizeof(Number));
    }
    
    ~ParabolicSolver() {
        cudaFree(d_velocity_rhs);
        cudaFree(d_velocity_solution);
        free_state(init_U);
        cudaFree(d_density);
        cudaFree(d_velocity);
        cudaFree(d_internal_energy);
        cudaFree(d_internal_energy_rhs);
        cudaFree(d_internal_energy_solution);
        cudaFree(d_internal_energy_low_order);
        cudaFree(d_limiting_coefficients);        

        if (element_connectivity != nullptr) {
            delete element_connectivity;
        }
    }

    void set_matrices(const MijMatrix<Number>& mij,
                      const MiMatrix<Number>& mi,
                      const CijMatrix<dim, Number>& cij,
                      const BoundaryData<dim, Number>& bd) {
        mass_matrix = mij;
        mi_matrix = mi;
        cij_matrix = cij;
        boundary_data = bd;
    }
    
    void set_element_connectivity(const OfflineData<dim, double>& offline_data) {
        if (element_connectivity != nullptr) {
            delete element_connectivity;
        }
        element_connectivity = new ElementConnectivity<dim, Number>();
        element_connectivity->build_from_triangulation(offline_data);
        
        std::cout << "  Element connectivity initialized:" << std::endl;
        std::cout << "    Elements: " << element_connectivity->n_elements << std::endl;
        std::cout << "    Nodes: " << element_connectivity->n_nodes << std::endl;
        
        if (element_connectivity->d_element_nodes == nullptr || 
            element_connectivity->d_jacobian_data == nullptr) {
            printf("ERROR: Element connectivity GPU memory allocation failed!\n");
        }
    }   
    
    void compute_step_parabolic(State<dim, Number>& d_U_old,
                                State<dim, Number>& d_U_new,
                                Number tau);
};

template<int dim, typename Number>
void ParabolicSolver<dim, Number>::compute_step_parabolic(
    State<dim, Number>& d_U_old, 
    State<dim, Number>& d_U_new, 
    Number tau)
{
    const int threads = 256;
    const int blocks = (n_dofs + threads - 1) / threads;

    /*
     * Step 1: Build velocity RHS and solve velocity system
     */
    
    build_velocity_rhs_kernel<dim, Number><<<blocks, threads, 0, stream>>>(
        d_U_old,
        init_U,
        mi_matrix.values,
        d_density,
        d_velocity,
        d_velocity_rhs,
        d_internal_energy,
        boundary_data,
        n_dofs);
    
    // Solve velocity update
    Number lambda = -Number(2.0/3.0) * mu;
    velocity_solver.set_system_matrices(
        d_density,
        mi_matrix.values,
        cij_matrix.row_offsets,
        cij_matrix.col_indices,
        cij_matrix.values,
        mass_matrix.values,
        mu,
        lambda);
    
    cudaMemcpyAsync(d_velocity_solution, d_velocity,
                    n_dofs * dim * sizeof(Number),
                    cudaMemcpyDeviceToDevice, stream);
    
    Number tolerance = std::is_same_v<Number, float> ? Number(1e-5) : Number(1e-8);
    int iterations = velocity_solver.solve(
        d_velocity_solution,
        d_velocity_rhs,
        tau,
        tolerance,
        1000);
    
    if (iterations < 0) {
        printf("Warning: Velocity solver failed to converge\n");
    }

    /*
     * Step 2-a: Build internal energy RHS
     */
    
    cudaMemsetAsync(d_internal_energy_rhs, 0, n_dofs * sizeof(Number), stream);
    
    // Kernel 1: Compute viscous heating
    const int element_blocks = (element_connectivity->n_elements + 127) / 128;
    constexpr int nodes_per_elem = (dim == 2) ? 4 : 8;
    constexpr size_t shared_mem_size = 128 * nodes_per_elem * dim * sizeof(Number);
    compute_viscous_heating_kernel<dim, Number, nodes_per_elem><<<element_blocks, 128, shared_mem_size, stream>>>(
        d_velocity_solution,
        d_internal_energy_rhs,
        element_connectivity->d_element_nodes,
        element_connectivity->d_jacobian_data,
        mi_matrix.values,
        mu,
        lambda,
        element_connectivity->n_elements,
        n_dofs);
    
    // Kernel 2: Complete RHS
    complete_internal_energy_rhs_kernel<dim, Number><<<blocks, threads, 0, stream>>>(
        d_U_old,
        init_U,
        d_velocity,
        d_velocity_solution,
        d_density,
        d_internal_energy,
        d_internal_energy_rhs,
        mi_matrix.values,
        boundary_data,
        tau,
        false,
        n_dofs);
    
    /*
     * Step 2-b: Solve internal energy system
     */
    
    energy_solver.set_system_matrices(
        d_density,
        mi_matrix.values,
        cij_matrix.row_offsets,
        cij_matrix.col_indices,
        cij_matrix.values,
        mass_matrix.values,
        cv_inverse_kappa);
    
    cudaMemcpyAsync(d_internal_energy_solution, d_internal_energy,
                    n_dofs * sizeof(Number),
                    cudaMemcpyDeviceToDevice, stream);
    
    Number tolerance_energy = std::is_same_v<Number, float> ? Number(1e-6) : Number(1e-10);
    int energy_iterations = energy_solver.solve(
        d_internal_energy_solution,
        d_internal_energy_rhs,
        tau,
        tolerance_energy,
        1000);
    
    if (energy_iterations < 0) {
        printf("Warning: Energy solver failed to converge\n");
    }
    
    /*
     * Step 3: Update conservative state
     */

    copy_density_kernel<dim, Number><<<blocks, threads, 0, stream>>>(
        d_U_new, d_U_old, n_dofs);

    update_momentum_from_velocity_kernel<dim, Number><<<blocks, threads, 0, stream>>>(
        d_U_new, d_U_old, d_velocity_solution, n_dofs);

    update_total_energy_from_internal_energy_kernel<dim, Number><<<blocks, threads, 0, stream>>>(
        d_U_new, d_U_old, d_internal_energy_solution, n_dofs);
}

#endif