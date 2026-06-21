

#include <iostream>
#include <fstream>
#include <cuda_bf16.h>
#include <deal.II/grid/tria.h>
#include "mesh.h"
#include "offline_data.h"
#include "data_struct.cuh"
#include "becker_solution.h"


using Number = double;      

using Number_cu = float;    
constexpr int dim = 2;      



template<int dim, typename Number, typename Number_cu>
void transfer_offline_data_to_gpu(
    const OfflineData<dim>& offline_data,
    MijMatrix<Number_cu>& d_mij_matrix,
    MiMatrix<Number_cu>& d_mi_matrix,
    MiMatrixInverse<Number_cu>& d_mi_inv_matrix,
    CijMatrix<dim, Number_cu>& d_cij_matrix,
    Sparsity& d_sparsity,
    int& nnz_mij,
    int& nnz_cij)
{
    const int n_dofs = offline_data.dof_handler.n_dofs();
    nnz_mij = 0;
    for (const auto& row : offline_data.sparsity)
        nnz_mij += row.size();
    nnz_cij = nnz_mij;

    std::vector<int> row_offsets(n_dofs + 1);
    std::vector<int> col_indices_mij(nnz_mij);
    std::vector<int> col_indices_cij(nnz_cij);
    std::vector<Number_cu> mass_values(nnz_mij);
    std::vector<Number_cu> cij_values(nnz_cij * dim);

    int offset = 0;
    for (int i = 0; i < n_dofs; ++i) {
        row_offsets[i] = offset;
        const auto& sparsity_row = offline_data.sparsity[i];

        for (size_t col_idx = 0; col_idx < sparsity_row.size(); ++col_idx) {
            col_indices_mij[offset] = sparsity_row[col_idx];
            col_indices_cij[offset] = sparsity_row[col_idx];
            mass_values[offset] = static_cast<Number_cu>(offline_data.mass_matrix[i][col_idx]);

            for (int d = 0; d < dim; ++d)
                cij_values[offset * dim + d] = static_cast<Number_cu>(offline_data.c_ij[i][col_idx][d]);

            offset++;
        }
    }
    row_offsets[n_dofs] = offset;

    CUDA_CHECK(cudaMalloc(&d_sparsity.row_offsets, (n_dofs + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sparsity.col_indices, nnz_mij * sizeof(int)));

    CUDA_CHECK(cudaMalloc(&d_mij_matrix.row_offsets, (n_dofs + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mij_matrix.col_indices, nnz_mij * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mij_matrix.values, nnz_mij * sizeof(Number_cu)));

    CUDA_CHECK(cudaMalloc(&d_cij_matrix.row_offsets, (n_dofs + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cij_matrix.col_indices, nnz_cij * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_cij_matrix.values, nnz_cij * dim * sizeof(Number_cu)));

    CUDA_CHECK(cudaMalloc(&d_mi_matrix.values, n_dofs * sizeof(Number_cu)));
    CUDA_CHECK(cudaMalloc(&d_mi_inv_matrix.values, n_dofs * sizeof(Number_cu)));

    std::vector<Number_cu> lumped_mass(n_dofs);
    std::vector<Number_cu> lumped_mass_inv(n_dofs);
    for (int i = 0; i < n_dofs; ++i) {
        lumped_mass[i] = static_cast<Number_cu>(offline_data.lumped_mass_matrix[i]);
        lumped_mass_inv[i] = static_cast<Number_cu>(offline_data.lumped_mass_matrix_inverse[i]);
    }

    CUDA_CHECK(cudaMemcpy(d_sparsity.row_offsets, row_offsets.data(), (n_dofs + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sparsity.col_indices, col_indices_mij.data(), nnz_mij * sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_mij_matrix.row_offsets, row_offsets.data(), (n_dofs + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mij_matrix.col_indices, col_indices_mij.data(), nnz_mij * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mij_matrix.values, mass_values.data(), nnz_mij * sizeof(Number_cu), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_cij_matrix.row_offsets, row_offsets.data(), (n_dofs + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cij_matrix.col_indices, col_indices_cij.data(), nnz_cij * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cij_matrix.values, cij_values.data(), nnz_cij * dim * sizeof(Number_cu), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemcpy(d_mi_matrix.values, lumped_mass.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice)); 
    CUDA_CHECK(cudaMemcpy(d_mi_inv_matrix.values, lumped_mass_inv.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
}


template<int dim, typename Number, typename Number_cu>
void transfer_boundary_data_to_gpu(
    const OfflineData<dim>& offline_data,
    BoundaryData<dim, Number_cu>& d_boundary_data,
    CouplingPairs& d_coupling_pairs,
    Number_cu& measure_of_omega,
    int n_dofs)
{
    std::vector<int> boundary_dofs;
    std::vector<int> boundary_ids;
    std::vector<Number_cu> boundary_normals;

    for (const auto& bd : offline_data.boundary_map) {
        boundary_dofs.push_back(bd.dof_index);
        boundary_ids.push_back(bd.id);
        for (int d = 0; d < dim; ++d) {
            boundary_normals.push_back(static_cast<Number_cu>(bd.normal[d]));
        }
    }

    d_boundary_data.n_boundary_dofs = boundary_dofs.size();

    std::vector<int> bc_type(n_dofs, -1);
    std::vector<int> bc_index(n_dofs, -1);

    for (size_t b = 0; b < boundary_dofs.size(); ++b) {
        const int dof = boundary_dofs[b];

        if (bc_type[dof] == -1) {
            bc_type[dof] = boundary_ids[b];
            bc_index[dof] = static_cast<int>(b);
        }
    }

    CUDA_CHECK(cudaMalloc(&d_boundary_data.bc_type, n_dofs * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_boundary_data.bc_index, n_dofs * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_boundary_data.bc_type, bc_type.data(),
                          n_dofs * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_boundary_data.bc_index, bc_index.data(),
                          n_dofs * sizeof(int), cudaMemcpyHostToDevice));

    if (d_boundary_data.n_boundary_dofs > 0) {
        CUDA_CHECK(cudaMalloc(&d_boundary_data.boundary_dofs,
                              d_boundary_data.n_boundary_dofs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_boundary_data.boundary_ids,
                              d_boundary_data.n_boundary_dofs * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_boundary_data.boundary_normals,
                              d_boundary_data.n_boundary_dofs * dim * sizeof(Number_cu)));

        CUDA_CHECK(cudaMemcpy(d_boundary_data.boundary_dofs, boundary_dofs.data(),
                              d_boundary_data.n_boundary_dofs * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_boundary_data.boundary_ids, boundary_ids.data(),
                              d_boundary_data.n_boundary_dofs * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_boundary_data.boundary_normals, boundary_normals.data(),
                              d_boundary_data.n_boundary_dofs * dim * sizeof(Number_cu),
                              cudaMemcpyHostToDevice));
    }

    std::vector<int> internal_pairs_flat;
    std::vector<int> boundary_pairs_flat;

    for (const auto& [i, col_idx, j] : offline_data.coupling_internal_pairs) {
        internal_pairs_flat.push_back(i);
        internal_pairs_flat.push_back(col_idx);
        internal_pairs_flat.push_back(j);
    }

    for (const auto& [i, col_idx, j] : offline_data.coupling_boundary_pairs) {
        boundary_pairs_flat.push_back(i);
        boundary_pairs_flat.push_back(col_idx);
        boundary_pairs_flat.push_back(j);
    }

    d_coupling_pairs.n_internal_pairs = offline_data.coupling_internal_pairs.size();
    d_coupling_pairs.n_boundary_pairs = offline_data.coupling_boundary_pairs.size();

    if (d_coupling_pairs.n_internal_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_coupling_pairs.internal_pairs, 
                              d_coupling_pairs.n_internal_pairs * 3 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_coupling_pairs.internal_pairs, 
                              internal_pairs_flat.data(),
                              d_coupling_pairs.n_internal_pairs * 3 * sizeof(int), 
                              cudaMemcpyHostToDevice));
    }

    if (d_coupling_pairs.n_boundary_pairs > 0) {
        CUDA_CHECK(cudaMalloc(&d_coupling_pairs.boundary_pairs, 
                              d_coupling_pairs.n_boundary_pairs * 3 * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(d_coupling_pairs.boundary_pairs, 
                              boundary_pairs_flat.data(),
                              d_coupling_pairs.n_boundary_pairs * 3 * sizeof(int), 
                              cudaMemcpyHostToDevice));
    }
    
    measure_of_omega = static_cast<Number_cu>(offline_data.measure_of_omega);
}


template <int d>
static void compute_mesh(dealii::Triangulation<d>& triangulation,
                                 const Configuration& config)
{
    const std::string& g = config.geometry_type;
    if constexpr (d == 2) {
        if      (g == "rectangle")             MeshGenerator::create_rectangle_mesh(triangulation, config);
        else if (g == "cylinder")              MeshGenerator::create_cylinder_mesh(triangulation, config);
        else if (g == "channel with cylinder") MeshGenerator::create_channel_with_cylinder_mesh(triangulation, config);
        else if (g == "airfoil")               MeshGenerator::create_airfoil_mesh(triangulation, config);
        else if (g == "capsule")               MeshGenerator::create_capsule_mesh(triangulation, config);
        else if (g == "mesh_file")             MeshGenerator::create_mesh_from_file(triangulation, config);
        else throw std::runtime_error("Unknown 2D geometry type: " + g);
    } else {  
        if      (g == "sphere_channel")        MeshGenerator::create_sphere_in_channel_mesh(triangulation, config);
        else if (g == "cylinder")              MeshGenerator::create_cylinder_mesh(triangulation, config);
        else if (g == "channel with cylinder") MeshGenerator::create_channel_with_cylinder_mesh(triangulation, config);
        else if (g == "capsule")               MeshGenerator::create_capsule_mesh(triangulation, config);
        else if (g == "mesh_file")             MeshGenerator::create_mesh_from_file(triangulation, config);
        else throw std::runtime_error("Unknown 3D geometry type: " + g);
    }
}

int main(int argc, char* argv[]) {
    try {

        Configuration config;

        std::string param_file = "../cases/ns-mach3-cylinder-2d.prm";

        if (argc > 1) {
            param_file = argv[1];
        }
        config.read_parameters(param_file);

        std::cout << "=== GPU-accelerated Navier-Stokes solver ===" << std::endl;
        std::cout << "CPU Precision: " << (sizeof(Number) == 4 ? "float" : "double") << std::endl;
        std::cout << "GPU Precision: " << (sizeof(Number_cu) == 4 ? "float" : "double") << std::endl;
        std::cout << "Dimension: " << dim << "D" << std::endl;
        std::cout << "Data Structure: Structure of Arrays (SoA)" << std::endl;
        std::cout << "Configuration:" << std::endl;
        std::cout << "  Final time: " << config.final_time << std::endl;
        std::cout << "  CFL min: " << config.cfl_min << std::endl;
        std::cout << "  CFL max: " << config.cfl_max << std::endl;
        std::cout << "  CFL number: " << config.cfl_number << std::endl;
        std::cout << "  Mesh refinement: " << config.mesh_refinement << std::endl;
        std::cout << "  final_time = " << config.final_time << std::endl;
        std::cout << "  timer_granularity = " << config.timer_granularity << std::endl;

        dealii::Triangulation<dim> triangulation;
        compute_mesh<dim>(triangulation, config);

        std::cout << "Mesh: " << triangulation.n_active_cells() << " cells, "
                  << triangulation.n_vertices() << " vertices" << std::endl;

        OfflineData<dim> offline_data(triangulation);
        const int n_dofs = offline_data.dof_handler.n_dofs();
        std::cout << "DoFs: " << n_dofs << std::endl;

        VTUOutput<dim> output(offline_data.dof_handler, config.basename, offline_data);

        std::cout << "\nTransferring data to device..." << std::endl;
        MijMatrix<Number_cu> d_mass_matrix;
        MiMatrix<Number_cu> d_lumped_mass;
        MiMatrixInverse<Number_cu> d_lumped_mass_inv;
        CijMatrix<dim, Number_cu> d_cij;
        Sparsity d_sparsity;
        BoundaryData<dim, Number_cu> d_boundary_data;
        CouplingPairs d_coupling_pairs;
        State<dim, Number_cu> d_U;
        int nnz_mij, nnz_cij;
        Number_cu measure_of_omega;

        transfer_offline_data_to_gpu<dim, Number, Number_cu>(
            offline_data, d_mass_matrix, d_lumped_mass, 
            d_lumped_mass_inv, d_cij, d_sparsity, nnz_mij, nnz_cij);
        std::cout << "  Non-zeros in M_ij: " << nnz_mij << std::endl;
        std::cout << "  Non-zeros in C_ij: " << nnz_cij << std::endl;

        transfer_boundary_data_to_gpu<dim, Number, Number_cu>(
            offline_data, d_boundary_data, d_coupling_pairs, measure_of_omega, n_dofs);
        std::cout << "  Boundary DoFs: " << d_boundary_data.n_boundary_dofs << std::endl;
        std::cout << "  Internal coupling pairs: " << d_coupling_pairs.n_internal_pairs << std::endl;
        std::cout << "  Boundary coupling pairs: " << d_coupling_pairs.n_boundary_pairs << std::endl;

        allocate_state(d_U, n_dofs);

        std::vector<Number_cu> h_rho(n_dofs);
        std::vector<Number_cu> h_momentum_x(n_dofs);
        std::vector<Number_cu> h_momentum_y(n_dofs);
        std::vector<Number_cu> h_momentum_z(n_dofs);
        std::vector<Number_cu> h_energy(n_dofs);

        std::cout << "rho: " << config.primitive_state[0] << std::endl;
        std::cout << "u: " << config.primitive_state[1] << std::endl;
        std::cout << "pressure: " << config.primitive_state[2] << std::endl;

        if (config.becker_verification) {

            std::cout << "\nSetting Becker solution initial condition..." << std::endl;
            BeckerSolutionCPU becker(config.gamma,
                                      config.becker_velocity_left,
                                      config.becker_velocity_right,
                                      config.becker_density_left,
                                      config.becker_velocity_galilean,
                                      config.mu,
                                      config.becker_position);

            for (int i = 0; i < n_dofs; ++i) {
                const double x = offline_data.node_positions[i][0];
                auto state = becker.compute(x, 0.0);  
                h_rho[i] = static_cast<Number_cu>(state[0]);
                h_momentum_x[i] = static_cast<Number_cu>(state[1]);
                h_momentum_y[i] = static_cast<Number_cu>(state[2]);
                if constexpr (dim == 3)
                    h_momentum_z[i] = Number_cu(0);
                h_energy[i] = static_cast<Number_cu>(state[3]);
            }

            {
                Number_cu rho_min = h_rho[0], rho_max = h_rho[0];
                Number_cu mx_min = h_momentum_x[0], mx_max = h_momentum_x[0];
                double xmin_rho = offline_data.node_positions[0][0];
                double xmax_rho = offline_data.node_positions[0][0];
                for (int i = 1; i < n_dofs; ++i) {
                    if (h_rho[i] < rho_min) { rho_min = h_rho[i]; xmin_rho = offline_data.node_positions[i][0]; }
                    if (h_rho[i] > rho_max) { rho_max = h_rho[i]; xmax_rho = offline_data.node_positions[i][0]; }
                    if (h_momentum_x[i] < mx_min) mx_min = h_momentum_x[i];
                    if (h_momentum_x[i] > mx_max) mx_max = h_momentum_x[i];
                }
                std::cout << "  Becker IC: rho_min=" << rho_min << " at x=" << xmin_rho
                          << ", rho_max=" << rho_max << " at x=" << xmax_rho
                          << ", mx=[" << mx_min << "," << mx_max << "]" << std::endl;

                auto state_L = becker.compute(-0.25, 0.0);
                auto state_R = becker.compute(+0.25, 0.0);
                std::cout << "  Becker exact @ x=-0.25,t=0: rho=" << state_L[0] << ", mx=" << state_L[1] << std::endl;
                std::cout << "  Becker exact @ x=+0.25,t=0: rho=" << state_R[0] << ", mx=" << state_R[1] << std::endl;
                auto state_LT = becker.compute(-0.25, config.final_time);
                auto state_RT = becker.compute(+0.25, config.final_time);
                std::cout << "  Becker exact @ x=-0.25,t=Tf: rho=" << state_LT[0] << ", mx=" << state_LT[1] << std::endl;
                std::cout << "  Becker exact @ x=+0.25,t=Tf: rho=" << state_RT[0] << ", mx=" << state_RT[1] << std::endl;
            }
        } else {

            const Number_cu vel_mag = static_cast<Number_cu>(config.primitive_state[1]);
            const Number_cu p = static_cast<Number_cu>(config.primitive_state[2]);
            const Number_cu rho = static_cast<Number_cu>(config.primitive_state[0]);
            const Number_cu gamma = static_cast<Number_cu>(config.gamma);

            const Number_cu u = vel_mag * static_cast<Number_cu>(config.direction[0]);
            const Number_cu v = vel_mag * static_cast<Number_cu>(config.direction[1]);
            const Number_cu w = vel_mag * static_cast<Number_cu>(config.direction[2]);

            Number_cu kinetic_energy;
            if constexpr (dim == 2) {
                kinetic_energy = Number_cu(0.5) * rho * (u * u + v * v);
            } else {
                kinetic_energy = Number_cu(0.5) * rho * (u * u + v * v + w * w);
            }
            const Number_cu E = p / (gamma - Number_cu(1)) + kinetic_energy;

            for (int i = 0; i < n_dofs; ++i) {
                h_rho[i] = rho;
                h_momentum_x[i] = rho * u;
                h_momentum_y[i] = rho * v;
                if constexpr (dim == 3)
                    h_momentum_z[i] = rho * w;
                h_energy[i] = E;
            }
        }

        CUDA_CHECK(cudaMemcpy(d_U.rho, h_rho.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_U.momentum_x, h_momentum_x.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
        if constexpr (dim >= 2) {
            CUDA_CHECK(cudaMemcpy(d_U.momentum_y, h_momentum_y.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
        }
        if constexpr (dim == 3) {
            CUDA_CHECK(cudaMemcpy(d_U.momentum_z, h_momentum_z.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
        }
        CUDA_CHECK(cudaMemcpy(d_U.energy, h_energy.data(), n_dofs * sizeof(Number_cu), cudaMemcpyHostToDevice));
        
        std::cout << "Initial conditions transferred" << std::endl;

        const auto t0 = std::chrono::high_resolution_clock::now();
        const std::time_t time_now = std::chrono::system_clock::to_time_t(t0);
        std::cout << "\nStarting time loop, at time: " << std::ctime(&time_now);
        Number_cu t = cuda_time_loop<dim, Number_cu>(
            d_mass_matrix,
            d_lumped_mass,
            d_lumped_mass_inv,
            d_cij,
            d_sparsity,
            d_U,
            d_boundary_data,
            d_coupling_pairs,
            measure_of_omega,
            n_dofs,
            nnz_mij,
            nnz_cij,
            config,
            offline_data,
            &output);

        std::cout << "\nSimulation complete!" << std::endl;
        std::cout << "Final time: " << t << std::endl;
        const auto t1 = std::chrono::high_resolution_clock::now();
        const auto duration = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Comp. time (sec.): " << duration << std::endl;

        if (config.becker_verification) {
            std::cout << "\n=== Becker Verification Error Computation ===" << std::endl;

            std::vector<Number_cu> final_rho(n_dofs);
            std::vector<Number_cu> final_mx(n_dofs);
            std::vector<Number_cu> final_my(n_dofs);
            std::vector<Number_cu> final_energy(n_dofs);
            CUDA_CHECK(cudaMemcpy(final_rho.data(), d_U.rho, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(final_mx.data(), d_U.momentum_x, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
            if constexpr (dim >= 2)
                CUDA_CHECK(cudaMemcpy(final_my.data(), d_U.momentum_y, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(final_energy.data(), d_U.energy, n_dofs * sizeof(Number_cu), cudaMemcpyDeviceToHost));

            BeckerSolutionCPU becker(config.gamma,
                                      config.becker_velocity_left,
                                      config.becker_velocity_right,
                                      config.becker_density_left,
                                      config.becker_velocity_galilean,
                                      config.mu,
                                      config.becker_position);

            const double t_final = static_cast<double>(t);

            double L1_error = 0.0, L2_error = 0.0, Linf_error = 0.0;
            double L1_exact = 0.0, L2_exact = 0.0, Linf_exact = 0.0;

            double L1_rho = 0.0, L1_mx = 0.0, L1_my = 0.0, L1_E = 0.0;
            double Linf_rho = 0.0, Linf_mx = 0.0, Linf_my = 0.0, Linf_E = 0.0;
            int idx_Linf_rho = 0;

            int nan_count = 0;
            int first_nan_idx = -1;
            for (int i = 0; i < n_dofs; ++i) {

                if (offline_data.periodic_master[i] != static_cast<unsigned int>(i)) continue;
                const double x = offline_data.node_positions[i][0];
                auto exact = becker.compute(x, t_final);
                const double mi = offline_data.lumped_mass_matrix[i];

                if (std::isnan(static_cast<double>(final_rho[i])) ||
                    std::isnan(static_cast<double>(final_mx[i])) ||
                    std::isnan(static_cast<double>(final_my[i])) ||
                    std::isnan(static_cast<double>(final_energy[i]))) {
                    if (first_nan_idx < 0) first_nan_idx = i;
                    ++nan_count;
                    continue;
                }

                double err_rho = std::abs(static_cast<double>(final_rho[i]) - exact[0]);
                double err_mx  = std::abs(static_cast<double>(final_mx[i])  - exact[1]);
                double err_my  = std::abs(static_cast<double>(final_my[i])  - exact[2]);
                double err_E   = std::abs(static_cast<double>(final_energy[i]) - exact[3]);
                double err_total = err_rho + err_mx + err_my + err_E;

                double exact_total = std::abs(exact[0]) + std::abs(exact[1]) +
                                     std::abs(exact[2]) + std::abs(exact[3]);

                L1_error += err_total * mi;
                L2_error += err_total * err_total * mi;
                Linf_error = std::max(Linf_error, err_total);

                L1_exact += exact_total * mi;
                L2_exact += exact_total * exact_total * mi;
                Linf_exact = std::max(Linf_exact, exact_total);

                L1_rho += err_rho * mi; L1_mx += err_mx * mi;
                L1_my  += err_my  * mi; L1_E  += err_E  * mi;
                if (err_rho > Linf_rho) { Linf_rho = err_rho; idx_Linf_rho = i; }
                Linf_mx  = std::max(Linf_mx,  err_mx);
                Linf_my  = std::max(Linf_my,  err_my);
                Linf_E   = std::max(Linf_E,   err_E);
            }
            std::cout << std::scientific << std::setprecision(3);
            if (nan_count > 0) {
                std::cout << "  *** NaN DETECTED: " << nan_count << " DOFs have NaN; first at idx=" << first_nan_idx
                          << " (x=" << offline_data.node_positions[first_nan_idx][0]
                          << ", y=" << offline_data.node_positions[first_nan_idx][1] << ")"
                          << " rho=" << static_cast<double>(final_rho[first_nan_idx])
                          << " mx=" << static_cast<double>(final_mx[first_nan_idx])
                          << " E=" << static_cast<double>(final_energy[first_nan_idx]) << std::endl;
            }
            std::cout << "  Per-component L1 errors:  rho=" << L1_rho
                      << " mx=" << L1_mx << " my=" << L1_my << " E=" << L1_E << std::endl;
            std::cout << "  Per-component Linf errors: rho=" << Linf_rho
                      << " mx=" << Linf_mx << " my=" << Linf_my << " E=" << Linf_E << std::endl;
            {
                const double x_Lr = offline_data.node_positions[idx_Linf_rho][0];
                const double y_Lr = (dim >= 2 ? offline_data.node_positions[idx_Linf_rho][1] : 0.0);
                auto exact_at = becker.compute(x_Lr, t_final);
                std::cout << "  Linf rho location: idx=" << idx_Linf_rho
                          << " at (x=" << x_Lr << ", y=" << y_Lr << ")"
                          << " computed_rho=" << static_cast<double>(final_rho[idx_Linf_rho])
                          << " exact_rho=" << exact_at[0] << std::endl;

                double y_min_at_x = 0, y_max_at_x = 0;
                double rho_min_at_x = 1e300, rho_max_at_x = -1e300;
                int count_at_x = 0;
                for (int i = 0; i < n_dofs; ++i) {
                    if (std::abs(offline_data.node_positions[i][0] - x_Lr) < 1e-10) {
                        const double yp = offline_data.node_positions[i][1];
                        const double rp = static_cast<double>(final_rho[i]);
                        if (rp < rho_min_at_x) { rho_min_at_x = rp; y_min_at_x = yp; }
                        if (rp > rho_max_at_x) { rho_max_at_x = rp; y_max_at_x = yp; }
                        ++count_at_x;
                    }
                }
                std::cout << "  y-invariance @ x=" << x_Lr << ": rho_min=" << rho_min_at_x
                          << " @ y=" << y_min_at_x << ", rho_max=" << rho_max_at_x
                          << " @ y=" << y_max_at_x << " (across " << count_at_x << " DOFs)" << std::endl;
            }

            L2_error = std::sqrt(L2_error);
            L2_exact = std::sqrt(L2_exact);

            double L1_normalized = (L1_exact > 0) ? L1_error / L1_exact : L1_error;
            double L2_normalized = (L2_exact > 0) ? L2_error / L2_exact : L2_error;
            double Linf_normalized = (Linf_exact > 0) ? Linf_error / Linf_exact : Linf_error;

            std::cout << std::scientific << std::setprecision(6);
            std::cout << "  DOFs:           " << n_dofs << std::endl;
            std::cout << "  Final time:     " << t_final << std::endl;
            std::cout << "  L1 error:       " << L1_normalized << std::endl;
            std::cout << "  L2 error:       " << L2_normalized << std::endl;
            std::cout << "  Linf error:     " << Linf_normalized << std::endl;
            std::cout << "  GPU precision:  " << (sizeof(Number_cu) == 4 ? "float" : "double") << std::endl;

            std::string csv_file = "becker_convergence.csv";
            bool file_exists = false;
            {
                std::ifstream check(csv_file);
                file_exists = check.good();
            }

            std::ofstream csv(csv_file, std::ios::app);
            if (!file_exists) {
                csv << "n_dofs,t_final,L1_error,L2_error,Linf_error,precision,refinement,subdivisions_x" << std::endl;
            }
            csv << std::scientific << std::setprecision(16);
            csv << n_dofs << ","
                << t_final << ","
                << L1_normalized << ","
                << L2_normalized << ","
                << Linf_normalized << ","
                << (sizeof(Number_cu) == 4 ? "float" : "double") << ","
                << config.mesh_refinement << ","
                << config.rect_subdivisions_x << std::endl;
            csv.close();

            std::cout << "  Results appended to " << csv_file << std::endl;
        }

        free_state(d_U);
        CUDA_CHECK(cudaFree(d_sparsity.row_offsets));
        CUDA_CHECK(cudaFree(d_sparsity.col_indices));
        CUDA_CHECK(cudaFree(d_mass_matrix.row_offsets));
        CUDA_CHECK(cudaFree(d_mass_matrix.col_indices));
        CUDA_CHECK(cudaFree(d_mass_matrix.values));
        CUDA_CHECK(cudaFree(d_cij.row_offsets));
        CUDA_CHECK(cudaFree(d_cij.col_indices));
        CUDA_CHECK(cudaFree(d_cij.values));
        CUDA_CHECK(cudaFree(d_lumped_mass.values));
        CUDA_CHECK(cudaFree(d_lumped_mass_inv.values));
        CUDA_CHECK(cudaFree(d_boundary_data.boundary_dofs));
        CUDA_CHECK(cudaFree(d_boundary_data.boundary_ids));
        CUDA_CHECK(cudaFree(d_boundary_data.boundary_normals));
        CUDA_CHECK(cudaFree(d_boundary_data.bc_type));
        CUDA_CHECK(cudaFree(d_boundary_data.bc_index));
        if (d_coupling_pairs.n_internal_pairs > 0)
            CUDA_CHECK(cudaFree(d_coupling_pairs.internal_pairs));
        if (d_coupling_pairs.n_boundary_pairs > 0)
            CUDA_CHECK(cudaFree(d_coupling_pairs.boundary_pairs));
            
    } catch (std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
    }
    
    return 0;
}