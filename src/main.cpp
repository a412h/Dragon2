
#include <iostream>
#include <cuda_bf16.h>
#include <deal.II/grid/tria.h>
#include "mesh_reader.h"
#include "offline_data.h"
#include "data_struct.cuh"

using Number = double;
using Number_cu = float;

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
        bc_type[dof] = boundary_ids[b];
        bc_index[dof] = static_cast<int>(b);
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

template<int dim>
void run_simulation(const Configuration& config) {

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
    std::cout << "  timer_granularity = " << config.timer_granularity << std::endl;

    dealii::Triangulation<dim> triangulation;
    BoundaryMapping bc_mapping = parse_boundary_mapping(
        config.boundary_mapping,
        config.default_boundary_condition);
    GmshMeshReader<dim>::read_mesh(
        triangulation,
        config.mesh_file_path,
        bc_mapping);

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
    Number_cu t;
    if (config.time_scheme == "ssprk33") {
        t = cuda_time_loop<dim, Number_cu, TimeScheme::SSPRK33_CN>(
            d_mass_matrix, d_lumped_mass, d_lumped_mass_inv, d_cij,
            d_sparsity, d_U, d_boundary_data, d_coupling_pairs,
            measure_of_omega, n_dofs, nnz_mij, nnz_cij,
            config, offline_data, &output);
    } else {
        t = cuda_time_loop<dim, Number_cu, TimeScheme::ERK33_CN>(
            d_mass_matrix, d_lumped_mass, d_lumped_mass_inv, d_cij,
            d_sparsity, d_U, d_boundary_data, d_coupling_pairs,
            measure_of_omega, n_dofs, nnz_mij, nnz_cij,
            config, offline_data, &output);
    }

    std::cout << "\nSimulation complete!" << std::endl;
    std::cout << "Final time: " << t << std::endl;
    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "Comp. time (sec.): " << duration << std::endl;

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
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "Dragon2 Solver v1.0.0\n";
        std::cout << "GPU-accelerated Navier-Stokes/Euler solver\n\n";

        if (argc >= 2) {
            std::string arg1 = argv[1];
            if (arg1 == "--help" || arg1 == "-h") {
                std::cout << "Usage: ./solver_ns <config.cfg>\n";
                std::cout << "       ./solver_ns --help\n";
                return 0;
            }
        }

        std::string param_file;
        if (argc > 1 && argv[1][0] != '-') {
            param_file = argv[1];
        } else {
            std::cerr << "Usage: ./solver_ns <config.cfg>\n";
            return 1;
        }

        Configuration config;
        config.read_parameters(param_file);

        switch (config.dimension) {
            case 2:
                run_simulation<2>(config);
                break;
            case 3:
                run_simulation<3>(config);
                break;
            default:
                throw std::runtime_error("Unsupported dimension: " + std::to_string(config.dimension)
                    + ". Only 2D and 3D are supported.");
        }

    } catch (std::exception& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return 1;
    }

    return 0;
}