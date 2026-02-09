
#ifndef OFFLINE_DATA_H
#define OFFLINE_DATA_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/base/quadrature_lib.h>
#include <vector>
#include <array>
#include <map>
#include <set>
#include <tuple>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace dealii;

template<int dim, typename Number = double>
class OfflineData {
public:
    struct BoundaryDescription {
        unsigned int dof_index;
        std::array<Number, dim> normal;
        Number normal_mass;
        Number boundary_mass;
        types::boundary_id id;
        Point<dim> position;
    };

    struct CouplingDescription {
        unsigned int i;
        unsigned int col_idx;
        unsigned int j;

        bool operator==(const CouplingDescription& other) const {
            return i == other.i && col_idx == other.col_idx && j == other.j;
        }
    };

    using BoundaryMap = std::vector<BoundaryDescription>;
    using CouplingBoundaryPairs = std::vector<CouplingDescription>;

    std::vector<Number> lumped_mass_matrix;
    std::vector<Number> lumped_mass_matrix_inverse;
    std::vector<std::vector<Number>> mass_matrix;
    std::vector<std::vector<std::array<Number, dim>>> c_ij;
    std::vector<std::vector<unsigned int>> sparsity;

    BoundaryMap boundary_map;
    CouplingBoundaryPairs coupling_boundary_pairs;
    CouplingBoundaryPairs coupling_internal_pairs;

    Number measure_of_omega;
    std::vector<Point<dim>> node_positions;

    DoFHandler<dim> dof_handler;
    FE_Q<dim> finite_element;

    OfflineData(Triangulation<dim>& triangulation)
        : dof_handler(triangulation), finite_element(1) {

        dof_handler.distribute_dofs(finite_element);

        DoFRenumbering::Cuthill_McKee(dof_handler);

        const unsigned int n_dofs = dof_handler.n_dofs();

        DynamicSparsityPattern dsp(n_dofs, n_dofs);
        DoFTools::make_sparsity_pattern(dof_handler, dsp);

        sparsity.resize(n_dofs);
        for (unsigned int i = 0; i < n_dofs; ++i) {
            sparsity[i].clear();
            sparsity[i].push_back(i);
            for (auto it = dsp.begin(i); it != dsp.end(i); ++it) {
                if (it->column() != i) {
                    sparsity[i].push_back(it->column());
                }
            }
            std::sort(sparsity[i].begin() + 1, sparsity[i].end());
        }

        lumped_mass_matrix.resize(n_dofs, Number(0));
        lumped_mass_matrix_inverse.resize(n_dofs);
        mass_matrix.resize(n_dofs);
        c_ij.resize(n_dofs);

        for (unsigned int i = 0; i < n_dofs; ++i) {
            mass_matrix[i].resize(sparsity[i].size(), Number(0));
            c_ij[i].resize(sparsity[i].size());
            for (auto& entry : c_ij[i]) {
                entry.fill(Number(0));
            }
        }

        compute_matrices();

        for (unsigned int i = 0; i < n_dofs; ++i) {
            lumped_mass_matrix_inverse[i] = Number(1) / lumped_mass_matrix[i];
        }

        boundary_map = construct_boundary_map();
        coupling_boundary_pairs = collect_coupling_boundary_pairs();
        coupling_internal_pairs = collect_coupling_internal_pairs();

        extract_node_positions();

        std::cout << "Offline data computed:" << std::endl;
        std::cout << "  Dimension: " << dim << "D" << std::endl;
        std::cout << "  Precision: " << (sizeof(Number) == 4 ? "float" : "double") << std::endl;
        std::cout << "  DoFs: " << n_dofs << std::endl;
        std::cout << "  Boundary nodes: " << boundary_map.size() << std::endl;
        std::cout << "  Boundary coupling pairs: " << coupling_boundary_pairs.size() << std::endl;
        std::cout << "  Internal coupling pairs: " << coupling_internal_pairs.size() << std::endl;
    }

private:
    void compute_matrices() {
        QGauss<dim> quadrature(2);
        FEValues<dim> fe_values(finite_element, quadrature,
                                update_values | update_gradients |
                                update_JxW_values | update_quadrature_points);

        const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
        const unsigned int n_q_points = quadrature.size();

        std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
        measure_of_omega = Number(0);

        for (const auto& cell : dof_handler.active_cell_iterators()) {
            fe_values.reinit(cell);
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                const Number JxW = fe_values.JxW(q);
                measure_of_omega += JxW;

                for (unsigned int i = 0; i < dofs_per_cell; ++i) {
                    const unsigned int global_i = local_dof_indices[i];
                    const Number phi_i = fe_values.shape_value(i, q);

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const unsigned int global_j = local_dof_indices[j];
                        const Number phi_j = fe_values.shape_value(j, q);
                        const Number mass_contribution = phi_i * phi_j * JxW;

                        lumped_mass_matrix[global_i] += mass_contribution;

                        auto it = std::find(sparsity[global_i].begin(),
                                          sparsity[global_i].end(), global_j);
                        if (it != sparsity[global_i].end()) {
                            size_t idx = std::distance(sparsity[global_i].begin(), it);
                            mass_matrix[global_i][idx] += mass_contribution;
                        }
                    }

                    for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                        const unsigned int global_j = local_dof_indices[j];
                        const auto grad_phi_j = fe_values.shape_grad(j, q);

                        auto it = std::find(sparsity[global_i].begin(),
                                          sparsity[global_i].end(), global_j);
                        if (it != sparsity[global_i].end()) {
                            size_t idx = std::distance(sparsity[global_i].begin(), it);
                            for (int d = 0; d < dim; ++d) {
                                c_ij[global_i][idx][d] += phi_i * grad_phi_j[d] * JxW;
                            }
                        }
                    }
                }
            }
        }
    }

    BoundaryMap construct_boundary_map() {
        using BoundaryData = std::tuple<std::array<Number, dim>,
                                       Number,
                                       Number,
                                       types::boundary_id,
                                       Point<dim>>;

        std::multimap<unsigned int, BoundaryData> preliminary_map;

        QGauss<dim-1> face_quadrature(2);
        FEFaceValues<dim> fe_face_values(finite_element, face_quadrature,
                                         update_normal_vectors | update_values |
                                         update_JxW_values);

        MappingQ1<dim> mapping;
        std::vector<types::global_dof_index> local_dof_indices;
        const auto &support_points = finite_element.get_unit_support_points();

        for (const auto& cell : dof_handler.active_cell_iterators()) {
            const unsigned int dofs_per_cell = finite_element.dofs_per_cell;
            local_dof_indices.resize(dofs_per_cell);
            cell->get_dof_indices(local_dof_indices);

            for (unsigned int f : cell->face_indices()) {
                const auto face = cell->face(f);
                if (!face->at_boundary()) continue;

                const types::boundary_id id = face->boundary_id();
                fe_face_values.reinit(cell, f);

                for (unsigned int j = 0; j < dofs_per_cell; ++j) {
                    if (!finite_element.has_support_on_face(j, f))
                        continue;

                    const unsigned int dof_index = local_dof_indices[j];

                    std::array<Number, dim> normal;
                    normal.fill(Number(0));
                    Number boundary_mass = Number(0);

                    for (unsigned int q = 0; q < face_quadrature.size(); ++q) {
                        const auto& n = fe_face_values.normal_vector(q);
                        const Number JxW = fe_face_values.JxW(q);
                        const Number phi_j = fe_face_values.shape_value(j, q);

                        boundary_mass += phi_j * JxW;
                        for (int d = 0; d < dim; ++d) {
                            normal[d] += phi_j * n[d] * JxW;
                        }
                    }

                    Point<dim> position = mapping.transform_unit_to_real_cell(cell, support_points[j]);

                    preliminary_map.insert({dof_index,
                                          {normal, boundary_mass, boundary_mass, id, position}});
                }
            }
        }

        std::multimap<unsigned int, BoundaryData> filtered_map;

        for (auto entry : preliminary_map) {
            bool inserted = false;
            const auto range = filtered_map.equal_range(entry.first);

            for (auto it = range.first; it != range.second; ++it) {
                auto &[new_normal, new_normal_mass, new_boundary_mass, new_id, new_position] = entry.second;
                auto &[normal, normal_mass, boundary_mass, id, position] = it->second;

                if (id != new_id)
                    continue;

                Number normal_norm = Number(0);
                Number new_normal_norm = Number(0);
                for (int d = 0; d < dim; ++d) {
                    normal_norm += normal[d] * normal[d];
                    new_normal_norm += new_normal[d] * new_normal[d];
                }
                normal_norm = std::sqrt(normal_norm);
                new_normal_norm = std::sqrt(new_normal_norm);

                if (normal_norm > Number(1e-14) && new_normal_norm > Number(1e-14)) {
                    Number dot = Number(0);
                    for (int d = 0; d < dim; ++d) {
                        dot += normal[d] * new_normal[d];
                    }
                    dot /= (normal_norm * new_normal_norm);

                    if (dot > Number(0.5)) {
                        for (int d = 0; d < dim; ++d) {
                            normal[d] += new_normal[d];
                        }
                        boundary_mass += new_boundary_mass;
                        inserted = true;
                        break;
                    }
                }
            }

            if (!inserted) {
                filtered_map.insert(entry);
            }
        }

        BoundaryMap result;

        Point<dim> sphere_center;
        Point<dim> min_pos, max_pos;
        for (int d = 0; d < dim; ++d) {
            min_pos[d] = std::numeric_limits<Number>::max();
            max_pos[d] = std::numeric_limits<Number>::lowest();
        }
        unsigned int n_dynamic = 0;
        for (const auto& [index, data] : filtered_map) {
            auto [normal, normal_mass, boundary_mass, id, position] = data;
            if (id == 5) {
                for (int d = 0; d < dim; ++d) {
                    min_pos[d] = std::min(min_pos[d], position[d]);
                    max_pos[d] = std::max(max_pos[d], position[d]);
                }
                n_dynamic++;
            }
        }
        if (n_dynamic > 0) {
            for (int d = 0; d < dim; ++d) {
                sphere_center[d] = Number(0.5) * (min_pos[d] + max_pos[d]);
            }
            std::cout << "  Computed sphere center for DYNAMIC BC: ("
                      << sphere_center[0];
            if constexpr (dim >= 2) std::cout << ", " << sphere_center[1];
            if constexpr (dim == 3) std::cout << ", " << sphere_center[2];
            std::cout << ")" << std::endl;
        }

        for (const auto& [index, data] : filtered_map) {
            auto [normal, normal_mass, boundary_mass, id, position] = data;

            if (id == 5 && n_dynamic > 0) {
                Number radius = Number(0);
                for (int d = 0; d < dim; ++d) {
                    normal[d] = position[d] - sphere_center[d];
                    radius += normal[d] * normal[d];
                }
                radius = std::sqrt(radius) + Number(1e-14);
                for (int d = 0; d < dim; ++d) {
                    normal[d] /= radius;
                }
                normal_mass = radius;
            } else {
                Number norm_squared = Number(0);
                for (int d = 0; d < dim; ++d) {
                    norm_squared += normal[d] * normal[d];
                }
                const Number new_normal_mass = std::sqrt(norm_squared) + Number(1e-14);
                for (int d = 0; d < dim; ++d) {
                    normal[d] /= new_normal_mass;
                }
                normal_mass = new_normal_mass;
            }

            result.push_back({index, normal, normal_mass, boundary_mass, id, position});
        }

        return result;
    }

    CouplingBoundaryPairs collect_coupling_boundary_pairs() {
        std::set<unsigned int> boundary_indices;
        for (const auto& bd : boundary_map) {
            boundary_indices.insert(bd.dof_index);
        }

        CouplingBoundaryPairs result;

        for (const auto& bd : boundary_map) {
            const unsigned int i = bd.dof_index;
            const auto& sparsity_row = sparsity[i];

            if (sparsity_row.size() == 1) continue;

            for (size_t col_idx = 1; col_idx < sparsity_row.size(); ++col_idx) {
                const unsigned int j = sparsity_row[col_idx];

                if (boundary_indices.find(j) != boundary_indices.end()) {
                    result.push_back({i, static_cast<unsigned int>(col_idx), j});
                }
            }
        }

        return result;
    }

    CouplingBoundaryPairs collect_coupling_internal_pairs() {
        std::set<unsigned int> boundary_indices;
        for (const auto& bd : boundary_map)
            boundary_indices.insert(bd.dof_index);

        CouplingBoundaryPairs result;

        for (unsigned int i = 0; i < dof_handler.n_dofs(); ++i) {
            if (boundary_indices.find(i) != boundary_indices.end()) continue;

            const auto& sparsity_row = sparsity[i];

            if (sparsity_row.size() == 1) continue;

            for (size_t col_idx = 1; col_idx < sparsity_row.size(); ++col_idx) {
                const unsigned int j = sparsity_row[col_idx];

                if (j > i && boundary_indices.find(j) == boundary_indices.end()) {
                    result.push_back({i, static_cast<unsigned int>(col_idx), j});
                }
            }
        }

        return result;
    }

    void extract_node_positions() {
        node_positions.resize(dof_handler.n_dofs());

        std::map<types::global_dof_index, Point<dim>> support_points;
        MappingQ1<dim> mapping;
        DoFTools::map_dofs_to_support_points(mapping,
                                            dof_handler,
                                            support_points);

        for (const auto& [dof, point] : support_points) {
            node_positions[dof] = point;
        }
    }
};

#endif