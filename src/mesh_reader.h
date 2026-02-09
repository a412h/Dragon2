
#ifndef MESH_READER_H
#define MESH_READER_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <set>

using namespace dealii;

class Configuration;

namespace BoundaryCondition {
    enum Type {
        DO_NOTHING = 0,
        PERIODIC = 1,
        SLIP = 2,
        NO_SLIP = 3,
        DIRICHLET = 4,
        DYNAMIC = 5,
        DIRICHLET_MOMENTUM = 6
    };

    inline std::string type_to_string(Type type) {
        switch (type) {
            case DO_NOTHING: return "DO_NOTHING";
            case PERIODIC: return "PERIODIC";
            case SLIP: return "SLIP";
            case NO_SLIP: return "NO_SLIP";
            case DIRICHLET: return "DIRICHLET";
            case DYNAMIC: return "DYNAMIC";
            case DIRICHLET_MOMENTUM: return "DIRICHLET_MOMENTUM";
            default: return "UNKNOWN";
        }
    }

    inline Type string_to_type(const std::string& str) {
        std::string lower = str;
        std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

        if (lower == "do_nothing" || lower == "do nothing" || lower == "0")
            return DO_NOTHING;
        if (lower == "periodic" || lower == "1")
            return PERIODIC;
        if (lower == "slip" || lower == "2")
            return SLIP;
        if (lower == "no_slip" || lower == "no slip" || lower == "noslip" || lower == "3")
            return NO_SLIP;
        if (lower == "dirichlet" || lower == "4")
            return DIRICHLET;
        if (lower == "dynamic" || lower == "5")
            return DYNAMIC;
        if (lower == "dirichlet_momentum" || lower == "dirichlet momentum" || lower == "6")
            return DIRICHLET_MOMENTUM;

        throw std::runtime_error("Unknown boundary condition type: " + str);
    }
}

struct BoundaryMapping {
    std::map<int, BoundaryCondition::Type> physical_id_to_bc;
    BoundaryCondition::Type default_bc = BoundaryCondition::DO_NOTHING;

    void add_mapping(int physical_id, BoundaryCondition::Type bc_type) {
        physical_id_to_bc[physical_id] = bc_type;
    }

    void add_mapping(int physical_id, const std::string& bc_name) {
        physical_id_to_bc[physical_id] = BoundaryCondition::string_to_type(bc_name);
    }

    BoundaryCondition::Type get_bc_type(int physical_id) const {
        auto it = physical_id_to_bc.find(physical_id);
        if (it != physical_id_to_bc.end()) {
            return it->second;
        }
        return default_bc;
    }

    void print() const {
        std::cout << "  Boundary condition mappings:" << std::endl;
        std::cout << "    Default BC: " << BoundaryCondition::type_to_string(default_bc) << std::endl;
        for (const auto& [phys_id, bc_type] : physical_id_to_bc) {
            std::cout << "    Physical ID " << phys_id << " -> "
                      << BoundaryCondition::type_to_string(bc_type) << std::endl;
        }
    }
};

template <int dim>
class GmshMeshReader {
public:
    static void read_mesh(Triangulation<dim>& triangulation,
                          const std::string& filename,
                          const BoundaryMapping& bc_mapping,
                          int global_refinement = 0)
    {
        std::cout << "\n========================================" << std::endl;
        std::cout << "GmshMeshReader: Loading " << dim << "D mesh" << std::endl;
        std::cout << "  File: " << filename << std::endl;
        std::cout << "========================================" << std::endl;

        std::ifstream input_file(filename);
        if (!input_file.good()) {
            throw std::runtime_error("Mesh file not found: " + filename);
        }

        std::string first_line;
        std::getline(input_file, first_line);
        input_file.seekg(0);

        MshFormat format = detect_format(first_line);
        std::cout << "  Detected format: " << format_to_string(format) << std::endl;

        GridIn<dim> grid_in;
        grid_in.attach_triangulation(triangulation);

        try {
            grid_in.read_msh(input_file);
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to read mesh file: " + std::string(e.what()));
        }
        input_file.close();

        std::cout << "  Mesh loaded successfully" << std::endl;
        std::cout << "  Initial cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "  Initial vertices: " << triangulation.n_vertices() << std::endl;

        std::set<types::boundary_id> physical_ids;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < cell->n_faces(); ++f) {
                if (cell->face(f)->at_boundary()) {
                    physical_ids.insert(cell->face(f)->boundary_id());
                }
            }
        }

        std::cout << "  Physical IDs found in mesh: ";
        for (auto id : physical_ids) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        apply_boundary_conditions(triangulation, bc_mapping);

        if (global_refinement > 0) {
            std::cout << "  Applying " << global_refinement << " levels of global refinement..." << std::endl;
            triangulation.refine_global(global_refinement);
        }

        print_mesh_info(triangulation);

        bc_mapping.print();
        std::cout << "========================================\n" << std::endl;
    }

    static void read_mesh_from_config(Triangulation<dim>& triangulation,
                                       const std::string& mesh_file,
                                       const BoundaryMapping& bc_mapping,
                                       int mesh_refinement)
    {
        read_mesh(triangulation, mesh_file, bc_mapping, mesh_refinement);
    }

private:
    enum class MshFormat {
        VERSION_1,
        VERSION_2,
        VERSION_4,
        UNKNOWN
    };

    static MshFormat detect_format(const std::string& first_line) {
        if (first_line.find("$NOD") != std::string::npos) {
            return MshFormat::VERSION_1;
        }
        if (first_line.find("$MeshFormat") != std::string::npos) {
            return MshFormat::VERSION_2;
        }
        return MshFormat::UNKNOWN;
    }

    static std::string format_to_string(MshFormat format) {
        switch (format) {
            case MshFormat::VERSION_1: return "MSH version 1 (legacy)";
            case MshFormat::VERSION_2: return "MSH version 2.x";
            case MshFormat::VERSION_4: return "MSH version 4.x";
            default: return "Unknown format";
        }
    }

    static void apply_boundary_conditions(Triangulation<dim>& triangulation,
                                           const BoundaryMapping& bc_mapping)
    {
        std::cout << "  Applying boundary condition mappings..." << std::endl;

        std::map<types::boundary_id, unsigned int> original_counts;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < cell->n_faces(); ++f) {
                if (cell->face(f)->at_boundary()) {
                    original_counts[cell->face(f)->boundary_id()]++;
                }
            }
        }

        for (auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < cell->n_faces(); ++f) {
                auto face = cell->face(f);
                if (face->at_boundary()) {
                    int original_id = static_cast<int>(face->boundary_id());
                    BoundaryCondition::Type new_bc = bc_mapping.get_bc_type(original_id);
                    face->set_boundary_id(static_cast<types::boundary_id>(new_bc));
                }
            }
        }

        std::cout << "  Boundary ID remapping:" << std::endl;
        for (const auto& [orig_id, count] : original_counts) {
            BoundaryCondition::Type new_bc = bc_mapping.get_bc_type(orig_id);
            std::cout << "    Physical ID " << orig_id << " (" << count << " faces) -> "
                      << BoundaryCondition::type_to_string(new_bc)
                      << " (ID " << static_cast<int>(new_bc) << ")" << std::endl;
        }
    }

    static void print_mesh_info(const Triangulation<dim>& triangulation)
    {
        std::cout << "\n  Final mesh statistics:" << std::endl;
        std::cout << "    Active cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "    Vertices: " << triangulation.n_vertices() << std::endl;

        std::map<types::boundary_id, unsigned int> bc_counts;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f = 0; f < cell->n_faces(); ++f) {
                if (cell->face(f)->at_boundary()) {
                    bc_counts[cell->face(f)->boundary_id()]++;
                }
            }
        }

        std::cout << "    Boundary faces by type:" << std::endl;
        for (const auto& [bc_id, count] : bc_counts) {
            std::cout << "      " << BoundaryCondition::type_to_string(
                static_cast<BoundaryCondition::Type>(bc_id))
                      << " (ID " << bc_id << "): " << count << " faces" << std::endl;
        }
    }
};

namespace MeshUtils {

    template <int dim>
    void export_to_msh(const Triangulation<dim>& triangulation,
                       const std::string& filename)
    {
        std::ofstream output(filename);
        GridOut grid_out;
        grid_out.write_msh(triangulation, output);
        output.close();
        std::cout << "Mesh exported to: " << filename << std::endl;
    }

    template <int dim>
    void export_to_vtk(const Triangulation<dim>& triangulation,
                       const std::string& filename)
    {
        std::ofstream output(filename);
        GridOut grid_out;
        grid_out.write_vtk(triangulation, output);
        output.close();
        std::cout << "Mesh exported to: " << filename << std::endl;
    }

    template <int dim>
    std::pair<Point<dim>, Point<dim>> get_bounding_box(const Triangulation<dim>& triangulation)
    {
        Point<dim> min_point, max_point;

        bool first = true;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int v = 0; v < cell->n_vertices(); ++v) {
                const Point<dim>& vertex = cell->vertex(v);
                if (first) {
                    min_point = vertex;
                    max_point = vertex;
                    first = false;
                } else {
                    for (unsigned int d = 0; d < dim; ++d) {
                        min_point[d] = std::min(min_point[d], vertex[d]);
                        max_point[d] = std::max(max_point[d], vertex[d]);
                    }
                }
            }
        }

        return {min_point, max_point};
    }

    template <int dim>
    void print_bounding_box(const Triangulation<dim>& triangulation)
    {
        auto [min_pt, max_pt] = get_bounding_box(triangulation);

        std::cout << "  Bounding box:" << std::endl;
        std::cout << "    Min: (";
        for (unsigned int d = 0; d < dim; ++d) {
            std::cout << min_pt[d];
            if (d < dim - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;

        std::cout << "    Max: (";
        for (unsigned int d = 0; d < dim; ++d) {
            std::cout << max_pt[d];
            if (d < dim - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;

        std::cout << "    Size: (";
        for (unsigned int d = 0; d < dim; ++d) {
            std::cout << (max_pt[d] - min_pt[d]);
            if (d < dim - 1) std::cout << ", ";
        }
        std::cout << ")" << std::endl;
    }

    template <int dim>
    void scale_mesh(Triangulation<dim>& triangulation, double scale_factor)
    {
        GridTools::scale(scale_factor, triangulation);
        std::cout << "  Mesh scaled by factor: " << scale_factor << std::endl;
    }

    template <int dim>
    void translate_mesh(Triangulation<dim>& triangulation, const Tensor<1, dim>& offset)
    {
        GridTools::shift(offset, triangulation);
        std::cout << "  Mesh translated" << std::endl;
    }

    template <int dim>
    bool verify_mesh(const Triangulation<dim>& triangulation)
    {
        std::cout << "\n  Mesh verification:" << std::endl;

        bool valid = true;
        unsigned int n_negative = 0;
        double min_measure = std::numeric_limits<double>::max();
        double max_measure = 0;

        for (const auto& cell : triangulation.active_cell_iterators()) {
            double measure = cell->measure();
            min_measure = std::min(min_measure, measure);
            max_measure = std::max(max_measure, measure);

            if (measure <= 0) {
                n_negative++;
                valid = false;
            }
        }

        std::cout << "    Cell measure range: [" << min_measure << ", " << max_measure << "]" << std::endl;
        std::cout << "    Measure ratio (max/min): " << max_measure / min_measure << std::endl;

        if (n_negative > 0) {
            std::cout << "    WARNING: " << n_negative << " cells with non-positive measure!" << std::endl;
        }

        if (valid) {
            std::cout << "    Mesh verification PASSED" << std::endl;
        } else {
            std::cout << "    Mesh verification FAILED" << std::endl;
        }

        return valid;
    }
}

inline BoundaryMapping parse_boundary_mapping(const std::string& mapping_str,
                                               const std::string& default_bc_str = "do_nothing")
{
    BoundaryMapping mapping;
    mapping.default_bc = BoundaryCondition::string_to_type(default_bc_str);

    if (mapping_str.empty()) {
        return mapping;
    }

    std::stringstream ss(mapping_str);
    std::string entry;

    while (std::getline(ss, entry, ',')) {
        size_t start = entry.find_first_not_of(" \t");
        size_t end = entry.find_last_not_of(" \t");
        if (start == std::string::npos) continue;
        entry = entry.substr(start, end - start + 1);

        size_t colon_pos = entry.find(':');
        if (colon_pos == std::string::npos) {
            throw std::runtime_error("Invalid boundary mapping format: " + entry +
                                     " (expected 'physical_id:bc_type')");
        }

        std::string id_str = entry.substr(0, colon_pos);
        std::string type_str = entry.substr(colon_pos + 1);

        start = id_str.find_first_not_of(" \t");
        end = id_str.find_last_not_of(" \t");
        id_str = id_str.substr(start, end - start + 1);

        start = type_str.find_first_not_of(" \t");
        end = type_str.find_last_not_of(" \t");
        type_str = type_str.substr(start, end - start + 1);

        int physical_id = std::stoi(id_str);
        mapping.add_mapping(physical_id, type_str);
    }

    return mapping;
}

#endif
