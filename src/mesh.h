#ifndef MESH_H
#define MESH_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include "configuration.h"
#include <cmath>

using namespace dealii;

class MeshGenerator {
public:
    enum BoundaryId {
        DO_NOTHING = 0,
        PERIODIC = 1,
        SLIP = 2,
        NO_SLIP = 3,
        DIRICHLET = 4,
        DYNAMIC = 5,
        DIRICHLET_MOMENTUM = 6
    };
    
    static void create_cylinder_mesh(Triangulation<2>& triangulation,
                                     const Configuration& config) {
        const double cylinder_diameter = config.object_diameter;
        const double cylinder_position = config.object_position;
        const double length = config.length;
        const double height = config.height;
        
        // Create 6 pieces and merge them (object tria7 is for merging only)
        Triangulation<2> tria1, tria2, tria3, tria4, tria5, tria6, tria7;
        
        // Central piece with cylinder hole (tria1)
        GridGenerator::hyper_cube_with_cylindrical_hole(
            tria1,
            cylinder_diameter / 2.0,     // Inner radius
            cylinder_diameter,           // Outer radius  
            0.5,                         // L (unused in 2D)
            1,                           // Repetitions
            false);                      // Colorize
        
        // Bottom left piece (tria2)
        GridGenerator::subdivided_hyper_rectangle(
            tria2,
            {2, 1},                      // subdivisions
            Point<2>(-cylinder_diameter, -cylinder_diameter),
            Point<2>(cylinder_diameter, -height / 2.0));
        
        // Top left piece (tria3)
        GridGenerator::subdivided_hyper_rectangle(
            tria3,
            {2, 1},                      // subdivisions
            Point<2>(-cylinder_diameter, cylinder_diameter),
            Point<2>(cylinder_diameter, height / 2.0));
        
        // Right middle piece (tria4)
        GridGenerator::subdivided_hyper_rectangle(
            tria4,
            {6, 2},
            Point<2>(cylinder_diameter, -cylinder_diameter),
            Point<2>(length - cylinder_position, cylinder_diameter));
        
        // Right top piece (tria5)
        GridGenerator::subdivided_hyper_rectangle(
            tria5,
            {6, 1},                      // Subdivisions
            Point<2>(cylinder_diameter, cylinder_diameter),
            Point<2>(length - cylinder_position, height / 2.0));
        
        // Right bottom piece (tria6)
        GridGenerator::subdivided_hyper_rectangle(
            tria6,
            {6, 1},                      // Subdivisions
            Point<2>(cylinder_diameter, -height / 2.0),
            Point<2>(length - cylinder_position, -cylinder_diameter));
        
        // Set mesh smoothing for tria7 before merging
        tria7.set_mesh_smoothing(triangulation.get_mesh_smoothing());
        
        // Merge all 6 pieces
        GridGenerator::merge_triangulations(
            {&tria1, &tria2, &tria3, &tria4, &tria5, &tria6},
            tria7,
            1.e-12,                      // Tolerance
            true);                       // Copy_manifold_ids
        
        // Copy to main triangulation
        triangulation.copy_triangulation(tria7);
        
        // Restore polar manifold for cylinder
        triangulation.set_manifold(0, PolarManifold<2>(Point<2>()));
        
        for (auto cell : triangulation.active_cell_iterators()) {
            for (unsigned int v : cell->vertex_indices()) {
                auto &vertex = cell->vertex(v);
                if (vertex[0] <= -cylinder_diameter + 1.e-6) {
                    vertex[0] = -cylinder_position;
                }
            }
        }
        
        // Set boundary ids
        for (auto cell : triangulation.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                const auto face = cell->face(f);
                if (!face->at_boundary()) continue;
                
                const auto center = face->center();
                
                // Right boundary: do_nothing (0)
                if (center[0] > length - cylinder_position - 1.e-6) {
                    face->set_boundary_id(DO_NOTHING);
                    continue;
                }
                
                // Left boundary: dirichlet (4)
                if (center[0] < -cylinder_position + 1.e-6) {
                    face->set_boundary_id(DIRICHLET);
                    continue;
                }
                
                // Everything else (top, bottom, cylinder): slip (2)
                face->set_boundary_id(SLIP);
            }
        }
        
        // Refine globally
        triangulation.refine_global(config.mesh_refinement);
        
        // Print statistics
        std::cout << "\nMesh created:" << std::endl;
        std::cout << "  Domain: [-" << cylinder_position << ", " 
                  << length - cylinder_position << "] x [-" 
                  << height/2.0 << ", " << height/2.0 << "]" << std::endl;
        std::cout << "  Cylinder: diameter = " << cylinder_diameter << " at (0, 0)" << std::endl;
        std::cout << "  Active cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "  Vertices: " << triangulation.n_vertices() << std::endl;
        
        // Count boundaries
        std::map<types::boundary_id, unsigned int> boundary_count;
        for (auto cell : triangulation.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                if (cell->face(f)->at_boundary()) {
                    boundary_count[cell->face(f)->boundary_id()]++;
                }
            }
        }
        
        std::cout << "\nBoundary faces:" << std::endl;
        std::cout << "  Do-nothing (right): " << boundary_count[DO_NOTHING] << std::endl;
        std::cout << "  Slip (walls/cylinder): " << boundary_count[SLIP] << std::endl;
        std::cout << "  Dirichlet (left): " << boundary_count[DIRICHLET] << std::endl;
    }

    static void create_cylinder_mesh(Triangulation<3>& triangulation,
                                         const Configuration& config) {
            // Create 2D mesh first
            Triangulation<2> tria_2d;
            Configuration config_2d = config;
            create_cylinder_mesh(tria_2d, config_2d);
            
            const double height = config.height;
            
            // Extrude 2D mesh into 3D
            GridGenerator::extrude_triangulation(tria_2d, 4, height, triangulation, true);
            
            // Shift to center the mesh in z-direction
            GridTools::shift(Tensor<1, 3>{{0, 0, -height / 2.0}}, triangulation);
            
            // Restore cylindrical manifold with z-axis
            triangulation.set_manifold(0, CylindricalManifold<3>(Tensor<1, 3>{{0., 0., 1.}}, Point<3>()));
            
            // Set boundary ids for 3D
            for (auto cell : triangulation.active_cell_iterators()) {
                for (auto f : cell->face_indices()) {
                    const auto face = cell->face(f);
                    if (!face->at_boundary()) continue;
                    
                    const auto center = face->center();
                    
                    const double cylinder_position = config.object_position;
                    const double length = config.length;
                    
                    // Right boundary: do_nothing (0)
                    if (center[0] > length - cylinder_position - 1.e-6) {
                        face->set_boundary_id(DO_NOTHING);
                        continue;
                    }
                    
                    // Left boundary: dirichlet (4)
                    if (center[0] < -cylinder_position + 1.e-6) {
                        face->set_boundary_id(DIRICHLET);
                        continue;
                    }
                    
                    // Everything else: slip (2)
                    face->set_boundary_id(SLIP);
                }
            }
        }   
};

#endif