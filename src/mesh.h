

#ifndef MESH_H
#define MESH_H

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_in.h>
#include "configuration.h"
#include <cmath>

using namespace dealii;


template <int dim, int spacedim>
class TorusSectionManifold : public Manifold<dim, spacedim>
{
public:
    TorusSectionManifold(const double cx, const double cy, const double r_arc)
        : cx_(cx), cy_(cy), r_arc_(r_arc)
    {}

    virtual Point<spacedim>
    get_intermediate_point(const Point<spacedim> &p1,
                           const Point<spacedim> &p2,
                           const double           w) const override
    {
        double theta1, phi1, theta2, phi2;
        point_to_angles(p1, theta1, phi1);
        point_to_angles(p2, theta2, phi2);

        double dtheta = theta2 - theta1;
        if (dtheta > M_PI) dtheta -= 2.0 * M_PI;
        else if (dtheta < -M_PI) dtheta += 2.0 * M_PI;

        double dphi = phi2 - phi1;
        if (dphi > M_PI) dphi -= 2.0 * M_PI;
        else if (dphi < -M_PI) dphi += 2.0 * M_PI;

        const double theta = theta1 + w * dtheta;
        const double phi = phi1 + w * dphi;

        return angles_to_point(theta, phi);
    }

    virtual Point<spacedim>
    project_to_manifold(const ArrayView<const Point<spacedim>> &,
                        const Point<spacedim>                  &candidate) const override
    {
        double theta, phi;
        point_to_angles(candidate, theta, phi);
        return angles_to_point(theta, phi);
    }

    virtual Point<spacedim>
    get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                  const ArrayView<const double>          &weights) const override
    {
        double sum_cos_theta = 0.0, sum_sin_theta = 0.0;
        double sum_cos_phi = 0.0, sum_sin_phi = 0.0;

        for (unsigned int i = 0; i < surrounding_points.size(); ++i)
        {
            double theta, phi;
            point_to_angles(surrounding_points[i], theta, phi);
            sum_cos_theta += weights[i] * std::cos(theta);
            sum_sin_theta += weights[i] * std::sin(theta);
            sum_cos_phi += weights[i] * std::cos(phi);
            sum_sin_phi += weights[i] * std::sin(phi);
        }

        const double theta = std::atan2(sum_sin_theta, sum_cos_theta);
        const double phi = std::atan2(sum_sin_phi, sum_cos_phi);

        return angles_to_point(theta, phi);
    }

    virtual std::unique_ptr<Manifold<dim, spacedim>>
    clone() const override
    {
        return std::make_unique<TorusSectionManifold<dim, spacedim>>(cx_, cy_, r_arc_);
    }

private:
    void point_to_angles(const Point<spacedim> &p, double &theta, double &phi) const
    {
        phi = std::atan2(p[2], p[1]);
        const double R = std::sqrt(p[1] * p[1] + p[2] * p[2]);
        theta = std::atan2(R - cy_, p[0] - cx_);
    }

    Point<spacedim> angles_to_point(const double theta, const double phi) const
    {
        const double x = cx_ + r_arc_ * std::cos(theta);
        const double R = cy_ + r_arc_ * std::sin(theta);
        return Point<spacedim>(x, R * std::cos(phi), R * std::sin(phi));
    }

    const double cx_, cy_, r_arc_;
};


template <int dim, int spacedim>
class ConeManifold : public Manifold<dim, spacedim>
{
public:
    ConeManifold(const double x1, const double R1,
                 const double x2, const double R2)
        : x1_(x1), R1_(R1), x2_(x2), R2_(R2)
    {}

    virtual Point<spacedim>
    get_intermediate_point(const Point<spacedim> &p1,
                           const Point<spacedim> &p2,
                           const double           w) const override
    {
        const double x = (1.0 - w) * p1[0] + w * p2[0];
        const double R_at_x = get_radius(x);

        const double phi1 = std::atan2(p1[2], p1[1]);
        const double phi2 = std::atan2(p2[2], p2[1]);

        double dphi = phi2 - phi1;
        if (dphi > M_PI) dphi -= 2.0 * M_PI;
        else if (dphi < -M_PI) dphi += 2.0 * M_PI;

        const double phi = phi1 + w * dphi;

        return Point<spacedim>(x, R_at_x * std::cos(phi), R_at_x * std::sin(phi));
    }

    virtual Point<spacedim>
    project_to_manifold(const ArrayView<const Point<spacedim>> &,
                        const Point<spacedim>                  &candidate) const override
    {
        const double x = candidate[0];
        const double R_at_x = get_radius(x);
        const double phi = std::atan2(candidate[2], candidate[1]);
        return Point<spacedim>(x, R_at_x * std::cos(phi), R_at_x * std::sin(phi));
    }

    virtual Point<spacedim>
    get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                  const ArrayView<const double>          &weights) const override
    {
        double x = 0.0;
        for (unsigned int i = 0; i < surrounding_points.size(); ++i)
            x += weights[i] * surrounding_points[i][0];

        const double R_at_x = get_radius(x);

        double sum_cos = 0.0, sum_sin = 0.0;
        for (unsigned int i = 0; i < surrounding_points.size(); ++i)
        {
            const double phi = std::atan2(surrounding_points[i][2], surrounding_points[i][1]);
            sum_cos += weights[i] * std::cos(phi);
            sum_sin += weights[i] * std::sin(phi);
        }
        const double phi = std::atan2(sum_sin, sum_cos);

        return Point<spacedim>(x, R_at_x * std::cos(phi), R_at_x * std::sin(phi));
    }

    virtual std::unique_ptr<Manifold<dim, spacedim>>
    clone() const override
    {
        return std::make_unique<ConeManifold<dim, spacedim>>(x1_, R1_, x2_, R2_);
    }

private:
    double get_radius(const double x) const
    {
        const double x_clamped = std::max(std::min(x, std::max(x1_, x2_)),
                                          std::min(x1_, x2_));
        return R1_ + (x_clamped - x1_) * (R2_ - R1_) / (x2_ - x1_);
    }

    const double x1_, R1_, x2_, R2_;
};


template <int dim, int spacedim>
class ArcManifold2D : public Manifold<dim, spacedim>
{
public:
    ArcManifold2D(const double cx, const double cy, const double r_arc)
        : cx_(cx), cy_(cy), r_arc_(r_arc)
    {}

    virtual Point<spacedim>
    get_intermediate_point(const Point<spacedim> &p1,
                           const Point<spacedim> &p2,
                           const double           w) const override
    {
        double theta1, theta2;
        point_to_angle(p1, theta1);
        point_to_angle(p2, theta2);

        double dtheta = theta2 - theta1;
        if (dtheta > M_PI) dtheta -= 2.0 * M_PI;
        else if (dtheta < -M_PI) dtheta += 2.0 * M_PI;

        const double theta = theta1 + w * dtheta;
        return angle_to_point(theta);
    }

    virtual Point<spacedim>
    project_to_manifold(const ArrayView<const Point<spacedim>> &,
                        const Point<spacedim>                  &candidate) const override
    {
        double theta;
        point_to_angle(candidate, theta);
        return angle_to_point(theta);
    }

    virtual Point<spacedim>
    get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                  const ArrayView<const double>          &weights) const override
    {
        double sum_cos = 0.0, sum_sin = 0.0;

        for (unsigned int i = 0; i < surrounding_points.size(); ++i)
        {
            double theta;
            point_to_angle(surrounding_points[i], theta);
            sum_cos += weights[i] * std::cos(theta);
            sum_sin += weights[i] * std::sin(theta);
        }

        const double theta = std::atan2(sum_sin, sum_cos);
        return angle_to_point(theta);
    }

    virtual std::unique_ptr<Manifold<dim, spacedim>>
    clone() const override
    {
        return std::make_unique<ArcManifold2D<dim, spacedim>>(cx_, cy_, r_arc_);
    }

private:
    void point_to_angle(const Point<spacedim> &p, double &theta) const
    {

        theta = std::atan2(p[1] - cy_, p[0] - cx_);
    }

    Point<spacedim> angle_to_point(const double theta) const
    {

        const double x = cx_ + r_arc_ * std::cos(theta);
        const double R = cy_ + r_arc_ * std::sin(theta);
        return Point<spacedim>(x, R);
    }

    const double cx_, cy_, r_arc_;
};


template <int dim, int spacedim = dim>
class CircularArcManifold : public Manifold<dim, spacedim>
{
public:
  CircularArcManifold(const double cx, const double cy, const double r_arc)
    : cx_(cx), cy_(cy), r_arc_(r_arc)
  {}

  virtual Point<spacedim>
  get_intermediate_point(const Point<spacedim> &p1,
                         const Point<spacedim> &p2,
                         const double           w) const override
  {
    double theta1 = std::atan2(p1[1] - cy_, p1[0] - cx_);
    double theta2 = std::atan2(p2[1] - cy_, p2[0] - cx_);

    double dtheta = theta2 - theta1;
    if (dtheta > M_PI) dtheta -= 2.0 * M_PI;
    else if (dtheta < -M_PI) dtheta += 2.0 * M_PI;

    const double theta = theta1 + w * dtheta;

    return Point<spacedim>(cx_ + r_arc_ * std::cos(theta),
                           cy_ + r_arc_ * std::sin(theta));
  }

  virtual Point<spacedim>
  project_to_manifold(const ArrayView<const Point<spacedim>> &,
                      const Point<spacedim>                  &candidate) const override
  {
    const double theta = std::atan2(candidate[1] - cy_, candidate[0] - cx_);
    return Point<spacedim>(cx_ + r_arc_ * std::cos(theta),
                           cy_ + r_arc_ * std::sin(theta));
  }

  virtual Point<spacedim>
  get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                const ArrayView<const double>          &weights) const override
  {
    double sum_cos = 0.0, sum_sin = 0.0;

    for (unsigned int i = 0; i < surrounding_points.size(); ++i)
      {
        const double theta = std::atan2(surrounding_points[i][1] - cy_,
                                        surrounding_points[i][0] - cx_);
        sum_cos += weights[i] * std::cos(theta);
        sum_sin += weights[i] * std::sin(theta);
      }

    const double theta = std::atan2(sum_sin, sum_cos);

    return Point<spacedim>(cx_ + r_arc_ * std::cos(theta),
                           cy_ + r_arc_ * std::sin(theta));
  }

  virtual std::unique_ptr<Manifold<dim, spacedim>>
  clone() const override
  {
    return std::make_unique<CircularArcManifold<dim, spacedim>>(cx_, cy_, r_arc_);
  }

private:
  const double cx_, cy_, r_arc_;
};


template <int dim, int spacedim = dim>
class LineSegmentManifold : public Manifold<dim, spacedim>
{
public:
  LineSegmentManifold(const double x1, const double y1,
                      const double x2, const double y2)
    : x1_(x1), y1_(y1), x2_(x2), y2_(y2)
  {}

  virtual Point<spacedim>
  get_intermediate_point(const Point<spacedim> &p1,
                         const Point<spacedim> &p2,
                         const double           w) const override
  {
    double t1 = project_to_line(p1);
    double t2 = project_to_line(p2);
    double t = (1.0 - w) * t1 + w * t2;
    return point_on_line(t);
  }

  virtual Point<spacedim>
  project_to_manifold(const ArrayView<const Point<spacedim>> &,
                      const Point<spacedim>                  &candidate) const override
  {
    double t = project_to_line(candidate);
    return point_on_line(t);
  }

  virtual Point<spacedim>
  get_new_point(const ArrayView<const Point<spacedim>> &surrounding_points,
                const ArrayView<const double>          &weights) const override
  {
    double t = 0.0;
    for (unsigned int i = 0; i < surrounding_points.size(); ++i)
      t += weights[i] * project_to_line(surrounding_points[i]);
    return point_on_line(t);
  }

  virtual std::unique_ptr<Manifold<dim, spacedim>>
  clone() const override
  {
    return std::make_unique<LineSegmentManifold<dim, spacedim>>(x1_, y1_, x2_, y2_);
  }

private:
  double project_to_line(const Point<spacedim> &p) const
  {
    double dx = x2_ - x1_;
    double dy = y2_ - y1_;
    double L2 = dx * dx + dy * dy;
    if (L2 < 1e-20) return 0.0;
    return ((p[0] - x1_) * dx + (p[1] - y1_) * dy) / L2;
  }

  Point<spacedim> point_on_line(double t) const
  {
    t = std::max(0.0, std::min(1.0, t));
    return Point<spacedim>(x1_ + t * (x2_ - x1_),
                           y1_ + t * (y2_ - y1_));
  }

  const double x1_, y1_, x2_, y2_;
};

class MeshGenerator {
public:
    enum BoundaryId {
        DO_NOTHING = 0,
        PERIODIC = 1,
        SLIP = 2,
        NO_SLIP = 3,
        DIRICHLET = 4,
        DYNAMIC = 5,
        DIRICHLET_MOMENTUM = 6,

        PERIODIC_BOTTOM = 11,
        PERIODIC_TOP    = 12
    };

    template<int dim_local>
    static void create_mesh_from_file(Triangulation<dim_local>& triangulation,
                                      const Configuration& config)
    {
        const std::string& filename = config.mesh_file_path;
        if (filename.empty()) {
            throw std::runtime_error(
                "mesh_file geometry requires non-empty 'file path' parameter");
        }
        std::ifstream input_file(filename);
        if (!input_file.good()) {
            throw std::runtime_error("Mesh file not found: " + filename);
        }

        std::cout << "\nLoading mesh from file: " << filename << std::endl;
        GridIn<dim_local> grid_in;
        grid_in.attach_triangulation(triangulation);
        grid_in.read_msh(input_file);
        input_file.close();

        std::map<types::boundary_id, unsigned int> bid_counts;
        for (const auto& cell : triangulation.active_cell_iterators()) {
            for (const auto f : cell->face_indices()) {
                if (!cell->face(f)->at_boundary()) continue;
                bid_counts[cell->face(f)->boundary_id()]++;
            }
        }
        std::cout << "  Boundary face IDs found:";
        for (const auto& kv : bid_counts) {
            std::cout << " " << static_cast<int>(kv.first) << "(" << kv.second << ")";
        }
        std::cout << std::endl;

        if (config.mesh_refinement > 0) {
            triangulation.refine_global(config.mesh_refinement);
        }
        std::cout << "  Total cells after refinement: "
                  << triangulation.n_active_cells() << std::endl;
    }

    template<int dim_local>
    static void create_rectangle_mesh(Triangulation<dim_local>& triangulation,
                                      const Configuration& config)
    {
        static_assert(dim_local == 2, "Rectangle mesh only supports 2D");

        const double x_left = config.rect_x_left;
        const double x_right = config.rect_x_right;
        const double y_bottom = config.rect_y_bottom;
        const double y_top = config.rect_y_top;
        const unsigned int nx = config.rect_subdivisions_x;
        const unsigned int ny = config.rect_subdivisions_y;

        Point<dim_local> p1(x_left, y_bottom);
        Point<dim_local> p2(x_right, y_top);

        std::vector<unsigned int> subdivisions = {nx, ny};
        GridGenerator::subdivided_hyper_rectangle(
            triangulation, subdivisions, p1, p2, true);

        const bool periodic_y = config.rect_periodic_y;
        const types::boundary_id PERIODIC_BOTTOM = 11;
        const types::boundary_id PERIODIC_TOP    = 12;
        for (auto& cell : triangulation.active_cell_iterators()) {
            for (unsigned int f : cell->face_indices()) {
                if (!cell->face(f)->at_boundary()) continue;
                const types::boundary_id old_id = cell->face(f)->boundary_id();

                if (old_id == 0 || old_id == 1) {
                    cell->face(f)->set_boundary_id(DIRICHLET);
                } else if (old_id == 2) {
                    cell->face(f)->set_boundary_id(periodic_y ? PERIODIC_BOTTOM : SLIP);
                } else { 
                    cell->face(f)->set_boundary_id(periodic_y ? PERIODIC_TOP : SLIP);
                }
            }
        }

        if (periodic_y) {
            std::vector<GridTools::PeriodicFacePair<
                typename Triangulation<dim_local>::cell_iterator>> periodic_pairs;
            GridTools::collect_periodic_faces(
                triangulation,
                PERIODIC_BOTTOM,
                PERIODIC_TOP,
                1,
                periodic_pairs);
            triangulation.add_periodicity(periodic_pairs);
            std::cout << "  Periodic y-faces registered: " << periodic_pairs.size() << " pairs" << std::endl;
        }

        triangulation.refine_global(config.mesh_refinement);

        std::cout << "Rectangle mesh created: [" << x_left << ", " << x_right
                  << "] x [" << y_bottom << ", " << y_top << "]" << std::endl;
        std::cout << "  Base subdivisions: " << nx << " x " << ny << std::endl;
        std::cout << "  Refinement levels: " << config.mesh_refinement << std::endl;
        std::cout << "  Total cells: " << triangulation.n_active_cells() << std::endl;
    }

    static void create_cylinder_mesh(Triangulation<2>& triangulation,
                                     const Configuration& config)
    {
        const double cylinder_diameter = config.object_diameter;
        const double cylinder_position = config.object_position;
        const double length = config.length;
        const double height = config.height;

        Triangulation<2> tria1, tria2, tria3, tria4, tria5, tria6, tria7;

        GridGenerator::hyper_cube_with_cylindrical_hole(
            tria1,
            cylinder_diameter / 2.0,     
            cylinder_diameter,           
            0.5,                         
            1,                           
            false);                      

        GridGenerator::subdivided_hyper_rectangle(
            tria2,
            {2, 1},                      
            Point<2>(-cylinder_diameter, -cylinder_diameter),
            Point<2>(cylinder_diameter, -height / 2.0));

        GridGenerator::subdivided_hyper_rectangle(
            tria3,
            {2, 1},                      
            Point<2>(-cylinder_diameter, cylinder_diameter),
            Point<2>(cylinder_diameter, height / 2.0));

        GridGenerator::subdivided_hyper_rectangle(
            tria4,
            {6, 2},
            Point<2>(cylinder_diameter, -cylinder_diameter),
            Point<2>(length - cylinder_position, cylinder_diameter));

        GridGenerator::subdivided_hyper_rectangle(
            tria5,
            {6, 1},                      
            Point<2>(cylinder_diameter, cylinder_diameter),
            Point<2>(length - cylinder_position, height / 2.0));

        GridGenerator::subdivided_hyper_rectangle(
            tria6,
            {6, 1},                      
            Point<2>(cylinder_diameter, -height / 2.0),
            Point<2>(length - cylinder_position, -cylinder_diameter));

        tria7.set_mesh_smoothing(triangulation.get_mesh_smoothing());

        GridGenerator::merge_triangulations(
            {&tria1, &tria2, &tria3, &tria4, &tria5, &tria6},
            tria7,
            1.e-12,                      
            true);                       

        triangulation.copy_triangulation(tria7);

        triangulation.set_manifold(0, PolarManifold<2>(Point<2>()));
        
        for (auto cell : triangulation.active_cell_iterators()) {
            for (unsigned int v : cell->vertex_indices()) {
                auto &vertex = cell->vertex(v);
                if (vertex[0] <= -cylinder_diameter + 1.e-6) {
                    vertex[0] = -cylinder_position;
                }
            }
        }

        for (auto cell : triangulation.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                const auto face = cell->face(f);
                if (!face->at_boundary()) continue;
                
                const auto center = face->center();

                if (center[0] > length - cylinder_position - 1.e-6) {
                    face->set_boundary_id(DO_NOTHING);
                    continue;
                }

                if (center[0] < -cylinder_position + 1.e-6) {
                    face->set_boundary_id(DIRICHLET);
                    continue;
                }

                if (center.norm() < cylinder_diameter) {
                    face->set_boundary_id(NO_SLIP);
                    continue;
                }

                face->set_boundary_id(SLIP);
            }
        }

        triangulation.refine_global(config.mesh_refinement);

        std::cout << "\nMesh created:" << std::endl;
        std::cout << "  Domain: [-" << cylinder_position << ", " 
                  << length - cylinder_position << "] x [-" 
                  << height/2.0 << ", " << height/2.0 << "]" << std::endl;
        std::cout << "  Cylinder: diameter = " << cylinder_diameter << " at (0, 0)" << std::endl;
        std::cout << "  Active cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "  Vertices: " << triangulation.n_vertices() << std::endl;

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
        std::cout << "  Slip (top/bottom walls): " << boundary_count[SLIP] << std::endl;
        std::cout << "  No-slip (cylinder): " << boundary_count[NO_SLIP] << std::endl;
        std::cout << "  Dirichlet (left): " << boundary_count[DIRICHLET] << std::endl;
    }

    static void create_cylinder_mesh(Triangulation<3>& triangulation,
                                         const Configuration& config)
    {

        Triangulation<2> tria_2d;
        Configuration config_2d = config;
        create_cylinder_mesh(tria_2d, config_2d);
        
        const double height = config.height;

        GridGenerator::extrude_triangulation(tria_2d, 4, height, triangulation, true);

        GridTools::shift(Tensor<1, 3>{{0, 0, -height / 2.0}}, triangulation);

        triangulation.set_manifold(0, CylindricalManifold<3>(Tensor<1, 3>{{0., 0., 1.}}, Point<3>()));

        for (auto cell : triangulation.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                const auto face = cell->face(f);
                if (!face->at_boundary()) continue;
                
                const auto center = face->center();
                
                const double cylinder_position = config.object_position;
                const double length = config.length;

                if (center[0] > length - cylinder_position - 1.e-6) {
                    face->set_boundary_id(DO_NOTHING);
                    continue;
                }

                if (center[0] < -cylinder_position + 1.e-6) {
                    face->set_boundary_id(DIRICHLET);
                    continue;
                }

                const double r_xy = std::hypot(center[0], center[1]);
                const bool is_z_cap = std::abs(center[2]) > height / 2.0 - 1.e-6;
                if (r_xy < config.object_diameter && !is_z_cap) {
                    face->set_boundary_id(NO_SLIP);
                    continue;
                }

                face->set_boundary_id(SLIP);
            }
        }
    }

    static void create_sphere_in_channel_mesh(Triangulation<3>& triangulation, const Configuration& config)
    {
        const double length_before_sphere = config.length_before_sphere;
        const double length_after_sphere = config.length_after_sphere;
        const double height_below_sphere = config.height_below_sphere;
        const double height_above_sphere = config.height_above_sphere;
        const double depth = config.depth;

        const unsigned int rep_before_sphere = config.rep_before_sphere;
        const unsigned int rep_after_sphere = config.rep_after_sphere;
        const unsigned int rep_below_sphere = config.rep_below_sphere;
        const unsigned int rep_above_sphere = config.rep_above_sphere;
        const unsigned int rep_depth = config.rep_depth;

        const double inner_radius = config.inner_radius;
        const double outer_radius = config.outer_radius;
        const unsigned int n_cells = config.n_cells;

        const int mesh_refinement = config.mesh_refinement;
        std::cout<<"mesh_refinement: " << mesh_refinement << std::endl;  

        const std::vector<double> lengths_heights_widths = {length_before_sphere, length_after_sphere, height_below_sphere, height_above_sphere, depth, depth}; 
        const std::vector<unsigned int> lengths_heights_widths_repetitions = {rep_before_sphere, rep_after_sphere, rep_below_sphere, rep_above_sphere, rep_depth, rep_depth}; 

        dealii::GridGenerator::uniform_channel_with_sphere(
            triangulation,
            lengths_heights_widths,
            inner_radius,
            outer_radius,
            true,
            true);  

        if (mesh_refinement > 0)
            triangulation.refine_global(mesh_refinement);

        for (auto &face : triangulation.active_face_iterators())
        {
            if (!face->at_boundary())
                continue;

            const auto bid = face->boundary_id();
            
            if (bid == 0)  
                face->set_boundary_id(DIRICHLET);
            else if (bid == 1)  
                face->set_boundary_id(DO_NOTHING);
            else if (bid == 2)  
                face->set_boundary_id(SLIP);
            else if (bid >= 3 && bid <= 6)  
                face->set_boundary_id(DO_NOTHING);
        }    
    }

    template <int dim>
    static void create_channel_with_cylinder_mesh(Triangulation<dim>& triangulation,
                                                  const Configuration& config)
    {
        const std::vector<unsigned int> lengths_and_heights = {
            config.length_before_cylinder,
            config.length_after_cylinder,
            config.height_below_cylinder,
            config.height_above_cylinder
        };

        GridGenerator::uniform_channel_with_cylinder(
            triangulation,
            lengths_and_heights,
            1.0,    
            1,      
            config.cwc_shell_region_radius,
            config.cwc_n_shells,
            config.cwc_skewness,
            config.cwc_use_transfinite_region,
            true);  

        if (config.mesh_refinement > 0)
            triangulation.refine_global(config.mesh_refinement);

        for (auto &face : triangulation.active_face_iterators())
        {
            if (!face->at_boundary())
                continue;

            const auto bid = face->boundary_id();

            if (bid == 0)       
                face->set_boundary_id(DIRICHLET);
            else if (bid == 1)  
                face->set_boundary_id(DO_NOTHING);
            else if (bid == 2)  
                face->set_boundary_id(SLIP);
            else if (bid == 3 || bid == 4)  
                face->set_boundary_id(SLIP);
        }

        std::map<types::boundary_id, unsigned int> boundary_count;
        for (const auto &cell : triangulation.active_cell_iterators())
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                if (cell->face(f)->at_boundary())
                    boundary_count[cell->face(f)->boundary_id()]++;

        std::cout << "\n  Channel with cylinder boundary faces:" << std::endl;
        std::cout << "  DO_NOTHING (outflow): " << boundary_count[DO_NOTHING] << std::endl;
        std::cout << "  Slip (walls/cylinder): " << boundary_count[SLIP] << std::endl;
        std::cout << "  Dirichlet (inflow): " << boundary_count[DIRICHLET] << std::endl;
    }

    static void create_airfoil_mesh(Triangulation<2>& triangulation, const Configuration& config)
    {

        std::cout << "\nLoading pre-generated OAT15a airfoil mesh..." << std::endl;
        
        const std::string filename = "../Meshes/oat15a_mesh.msh";
        
        std::ifstream input_file(filename);
        if (!input_file.good()) {
            std::cerr << "\n===============================================" << std::endl;
            std::cerr << "ERROR: Airfoil mesh file not found!" << std::endl;
            std::cerr << "===============================================" << std::endl;
            std::cerr << "Looking for: " << filename << std::endl;
            std::cerr << "===============================================\n" << std::endl;
            throw std::runtime_error("Airfoil mesh file not found");
        }
        
        GridIn<2> grid_in;
        grid_in.attach_triangulation(triangulation);
        grid_in.read_msh(input_file);
        input_file.close();
        
        std::cout << "Successfully loaded: " << filename << std::endl;

        for (auto cell : triangulation.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                if (!cell->face(f)->at_boundary()) continue;
                
                auto face = cell->face(f);
                const auto b_id = face->boundary_id();
                
                if (b_id == 3) {
                    face->set_boundary_id(NO_SLIP);  
                }
                else if (b_id == 5) {
                    face->set_boundary_id(DYNAMIC);  
                }
            }
        }
        
        print_mesh_info(triangulation, "Airfoil (OAT15a)");
    }

    static void create_capsule_mesh(Triangulation<2>& triangulation, const Configuration& config)
    {
        std::cout << "\nGenerating 2D NASA capsule mesh (O-grid topology)..." << std::endl;

        const double r1 = 1.98;      
        const double r2 = 9.9;       
        const double r3 = 275.62;    

        const Point<2> O(0.0, 0.0);
        const Point<2> A1(0.0, 40.53);
        const Point<2> B1(0.99, 42.24);
        const Point<2> A2(99.29, 99.00);
        const Point<2> B2(113.55, 93.80);
        const Point<2> Op(130.0, 0.0);

        const Point<2> A1p(0.0, -40.53);
        const Point<2> B1p(0.99, -42.24);
        const Point<2> A2p(99.29, -99.00);
        const Point<2> B2p(113.55, -93.80);

        const Point<2> O1(1.98, 40.53);
        const Point<2> O1p(1.98, -40.53);
        const Point<2> O2(104.24, 90.43);
        const Point<2> O2p(104.24, -90.43);
        const Point<2> O3(-145.62, 0.0);

        const Point<2> center(65.0, 0.0);
        const double capsule_diameter = 2.0 * 99.00;
        const double R_mid = 250.0;
        const double R_outer = (capsule_diameter / 2.0) * config.sphere_diameter_multiplier;

        const unsigned int n_inner = 6;
        const unsigned int n_outer = 4;
        const unsigned int n_total = n_inner + n_outer;

        const unsigned int n_segment_OA1 = 4;
        const unsigned int n_arc_A1B1 = 2;
        const unsigned int n_segment_B1A2 = 8;
        const unsigned int n_arc_A2B2 = 2;
        const unsigned int n_arc_B2Op = 6;

        std::cout << "  Capsule diameter: " << capsule_diameter << std::endl;
        std::cout << "  Inner zone: capsule to R_mid = " << R_mid << std::endl;
        std::cout << "  Outer zone: R_mid to R_outer = " << R_outer << std::endl;
        std::cout << "  Inner layers: " << n_inner << std::endl;
        std::cout << "  Outer layers: " << n_outer << std::endl;

        std::vector<Point<2>> capsule_boundary;
        std::vector<unsigned int> boundary_manifold_ids;

        auto normalize_angle_diff = [](double dangle) -> double {
            while (dangle > M_PI) dangle -= 2.0 * M_PI;
            while (dangle < -M_PI) dangle += 2.0 * M_PI;
            return dangle;
        };

        for (unsigned int i = 0; i <= n_segment_OA1; ++i) {
            double t = static_cast<double>(i) / n_segment_OA1;
            capsule_boundary.push_back(Point<2>((1-t)*O[0] + t*A1[0],
                                                (1-t)*O[1] + t*A1[1]));
            if (i < n_segment_OA1)
                boundary_manifold_ids.push_back(10);
        }

        double angle_A1 = std::atan2(A1[1] - O1[1], A1[0] - O1[0]);
        double angle_B1 = std::atan2(B1[1] - O1[1], B1[0] - O1[0]);
        double dangle_A1B1 = normalize_angle_diff(angle_B1 - angle_A1);
        for (unsigned int i = 1; i <= n_arc_A1B1; ++i) {
            double t = static_cast<double>(i) / n_arc_A1B1;
            double angle = angle_A1 + t * dangle_A1B1;
            capsule_boundary.push_back(Point<2>(O1[0] + r1 * std::cos(angle),
                                                O1[1] + r1 * std::sin(angle)));
            boundary_manifold_ids.push_back(11);
        }

        for (unsigned int i = 1; i <= n_segment_B1A2; ++i) {
            double t = static_cast<double>(i) / n_segment_B1A2;
            capsule_boundary.push_back(Point<2>((1-t)*B1[0] + t*A2[0],
                                                (1-t)*B1[1] + t*A2[1]));
            boundary_manifold_ids.push_back(12);
        }

        double angle_A2 = std::atan2(A2[1] - O2[1], A2[0] - O2[0]);
        double angle_B2 = std::atan2(B2[1] - O2[1], B2[0] - O2[0]);
        double dangle_A2B2 = normalize_angle_diff(angle_B2 - angle_A2);
        for (unsigned int i = 1; i <= n_arc_A2B2; ++i) {
            double t = static_cast<double>(i) / n_arc_A2B2;
            double angle = angle_A2 + t * dangle_A2B2;
            capsule_boundary.push_back(Point<2>(O2[0] + r2 * std::cos(angle),
                                                O2[1] + r2 * std::sin(angle)));
            boundary_manifold_ids.push_back(13);
        }

        double angle_B2_shield = std::atan2(B2[1] - O3[1], B2[0] - O3[0]);
        double angle_Op = std::atan2(Op[1] - O3[1], Op[0] - O3[0]);
        double dangle_B2Op = normalize_angle_diff(angle_Op - angle_B2_shield);
        for (unsigned int i = 1; i <= n_arc_B2Op; ++i) {
            double t = static_cast<double>(i) / n_arc_B2Op;
            double angle = angle_B2_shield + t * dangle_B2Op;
            capsule_boundary.push_back(Point<2>(O3[0] + r3 * std::cos(angle),
                                                O3[1] + r3 * std::sin(angle)));
            boundary_manifold_ids.push_back(14);
        }

        double angle_B2p_shield = std::atan2(B2p[1] - O3[1], B2p[0] - O3[0]);
        double dangle_OpB2p = normalize_angle_diff(angle_B2p_shield - angle_Op);
        for (unsigned int i = 1; i <= n_arc_B2Op; ++i) {
            double t = static_cast<double>(i) / n_arc_B2Op;
            double angle = angle_Op + t * dangle_OpB2p;
            capsule_boundary.push_back(Point<2>(O3[0] + r3 * std::cos(angle),
                                                O3[1] + r3 * std::sin(angle)));
            boundary_manifold_ids.push_back(24);
        }

        double angle_B2p = std::atan2(B2p[1] - O2p[1], B2p[0] - O2p[0]);
        double angle_A2p = std::atan2(A2p[1] - O2p[1], A2p[0] - O2p[0]);
        double dangle_B2pA2p = normalize_angle_diff(angle_A2p - angle_B2p);
        for (unsigned int i = 1; i <= n_arc_A2B2; ++i) {
            double t = static_cast<double>(i) / n_arc_A2B2;
            double angle = angle_B2p + t * dangle_B2pA2p;
            capsule_boundary.push_back(Point<2>(O2p[0] + r2 * std::cos(angle),
                                                O2p[1] + r2 * std::sin(angle)));
            boundary_manifold_ids.push_back(23);
        }

        for (unsigned int i = 1; i <= n_segment_B1A2; ++i) {
            double t = static_cast<double>(i) / n_segment_B1A2;
            capsule_boundary.push_back(Point<2>((1-t)*A2p[0] + t*B1p[0],
                                                (1-t)*A2p[1] + t*B1p[1]));
            boundary_manifold_ids.push_back(22);
        }

        double angle_B1p = std::atan2(B1p[1] - O1p[1], B1p[0] - O1p[0]);
        double angle_A1p = std::atan2(A1p[1] - O1p[1], A1p[0] - O1p[0]);
        double dangle_B1pA1p = normalize_angle_diff(angle_A1p - angle_B1p);
        for (unsigned int i = 1; i <= n_arc_A1B1; ++i) {
            double t = static_cast<double>(i) / n_arc_A1B1;
            double angle = angle_B1p + t * dangle_B1pA1p;
            capsule_boundary.push_back(Point<2>(O1p[0] + r1 * std::cos(angle),
                                                O1p[1] + r1 * std::sin(angle)));
            boundary_manifold_ids.push_back(21);
        }

        for (unsigned int i = 1; i < n_segment_OA1; ++i) {
            double t = static_cast<double>(i) / n_segment_OA1;
            capsule_boundary.push_back(Point<2>((1-t)*A1p[0] + t*O[0],
                                                (1-t)*A1p[1] + t*O[1]));
            boundary_manifold_ids.push_back(20);
        }

        const unsigned int n_boundary = capsule_boundary.size();
        std::cout << "  Capsule boundary points: " << n_boundary << std::endl;

        std::vector<Point<2>> all_vertices;
        std::vector<CellData<2>> all_cells;

        auto project_to_circle = [&](const Point<2> &p, double R) -> Point<2> {
            Tensor<1, 2> dir = p - center;
            double norm = dir.norm();
            if (norm < 1e-10) return center + Point<2>(R, 0);
            return center + (R / norm) * dir;
        };

        auto interp_inner = [&](const Point<2> &p_body, double t) -> Point<2> {
            Tensor<1, 2> dir = p_body - center;
            double r_body = dir.norm();
            if (r_body < 1e-10) return p_body + t * (project_to_circle(p_body, R_mid) - p_body);
            double r = r_body * std::pow(R_mid / r_body, t);
            return center + (r / r_body) * dir;
        };

        auto interp_outer = [&](const Point<2> &p_mid, double t) -> Point<2> {
            Tensor<1, 2> dir = p_mid - center;
            double r = R_mid + t * (R_outer - R_mid);
            return center + (r / R_mid) * dir;
        };

        auto get_radial_point = [&](const Point<2> &p_body, unsigned int k) -> Point<2> {
            if (k <= n_inner) {
                double t = static_cast<double>(k) / n_inner;
                return interp_inner(p_body, t);
            } else {
                Point<2> p_mid = project_to_circle(p_body, R_mid);
                double t = static_cast<double>(k - n_inner) / n_outer;
                return interp_outer(p_mid, t);
            }
        };

        std::vector<std::vector<unsigned int>> vertex_indices(n_total + 1,
                                                               std::vector<unsigned int>(n_boundary));

        for (unsigned int k = 0; k <= n_total; ++k)
            for (unsigned int j = 0; j < n_boundary; ++j) {
                Point<2> p = get_radial_point(capsule_boundary[j], k);
                all_vertices.push_back(p);
                vertex_indices[k][j] = all_vertices.size() - 1;
            }

        for (unsigned int k = 0; k < n_total; ++k)
            for (unsigned int j = 0; j < n_boundary; ++j) {
                unsigned int jn = (j + 1) % n_boundary;
                CellData<2> cell;
                cell.vertices[0] = vertex_indices[k][j];
                cell.vertices[1] = vertex_indices[k][jn];
                cell.vertices[2] = vertex_indices[k+1][j];
                cell.vertices[3] = vertex_indices[k+1][jn];
                cell.material_id = (k < n_inner) ? 0 : 1;
                all_cells.push_back(cell);
            }

        std::cout << "  Total vertices: " << all_vertices.size() << std::endl;
        std::cout << "  Total cells: " << all_cells.size() << std::endl;

        GridTools::consistently_order_cells(all_cells);
        triangulation.create_triangulation(all_vertices, all_cells, SubCellData());

        for (auto &cell : triangulation.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary()) {
                    Point<2> c = face->center();
                    double r = (c - center).norm();
                    if (r > 0.9 * R_outer)
                        face->set_boundary_id(DYNAMIC);
                    else
                        face->set_boundary_id(NO_SLIP);
                }

        SphericalManifold<2> outer_manifold(center);
        triangulation.set_manifold(1, outer_manifold);

        CircularArcManifold<2> fillet_upper_manifold(O1[0], O1[1], r1);
        CircularArcManifold<2> fillet_lower_manifold(O1p[0], O1p[1], r1);
        CircularArcManifold<2> shoulder_upper_manifold(O2[0], O2[1], r2);
        CircularArcManifold<2> shoulder_lower_manifold(O2p[0], O2p[1], r2);
        CircularArcManifold<2> shield_manifold(O3[0], O3[1], r3);
        LineSegmentManifold<2> nose_upper_manifold(O[0], O[1], A1[0], A1[1]);
        LineSegmentManifold<2> nose_lower_manifold(O[0], O[1], A1p[0], A1p[1]);
        LineSegmentManifold<2> cone_upper_manifold(B1[0], B1[1], A2[0], A2[1]);
        LineSegmentManifold<2> cone_lower_manifold(B1p[0], B1p[1], A2p[0], A2p[1]);

        triangulation.set_manifold(10, nose_upper_manifold);
        triangulation.set_manifold(11, fillet_upper_manifold);
        triangulation.set_manifold(12, cone_upper_manifold);
        triangulation.set_manifold(13, shoulder_upper_manifold);
        triangulation.set_manifold(14, shield_manifold);
        triangulation.set_manifold(20, nose_lower_manifold);
        triangulation.set_manifold(21, fillet_lower_manifold);
        triangulation.set_manifold(22, cone_lower_manifold);
        triangulation.set_manifold(23, shoulder_lower_manifold);
        triangulation.set_manifold(24, shield_manifold);

        for (auto &cell : triangulation.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary()) {
                    Point<2> c = face->center();
                    double r = (c - center).norm();

                    if (r > 0.9 * R_outer) {
                        face->set_all_manifold_ids(1);
                    } else {

                        unsigned int v0 = face->vertex_index(0);
                        unsigned int v1 = face->vertex_index(1);

                        int boundary_idx = -1;
                        for (unsigned int j = 0; j < n_boundary; ++j) {
                            unsigned int jn = (j + 1) % n_boundary;
                            if ((v0 == vertex_indices[0][j] && v1 == vertex_indices[0][jn]) ||
                                (v1 == vertex_indices[0][j] && v0 == vertex_indices[0][jn])) {
                                boundary_idx = j;
                                break;
                            }
                        }

                        if (boundary_idx >= 0 && boundary_idx < static_cast<int>(boundary_manifold_ids.size()))
                            face->set_all_manifold_ids(boundary_manifold_ids[boundary_idx]);
                    }
                }

        if (config.mesh_refinement > 0)
            triangulation.refine_global(config.mesh_refinement);

        print_mesh_info(triangulation, "2D NASA Capsule (O-grid)");
    }

    static void create_capsule_mesh(Triangulation<3>& triangulation, const Configuration& config)
    {
        std::cout << "\nGenerating NASA capsule mesh..." << std::endl;

        const double cx_shield = -145.62;
        const double r3 = 275.62;

        const double cx_shoulder = 104.24;
        const double cy_shoulder = 90.43;
        const double r2 = 9.9;
        const double angle_B2 = std::atan2(93.80 - cy_shoulder, 113.55 - cx_shoulder);
        const double angle_A2 = std::atan2(99.00 - cy_shoulder, 99.29 - cx_shoulder);

        const double x_A2 = 99.29, R_A2 = 99.00;
        const double x_B1 = 0.99,  R_B1 = 42.24;

        const double cx_fillet = 1.98;
        const double cy_fillet = 40.53;
        const double r1_fillet = 1.98;
        const double angle_B1 = std::atan2(42.24 - cy_fillet, 0.99 - cx_fillet);
        const double angle_A1 = std::atan2(40.53 - cy_fillet, 0.0 - cx_fillet);

        const double R_A1 = 40.53;
        const double theta_shield = std::atan2(93.80, 113.55 - cx_shield);

        const double capsule_diameter = 2.0 * 99.00;  

        const Point<3> center(65.0, 0.0, 0.0);
        const double R_mid = 250.0;     
        const double R_outer = (capsule_diameter / 2.0) * config.sphere_diameter_multiplier;

        const unsigned int n_inner = 6;   
        const unsigned int n_outer = 4;   

        std::cout << "  Capsule diameter: " << capsule_diameter << std::endl;
        std::cout << "  Sphere diameter multiplier: " << config.sphere_diameter_multiplier << std::endl;
        std::cout << "  Inner zone: capsule to R_mid = " << R_mid << std::endl;
        std::cout << "  Outer zone: R_mid to R_outer = " << R_outer << std::endl;
        std::cout << "  n_inner = " << n_inner << ", n_outer = " << n_outer << std::endl;

        Triangulation<2> disk;
        GridGenerator::hyper_ball_balanced(disk, Point<2>(), 1.0);
        disk.refine_global(2);

        std::vector<std::pair<double, unsigned int>> boundary_angles;
        for (const auto &cell : disk.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary())
                    for (unsigned int v = 0; v < 2; ++v)
                    {
                        unsigned int vi = face->vertex_index(v);
                        Point<2> p = face->vertex(v);
                        double angle = std::atan2(p[1], p[0]);
                        bool found = false;
                        for (const auto &ba : boundary_angles)
                            if (ba.second == vi) { found = true; break; }
                        if (!found)
                            boundary_angles.push_back({angle, vi});
                    }
        std::sort(boundary_angles.begin(), boundary_angles.end());
        const unsigned int n_phi = boundary_angles.size();

        std::cout << "  n_phi = " << n_phi << std::endl;

        auto to_sphere = [&](const Point<3> &p, double R) -> Point<3> {
            Tensor<1, 3> dir = p - center;
            double norm = dir.norm();
            if (norm < 1e-10) return center + Point<3>(R, 0, 0);
            return center + (R / norm) * dir;
        };

        auto interp_inner = [&](const Point<3> &p_body, double t) -> Point<3> {
            Tensor<1, 3> dir = p_body - center;
            double r_body = dir.norm();
            if (r_body < 1e-10) return p_body + t * (to_sphere(p_body, R_mid) - p_body);
            double r = r_body * std::pow(R_mid / r_body, t);
            return center + (r / r_body) * dir;
        };

        auto interp_outer = [&](const Point<3> &p_mid, double t) -> Point<3> {
            Tensor<1, 3> dir = p_mid - center;
            double r = R_mid + t * (R_outer - R_mid);
            return center + (r / R_mid) * dir;
        };

        std::vector<Point<3>> all_vertices;
        std::vector<CellData<3>> all_cells;

        const unsigned int n_shoulder = 1;
        const unsigned int n_cone = 8;
        const unsigned int n_fillet = 1;
        const unsigned int n_total_radial = n_inner + n_outer;

        std::vector<std::pair<double, double>> shoulder_xR(n_shoulder + 1);
        for (unsigned int i = 0; i <= n_shoulder; ++i) {
            double t = static_cast<double>(i) / n_shoulder;
            double arc = angle_B2 + t * (angle_A2 - angle_B2);
            shoulder_xR[i] = {cx_shoulder + r2 * std::cos(arc),
                              cy_shoulder + r2 * std::sin(arc)};
        }

        std::vector<std::pair<double, double>> cone_xR(n_cone + 1);
        for (unsigned int i = 0; i <= n_cone; ++i) {
            double t = static_cast<double>(i) / n_cone;
            cone_xR[i] = {x_A2 + t * (x_B1 - x_A2), R_A2 + t * (R_B1 - R_A2)};
        }

        std::vector<std::pair<double, double>> fillet_xR(n_fillet + 1);
        for (unsigned int i = 0; i <= n_fillet; ++i) {
            double t = static_cast<double>(i) / n_fillet;
            double arc = angle_B1 + t * (angle_A1 - angle_B1);
            fillet_xR[i] = {cx_fillet + r1_fillet * std::cos(arc),
                            cy_fillet + r1_fillet * std::sin(arc)};
        }

        auto get_radial_point = [&](const Point<3> &p_body, unsigned int k) -> Point<3> {
            if (k <= n_inner) {
                double t = static_cast<double>(k) / n_inner;
                return interp_inner(p_body, t);
            } else {
                Point<3> p_mid = to_sphere(p_body, R_mid);
                double t = static_cast<double>(k - n_inner) / n_outer;
                return interp_outer(p_mid, t);
            }
        };

        std::cout << "  Building shoulder..." << std::endl;
        std::vector<std::vector<std::vector<unsigned int>>> shoulder_vi(
            n_total_radial + 1, std::vector<std::vector<unsigned int>>(
                n_shoulder + 1, std::vector<unsigned int>(n_phi)));

        for (unsigned int k = 0; k <= n_total_radial; ++k)
            for (unsigned int i = 0; i <= n_shoulder; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    double phi = boundary_angles[j].first;
                    Point<3> p_body(shoulder_xR[i].first,
                                    shoulder_xR[i].second * std::cos(phi),
                                    shoulder_xR[i].second * std::sin(phi));
                    all_vertices.push_back(get_radial_point(p_body, k));
                    shoulder_vi[k][i][j] = all_vertices.size() - 1;
                }

        for (unsigned int k = 0; k < n_total_radial; ++k)
            for (unsigned int i = 0; i < n_shoulder; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    unsigned int jn = (j + 1) % n_phi;
                    CellData<3> cell;

                    cell.vertices[0] = shoulder_vi[k+1][i][j];
                    cell.vertices[1] = shoulder_vi[k+1][i][jn];
                    cell.vertices[2] = shoulder_vi[k+1][i+1][j];
                    cell.vertices[3] = shoulder_vi[k+1][i+1][jn];
                    cell.vertices[4] = shoulder_vi[k][i][j];
                    cell.vertices[5] = shoulder_vi[k][i][jn];
                    cell.vertices[6] = shoulder_vi[k][i+1][j];
                    cell.vertices[7] = shoulder_vi[k][i+1][jn];
                    cell.manifold_id = (k < n_inner) ? 1 : 2;
                    all_cells.push_back(cell);
                }

        std::cout << "  Building cone..." << std::endl;
        std::vector<std::vector<std::vector<unsigned int>>> cone_vi(
            n_total_radial + 1, std::vector<std::vector<unsigned int>>(
                n_cone + 1, std::vector<unsigned int>(n_phi)));

        for (unsigned int k = 0; k <= n_total_radial; ++k)
            for (unsigned int i = 0; i <= n_cone; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    if (i == 0) {
                        cone_vi[k][i][j] = shoulder_vi[k][n_shoulder][j];
                    } else {
                        double phi = boundary_angles[j].first;
                        Point<3> p_body(cone_xR[i].first,
                                        cone_xR[i].second * std::cos(phi),
                                        cone_xR[i].second * std::sin(phi));
                        all_vertices.push_back(get_radial_point(p_body, k));
                        cone_vi[k][i][j] = all_vertices.size() - 1;
                    }
                }

        for (unsigned int k = 0; k < n_total_radial; ++k)
            for (unsigned int i = 0; i < n_cone; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    unsigned int jn = (j + 1) % n_phi;
                    CellData<3> cell;

                    cell.vertices[0] = cone_vi[k+1][i][j];
                    cell.vertices[1] = cone_vi[k+1][i][jn];
                    cell.vertices[2] = cone_vi[k+1][i+1][j];
                    cell.vertices[3] = cone_vi[k+1][i+1][jn];
                    cell.vertices[4] = cone_vi[k][i][j];
                    cell.vertices[5] = cone_vi[k][i][jn];
                    cell.vertices[6] = cone_vi[k][i+1][j];
                    cell.vertices[7] = cone_vi[k][i+1][jn];
                    cell.manifold_id = (k < n_inner) ? 1 : 2;
                    all_cells.push_back(cell);
                }

        std::cout << "  Building fillet..." << std::endl;
        std::vector<std::vector<std::vector<unsigned int>>> fillet_vi(
            n_total_radial + 1, std::vector<std::vector<unsigned int>>(
                n_fillet + 1, std::vector<unsigned int>(n_phi)));

        for (unsigned int k = 0; k <= n_total_radial; ++k)
            for (unsigned int i = 0; i <= n_fillet; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    if (i == 0) {
                        fillet_vi[k][i][j] = cone_vi[k][n_cone][j];
                    } else {
                        double phi = boundary_angles[j].first;
                        Point<3> p_body(fillet_xR[i].first,
                                        fillet_xR[i].second * std::cos(phi),
                                        fillet_xR[i].second * std::sin(phi));
                        all_vertices.push_back(get_radial_point(p_body, k));
                        fillet_vi[k][i][j] = all_vertices.size() - 1;
                    }
                }

        for (unsigned int k = 0; k < n_total_radial; ++k)
            for (unsigned int i = 0; i < n_fillet; ++i)
                for (unsigned int j = 0; j < n_phi; ++j) {
                    unsigned int jn = (j + 1) % n_phi;
                    CellData<3> cell;

                    cell.vertices[0] = fillet_vi[k+1][i][j];
                    cell.vertices[1] = fillet_vi[k+1][i][jn];
                    cell.vertices[2] = fillet_vi[k+1][i+1][j];
                    cell.vertices[3] = fillet_vi[k+1][i+1][jn];
                    cell.vertices[4] = fillet_vi[k][i][j];
                    cell.vertices[5] = fillet_vi[k][i][jn];
                    cell.vertices[6] = fillet_vi[k][i+1][j];
                    cell.vertices[7] = fillet_vi[k][i+1][jn];
                    cell.manifold_id = (k < n_inner) ? 1 : 2;
                    all_cells.push_back(cell);
                }

        std::cout << "  Building nose cap..." << std::endl;
        std::vector<std::vector<unsigned int>> nose_vi(n_total_radial + 1);

        for (unsigned int k = 0; k <= n_total_radial; ++k) {
            nose_vi[k].resize(disk.n_vertices());
            for (unsigned int vi = 0; vi < disk.n_vertices(); ++vi) {
                bool is_boundary = false;
                unsigned int boundary_j = 0;
                for (unsigned int j = 0; j < n_phi; ++j)
                    if (boundary_angles[j].second == vi) {
                        is_boundary = true;
                        boundary_j = j;
                        break;
                    }

                if (is_boundary) {
                    nose_vi[k][vi] = fillet_vi[k][n_fillet][boundary_j];
                } else {
                    Point<2> p2d = disk.get_vertices()[vi];
                    Point<3> p_body(0.0, p2d[0] * R_A1, p2d[1] * R_A1);
                    all_vertices.push_back(get_radial_point(p_body, k));
                    nose_vi[k][vi] = all_vertices.size() - 1;
                }
            }
        }

        for (unsigned int k = 0; k < n_total_radial; ++k)
            for (const auto &cell2d : disk.active_cell_iterators()) {
                CellData<3> cell;

                cell.vertices[0] = nose_vi[k+1][cell2d->vertex_index(0)];
                cell.vertices[1] = nose_vi[k+1][cell2d->vertex_index(1)];
                cell.vertices[2] = nose_vi[k+1][cell2d->vertex_index(2)];
                cell.vertices[3] = nose_vi[k+1][cell2d->vertex_index(3)];
                cell.vertices[4] = nose_vi[k][cell2d->vertex_index(0)];
                cell.vertices[5] = nose_vi[k][cell2d->vertex_index(1)];
                cell.vertices[6] = nose_vi[k][cell2d->vertex_index(2)];
                cell.vertices[7] = nose_vi[k][cell2d->vertex_index(3)];
                cell.manifold_id = (k < n_inner) ? 1 : 2;
                all_cells.push_back(cell);
            }

        std::cout << "  Building shield..." << std::endl;
        const double L_shield = std::tan(theta_shield);
        std::vector<std::vector<unsigned int>> shield_vi(n_total_radial + 1);

        for (unsigned int k = 0; k <= n_total_radial; ++k) {
            shield_vi[k].resize(disk.n_vertices());
            for (unsigned int vi = 0; vi < disk.n_vertices(); ++vi) {
                bool is_boundary = false;
                unsigned int boundary_j = 0;
                for (unsigned int j = 0; j < n_phi; ++j)
                    if (boundary_angles[j].second == vi) {
                        is_boundary = true;
                        boundary_j = j;
                        break;
                    }

                if (is_boundary) {
                    shield_vi[k][vi] = shoulder_vi[k][0][boundary_j];
                } else {
                    Point<2> p2d = disk.get_vertices()[vi];
                    double u = p2d[0] * L_shield;
                    double w = p2d[1] * L_shield;
                    double norm = std::sqrt(1.0 + u * u + w * w);
                    Point<3> p_body(cx_shield + r3 / norm, r3 * u / norm, r3 * w / norm);
                    all_vertices.push_back(get_radial_point(p_body, k));
                    shield_vi[k][vi] = all_vertices.size() - 1;
                }
            }
        }

        for (unsigned int k = 0; k < n_total_radial; ++k)
            for (const auto &cell2d : disk.active_cell_iterators()) {
                CellData<3> cell;

                cell.vertices[0] = shield_vi[k][cell2d->vertex_index(0)];
                cell.vertices[1] = shield_vi[k][cell2d->vertex_index(1)];
                cell.vertices[2] = shield_vi[k][cell2d->vertex_index(2)];
                cell.vertices[3] = shield_vi[k][cell2d->vertex_index(3)];
                cell.vertices[4] = shield_vi[k+1][cell2d->vertex_index(0)];
                cell.vertices[5] = shield_vi[k+1][cell2d->vertex_index(1)];
                cell.vertices[6] = shield_vi[k+1][cell2d->vertex_index(2)];
                cell.vertices[7] = shield_vi[k+1][cell2d->vertex_index(3)];
                cell.manifold_id = (k < n_inner) ? 1 : 2;
                all_cells.push_back(cell);
            }

        std::cout << "  Creating triangulation with " << all_vertices.size()
                  << " vertices and " << all_cells.size() << " cells..." << std::endl;

        GridTools::consistently_order_cells(all_cells);

        triangulation.create_triangulation(all_vertices, all_cells, SubCellData());

        std::cout << "  Verifying mesh..." << std::endl;

        unsigned int n_negative_measure = 0;
        unsigned int n_non_conforming = 0;
        unsigned int n_boundary_faces = 0;
        unsigned int n_interior_faces = 0;
        double min_measure = std::numeric_limits<double>::max();
        double max_measure = 0.0;

        for (const auto &cell : triangulation.active_cell_iterators()) {

            double measure = cell->measure();
            min_measure = std::min(min_measure, measure);
            max_measure = std::max(max_measure, measure);
            if (measure <= 0) {
                std::cerr << "ERROR: Cell " << cell->index() << " has non-positive measure: "
                          << measure << std::endl;
                n_negative_measure++;
            }

            for (unsigned int f = 0; f < cell->n_faces(); ++f) {
                auto face = cell->face(f);
                if (face->at_boundary()) {
                    n_boundary_faces++;
                } else {
                    n_interior_faces++;

                    if (cell->neighbor(f).state() != IteratorState::valid) {
                        std::cerr << "ERROR: Cell " << cell->index()
                                  << " face " << f << " has no valid neighbor" << std::endl;
                        n_non_conforming++;
                        continue;
                    }

                    auto neighbor = cell->neighbor(f);
                    unsigned int neighbor_face_idx = cell->neighbor_of_neighbor(f);
                    auto neighbor_face = neighbor->face(neighbor_face_idx);

                    Point<3> face_center = face->center();
                    Point<3> neighbor_face_center = neighbor_face->center();
                    double center_dist = (face_center - neighbor_face_center).norm();

                    if (center_dist > 1e-10) {
                        std::cerr << "ERROR: Non-conforming face between cells "
                                  << cell->index() << " and " << neighbor->index()
                                  << ", center distance: " << center_dist << std::endl;
                        n_non_conforming++;
                    }

                    for (unsigned int v = 0; v < face->n_vertices(); ++v) {
                        Point<3> vertex = face->vertex(v);
                        bool found = false;
                        for (unsigned int nv = 0; nv < neighbor_face->n_vertices(); ++nv) {
                            if ((vertex - neighbor_face->vertex(nv)).norm() < 1e-10) {
                                found = true;
                                break;
                            }
                        }
                        if (!found) {
                            std::cerr << "ERROR: Vertex " << vertex << " of cell "
                                      << cell->index() << " not found in neighbor face" << std::endl;
                            n_non_conforming++;
                        }
                    }
                }
            }
        }

        n_interior_faces /= 2;

        std::cout << "  Mesh statistics:" << std::endl;
        std::cout << "    Cells: " << triangulation.n_active_cells() << std::endl;
        std::cout << "    Vertices: " << triangulation.n_vertices() << std::endl;
        std::cout << "    Boundary faces: " << n_boundary_faces << std::endl;
        std::cout << "    Interior faces: " << n_interior_faces << std::endl;
        std::cout << "    Cell measure range: [" << min_measure << ", " << max_measure << "]" << std::endl;
        std::cout << "    Measure ratio (max/min): " << max_measure / min_measure << std::endl;

        if (n_negative_measure > 0)
            std::cerr << "  WARNING: " << n_negative_measure << " cells with non-positive measure!" << std::endl;
        if (n_non_conforming > 0)
            std::cerr << "  WARNING: " << n_non_conforming << " non-conforming face issues!" << std::endl;

        if (n_negative_measure == 0 && n_non_conforming == 0)
            std::cout << "  Mesh verification PASSED - all cells valid and conforming." << std::endl;
        else
            std::cerr << "  Mesh verification FAILED!" << std::endl;

        double min_dynamic_x = 1e10, max_dynamic_x = -1e10;
        double min_noslip_x = 1e10, max_noslip_x = -1e10;

        for (auto &cell : triangulation.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary()) {
                    Point<3> c = face->center();
                    double r = (c - center).norm();
                    if (r > 0.9 * R_outer) {

                        face->set_boundary_id(DYNAMIC);
                        min_dynamic_x = std::min(min_dynamic_x, c[0]);
                        max_dynamic_x = std::max(max_dynamic_x, c[0]);
                    } else {

                        face->set_boundary_id(NO_SLIP);
                        min_noslip_x = std::min(min_noslip_x, c[0]);
                        max_noslip_x = std::max(max_noslip_x, c[0]);
                    }
                }

        std::cout << "  Boundary face x-ranges:" << std::endl;
        std::cout << "    DYNAMIC (far-field): x in [" << min_dynamic_x << ", " << max_dynamic_x << "]" << std::endl;
        std::cout << "    NO_SLIP (capsule): x in [" << min_noslip_x << ", " << max_noslip_x << "]" << std::endl;
        std::cout << "    Mesh center: x = " << center[0] << std::endl;

        FlatManifold<3> flat_manifold;
        triangulation.set_manifold(1, flat_manifold);
        triangulation.set_manifold(14, flat_manifold);

        SphericalManifold<3> spherical_manifold(center);
        triangulation.set_manifold(2, spherical_manifold);

        const Point<3> shield_center(cx_shield, 0.0, 0.0);
        SphericalManifold<3> shield_manifold(shield_center);
        triangulation.set_manifold(10, shield_manifold);

        TorusSectionManifold<3, 3> shoulder_manifold(cx_shoulder, cy_shoulder, r2);
        triangulation.set_manifold(11, shoulder_manifold);

        ConeManifold<3, 3> cone_manifold(x_A2, R_A2, x_B1, R_B1);
        triangulation.set_manifold(12, cone_manifold);

        TorusSectionManifold<3, 3> fillet_manifold(cx_fillet, cy_fillet, r1_fillet);
        triangulation.set_manifold(13, fillet_manifold);

        for (auto &cell : triangulation.active_cell_iterators())
            for (const auto &face : cell->face_iterators())
                if (face->at_boundary()) {
                    Point<3> c = face->center();
                    double r = (c - center).norm();

                    if (r > 0.9 * R_outer) {

                        face->set_all_manifold_ids(2);
                    } else {

                        double x = c[0];
                        if (x > 113.0)        
                            face->set_all_manifold_ids(10);
                        else if (x > 99.0)    
                            face->set_all_manifold_ids(11);
                        else if (x > 1.0)     
                            face->set_all_manifold_ids(12);
                        else if (x > 0.1)     
                            face->set_all_manifold_ids(13);
                        else                  
                            face->set_all_manifold_ids(14);
                    }
                }

        if (config.mesh_refinement > 0)
            triangulation.refine_global(config.mesh_refinement);

        print_mesh_info(triangulation, "NASA Capsule");
    }

private:
    template<int dim>
    static void print_mesh_info(const Triangulation<dim>& tria, 
                                  const std::string& name)
    {
        std::cout << "\n" << name << " mesh:" << std::endl;
        std::cout << "  Active cells: " << tria.n_active_cells() << std::endl;
        std::cout << "  Vertices: " << tria.n_vertices() << std::endl;
        
        std::map<types::boundary_id, unsigned int> boundary_count;
        for (auto cell : tria.active_cell_iterators()) {
            for (auto f : cell->face_indices()) {
                if (cell->face(f)->at_boundary()) {
                    boundary_count[cell->face(f)->boundary_id()]++;
                }
            }
        }
        
        std::cout << "  Boundary faces:" << std::endl;
        for (const auto& [bid, count] : boundary_count) {
            std::cout << "    Boundary ID " << bid << ": " << count << " faces";
            if (bid == DO_NOTHING) std::cout << " (DO_NOTHING)";
            else if (bid == PERIODIC) std::cout << " (PERIODIC)";
            else if (bid == SLIP) std::cout << " (SLIP)";
            else if (bid == NO_SLIP) std::cout << " (NO_SLIP/airfoil)";
            else if (bid == DIRICHLET) std::cout << " (DIRICHLET)";
            else if (bid == DYNAMIC) std::cout << " (DYNAMIC/far-field)";
            else
                std::cout << "error, bid = " << bid << std::endl;
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
};

#endif
