
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <deal.II/base/parameter_handler.h>
#include <string>
#include <array>
#include <sstream>
#include <fstream>
#include <iostream>

class Configuration {
public:

    std::string basename = "simulation";
    double final_time = 1.0;
    double timer_granularity = 0.01;

    int dimension = 2;
    std::string equation = "euler";
    double gamma = 1.4;
    double mu = 0.0;      
    double lambda = 0.0;  
    double kappa = 0.0;   

    std::string geometry_type = "cylinder";
    int mesh_refinement = 2;

    double height = 2.0;
    double length = 4.0;
    double object_diameter = 0.5;
    double object_position = 0.6;

    double length_before_sphere = 2;
    double length_after_sphere = 8;
    double height_below_sphere = 2;
    double height_above_sphere = 2;
    double depth = 2;

    int rep_before_sphere = 2;
    int rep_after_sphere = 8;
    int rep_below_sphere = 2;
    int rep_above_sphere = 2;
    int rep_depth = 2;    

    double inner_radius = 0.75;
    double outer_radius = 0.75;
    int n_cells = 6;

    std::string airfoil_type = "ONERA OAT15a";
    std::array<double, 2> airfoil_center = {{-0.0613, 0.0}};
    double airfoil_length = 0.2300;
    std::array<double, 2> psi_center = {{0.03, 0.0}};
    double psi_ratio = 3.00;
    double airfoil_height = 1.0;  
    double width = 0.046;         
    double grading_exponent = 6.0;
    double grading_epsilon = 0.0005;
    double grading_epsilon_trailing = 0.0100;
    int anisotropic_pre_refinement_airfoil = 1;
    int anisotropic_pre_refinement_trailing = 3;
    int subdivisions_z = 9;

    unsigned int length_before_cylinder = 3;
    unsigned int length_after_cylinder  = 9;
    unsigned int height_below_cylinder  = 3;
    unsigned int height_above_cylinder  = 3;
    double       cwc_shell_region_radius    = 0.75;
    unsigned int cwc_n_shells               = 2;
    double       cwc_skewness               = 2.0;
    bool         cwc_use_transfinite_region = true;

    double sphere_diameter_multiplier = 5.0;  

    double rect_x_left = -0.25;
    double rect_x_right = 0.25;
    double rect_y_bottom = 0.0;
    double rect_y_top = 0.01;      
    int rect_subdivisions_x = 100;
    int rect_subdivisions_y = 1;

    std::string mesh_file_path = "";

    std::string initial_condition = "uniform";  
    double becker_velocity_left = 1.0;
    double becker_velocity_right = 0.259259259259;  
    double becker_density_left = 1.0;
    double becker_velocity_galilean = 0.2;
    double becker_position = 0.0;  
    bool becker_verification = false;

    bool rect_periodic_y = false;

    std::array<double, 3> primitive_state = {{1.4, 3.0, 1.0}};  
    std::array<double, 3> direction = {{1.0, 0.0, 0.0}};  

    double cfl_min = 0.9;
    double cfl_max = 0.9;
    double cfl_number = 0.9;  

    bool is_navier_stokes() const {
        return (equation == "navier stokes" || equation == "navier_stokes");
    }

    bool is_airfoil() const {
        return (geometry_type == "airfoil");
    }
    
    void read_parameters(const std::string& filename) {
        read_parameters_impl(filename);
    }
    
private:

    std::string get_geometry_type_from_file(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        bool in_discretization = false;
        
        while (std::getline(file, line)) {
            size_t start = line.find_first_not_of(" \t");
            if (start == std::string::npos) continue;
            line = line.substr(start);
            
            if (line.find("subsection C - Discretization") != std::string::npos) {
                in_discretization = true;
                continue;
            }
            
            if (in_discretization && line.find("end") != std::string::npos) {
                break;
            }
            
            if (in_discretization && line.find("set geometry") != std::string::npos) {
                size_t eq_pos = line.find('=');
                if (eq_pos != std::string::npos) {
                    std::string value = line.substr(eq_pos + 1);
                    size_t comment_pos = value.find('#');
                    if (comment_pos != std::string::npos) {
                        value = value.substr(0, comment_pos);
                    }
                    start = value.find_first_not_of(" \t");
                    size_t end = value.find_last_not_of(" \t\r\n");
                    if (start != std::string::npos && end != std::string::npos) {
                        return value.substr(start, end - start + 1);
                    }
                }
            }
        }
        return "cylinder";
    }
    
    void declare_geometry_parameters(dealii::ParameterHandler& prm, const std::string& geom_type) {
        if (geom_type == "cylinder") {
            prm.enter_subsection("cylinder");
            {
                prm.declare_entry("height", "2", dealii::Patterns::Integer());
                prm.declare_entry("length", "4", dealii::Patterns::Integer());
                prm.declare_entry("object diameter", "0.5", dealii::Patterns::Double());
                prm.declare_entry("object position", "0.6", dealii::Patterns::Double());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "sphere_channel") {
            prm.enter_subsection("sphere_channel");
            {
                prm.declare_entry("length before sphere", "2", dealii::Patterns::Double());
                prm.declare_entry("length after sphere", "8", dealii::Patterns::Double());
                prm.declare_entry("height below sphere", "2", dealii::Patterns::Double());
                prm.declare_entry("height above sphere", "2", dealii::Patterns::Double());
                prm.declare_entry("depth", "1", dealii::Patterns::Double());

                prm.declare_entry("rep before sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("rep after sphere", "8", dealii::Patterns::Integer());
                prm.declare_entry("rep below sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("rep above sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("rep depth", "1", dealii::Patterns::Integer());                

                prm.declare_entry("inner radius", "0.5", dealii::Patterns::Double());
                prm.declare_entry("outer radius", "0.75", dealii::Patterns::Double());
                prm.declare_entry("number of cells", "6", dealii::Patterns::Integer());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "channel with cylinder") {
            prm.enter_subsection("channel with cylinder");
            {
                prm.declare_entry("length before cylinder", "3", dealii::Patterns::Integer());
                prm.declare_entry("length after cylinder", "9", dealii::Patterns::Integer());
                prm.declare_entry("height below cylinder", "3", dealii::Patterns::Integer());
                prm.declare_entry("height above cylinder", "3", dealii::Patterns::Integer());
                prm.declare_entry("shell region radius", "0.75", dealii::Patterns::Double());
                prm.declare_entry("number of shells", "2", dealii::Patterns::Integer());
                prm.declare_entry("skewness", "2.0", dealii::Patterns::Double());
                prm.declare_entry("use transfinite region", "true", dealii::Patterns::Bool());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "airfoil") {
            prm.enter_subsection("airfoil");
            {
                prm.declare_entry("airfoil type", "ONERA OAT15a", dealii::Patterns::Anything());
                prm.declare_entry("airfoil center", "-0.0613, 0.", dealii::Patterns::Anything());
                prm.declare_entry("airfoil length", "0.2300", dealii::Patterns::Double());
                prm.declare_entry("psi center", "0.03, 0.", dealii::Patterns::Anything());
                prm.declare_entry("psi ratio", "3.00", dealii::Patterns::Double());
                prm.declare_entry("height", "1.0", dealii::Patterns::Double());
                prm.declare_entry("width", "0.046", dealii::Patterns::Double());
                prm.declare_entry("grading exponent", "6.0", dealii::Patterns::Double());
                prm.declare_entry("grading epsilon", "0.0005", dealii::Patterns::Double());
                prm.declare_entry("grading epsilon trailing", "0.0100", dealii::Patterns::Double());
                prm.declare_entry("anisotropic pre refinement airfoil", "1", dealii::Patterns::Integer());
                prm.declare_entry("anisotropic pre refinement trailing", "3", dealii::Patterns::Integer());
                prm.declare_entry("subdivisions z", "9", dealii::Patterns::Integer());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "capsule") {
            prm.enter_subsection("capsule");
            {
                prm.declare_entry("sphere diameter multiplier", "5.0", dealii::Patterns::Double());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "rectangle") {
            prm.enter_subsection("rectangle");
            {
                prm.declare_entry("x left", "-0.25", dealii::Patterns::Double());
                prm.declare_entry("x right", "0.25", dealii::Patterns::Double());
                prm.declare_entry("y bottom", "0.0", dealii::Patterns::Double());
                prm.declare_entry("y top", "0.01", dealii::Patterns::Double());
                prm.declare_entry("subdivisions x", "100", dealii::Patterns::Integer());
                prm.declare_entry("subdivisions y", "1", dealii::Patterns::Integer());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "mesh_file") {
            prm.enter_subsection("mesh_file");
            {
                prm.declare_entry("file path", "", dealii::Patterns::Anything());
            }
            prm.leave_subsection();
        }
    }

    void read_geometry_parameters(dealii::ParameterHandler& prm, const std::string& geom_type) {
        if (geom_type == "cylinder") {
            prm.enter_subsection("cylinder");
            {
                height = prm.get_double("height");
                length = prm.get_double("length");
                object_diameter = prm.get_double("object diameter");
                object_position = prm.get_double("object position");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "sphere_channel") {
            prm.enter_subsection("sphere_channel");
            {
                length_before_sphere = prm.get_double("length before sphere");
                length_after_sphere = prm.get_double("length after sphere");
                height_below_sphere = prm.get_double("height below sphere");
                height_above_sphere = prm.get_double("height above sphere");
                depth = prm.get_double("depth");

                rep_before_sphere = prm.get_integer("rep before sphere");
                rep_after_sphere = prm.get_integer("rep after sphere");
                rep_below_sphere = prm.get_integer("rep below sphere");
                rep_above_sphere = prm.get_integer("rep above sphere");
                rep_depth = prm.get_integer("rep depth");

                inner_radius = prm.get_double("inner radius");
                outer_radius = prm.get_double("outer radius");
                n_cells = prm.get_integer("number of cells");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "channel with cylinder") {
            prm.enter_subsection("channel with cylinder");
            {
                length_before_cylinder = prm.get_integer("length before cylinder");
                length_after_cylinder  = prm.get_integer("length after cylinder");
                height_below_cylinder  = prm.get_integer("height below cylinder");
                height_above_cylinder  = prm.get_integer("height above cylinder");
                cwc_shell_region_radius    = prm.get_double("shell region radius");
                cwc_n_shells               = prm.get_integer("number of shells");
                cwc_skewness               = prm.get_double("skewness");
                cwc_use_transfinite_region = prm.get_bool("use transfinite region");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "airfoil") {
            prm.enter_subsection("airfoil");
            {
                airfoil_type = prm.get("airfoil type");

                std::string center_str = prm.get("airfoil center");
                std::stringstream ss_center(center_str);
                char comma;
                ss_center >> airfoil_center[0] >> comma >> airfoil_center[1];

                airfoil_length = prm.get_double("airfoil length");

                std::string psi_center_str = prm.get("psi center");
                std::stringstream ss_psi(psi_center_str);
                ss_psi >> psi_center[0] >> comma >> psi_center[1];

                psi_ratio = prm.get_double("psi ratio");
                airfoil_height = prm.get_double("height");
                width = prm.get_double("width");
                grading_exponent = prm.get_double("grading exponent");
                grading_epsilon = prm.get_double("grading epsilon");
                grading_epsilon_trailing = prm.get_double("grading epsilon trailing");
                anisotropic_pre_refinement_airfoil = prm.get_integer("anisotropic pre refinement airfoil");
                anisotropic_pre_refinement_trailing = prm.get_integer("anisotropic pre refinement trailing");
                subdivisions_z = prm.get_integer("subdivisions z");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "capsule") {
            prm.enter_subsection("capsule");
            {
                sphere_diameter_multiplier = prm.get_double("sphere diameter multiplier");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "rectangle") {
            prm.enter_subsection("rectangle");
            {
                rect_x_left = prm.get_double("x left");
                rect_x_right = prm.get_double("x right");
                rect_y_bottom = prm.get_double("y bottom");
                rect_y_top = prm.get_double("y top");
                rect_subdivisions_x = prm.get_integer("subdivisions x");
                rect_subdivisions_y = prm.get_integer("subdivisions y");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "mesh_file") {
            prm.enter_subsection("mesh_file");
            {
                mesh_file_path = prm.get("file path");
            }
            prm.leave_subsection();
        }
    }

    void read_parameters_impl(const std::string& filename) {

        std::ifstream file_check(filename);
        if (!file_check.good()) {
            throw std::runtime_error("Parameter file not found: " + filename);
        }
        file_check.close();

        std::string geom_type = get_geometry_type_from_file(filename);
        
        dealii::ParameterHandler prm;

        prm.enter_subsection("A - TimeLoop");
        {
            prm.declare_entry("basename", "simulation", dealii::Patterns::Anything());
            prm.declare_entry("final time", "1.0", dealii::Patterns::Double());
            prm.declare_entry("timer granularity", "0.01", dealii::Patterns::Double());
        }
        prm.leave_subsection();

        prm.enter_subsection("B - Equation");
        {
            prm.declare_entry("dimension", "2", dealii::Patterns::Integer());
            prm.declare_entry("equation", "euler", dealii::Patterns::Anything());
            prm.declare_entry("gamma", "1.4", dealii::Patterns::Double());
            prm.declare_entry("mu", "0.0", dealii::Patterns::Double());
            prm.declare_entry("lambda", "0.0", dealii::Patterns::Double());
            prm.declare_entry("kappa", "0.0", dealii::Patterns::Double());
        }
        prm.leave_subsection();

        prm.enter_subsection("C - Discretization");
        {
            prm.declare_entry("geometry", "cylinder", dealii::Patterns::Anything());
            prm.declare_entry("mesh refinement", "2", dealii::Patterns::Integer());
            declare_geometry_parameters(prm, geom_type);
        }
        prm.leave_subsection();

        prm.enter_subsection("E - InitialValues");
        {
            prm.declare_entry("configuration", "uniform", dealii::Patterns::Anything());
            prm.declare_entry("direction", "1, 0", dealii::Patterns::Anything());
            prm.declare_entry("position", "0.0", dealii::Patterns::Double());
            prm.enter_subsection("uniform");
            {
                prm.declare_entry("primitive state", "1.4, 3, 1", dealii::Patterns::Anything());
            }
            prm.leave_subsection();
            prm.enter_subsection("becker solution");
            {
                prm.declare_entry("velocity galilean frame", "0.2", dealii::Patterns::Double());
                prm.declare_entry("velocity left", "1.0", dealii::Patterns::Double());
                prm.declare_entry("velocity right", "0.259259259259", dealii::Patterns::Double());
                prm.declare_entry("density left", "1.0", dealii::Patterns::Double());
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();

        prm.enter_subsection("H - TimeIntegrator");
        {
            prm.declare_entry("cfl min", "0.90", dealii::Patterns::Double());
            prm.declare_entry("cfl max", "0.90", dealii::Patterns::Double());
        }
        prm.leave_subsection();

        prm.parse_input(filename);

        prm.enter_subsection("A - TimeLoop");
        {
            basename = prm.get("basename");
            final_time = prm.get_double("final time");
            timer_granularity = prm.get_double("timer granularity");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("B - Equation");
        {
            dimension = prm.get_integer("dimension");
            equation = prm.get("equation");
            gamma = prm.get_double("gamma");
            mu = prm.get_double("mu");
            lambda = prm.get_double("lambda");
            kappa = prm.get_double("kappa");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("C - Discretization");
        {
            geometry_type = prm.get("geometry");
            mesh_refinement = prm.get_integer("mesh refinement");
            read_geometry_parameters(prm, geometry_type);
        }
        prm.leave_subsection();
        
        prm.enter_subsection("E - InitialValues");
        {
            initial_condition = prm.get("configuration");
            becker_verification = (initial_condition == "becker solution");

            if (becker_verification) rect_periodic_y = true;

            std::string direction_str = prm.get("direction");
            std::stringstream ss_dir(direction_str);
            char comma;
            ss_dir >> direction[0] >> comma >> direction[1];

            if (ss_dir >> comma >> direction[2]) {

            } else {
                direction[2] = 0.0;  
            }

            prm.enter_subsection("uniform");
            {
                std::string state_str = prm.get("primitive state");
                std::stringstream ss(state_str);
                ss >> primitive_state[0] >> comma >> primitive_state[1] >> comma >> primitive_state[2];
            }
            prm.leave_subsection();

            becker_position = prm.get_double("position");

            prm.enter_subsection("becker solution");
            {
                becker_velocity_galilean = prm.get_double("velocity galilean frame");
                becker_velocity_left = prm.get_double("velocity left");
                becker_velocity_right = prm.get_double("velocity right");
                becker_density_left = prm.get_double("density left");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        prm.enter_subsection("H - TimeIntegrator");
        {
            cfl_min = prm.get_double("cfl min");
            cfl_max = prm.get_double("cfl max");
            cfl_number = cfl_min;  
        }
        prm.leave_subsection();
        
        print_configuration(filename);
    }
    
    void print_configuration(const std::string& filename) const {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Configuration loaded from: " << filename << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  Basename: " << basename << std::endl;
        std::cout << "  Dimension: " << dimension << "D" << std::endl;
        std::cout << "  Geometry: " << geometry_type << std::endl;
        
        if (is_airfoil()) {
            std::cout << "  NOTE: Airfoil mesh will be READ from file (oat15a_mesh.msh)" << std::endl;
            std::cout << "        NOT generated programmatically" << std::endl;
        }
        
        std::cout << "  Equation: " << equation << std::endl;
        std::cout << "  Final time: " << final_time << std::endl;
        std::cout << "  CFL: " << cfl_number << " (min: " << cfl_min << ", max: " << cfl_max << ")" << std::endl;
        std::cout << "  Mesh refinement: " << mesh_refinement << std::endl;
        
        std::cout << "\n  Physics parameters:" << std::endl;
        std::cout << "    gamma: " << gamma << std::endl;
        if (is_navier_stokes()) {
            std::cout << "    mu (dynamic viscosity): " << mu << std::endl;
            std::cout << "    lambda (bulk viscosity): " << lambda << std::endl;
            std::cout << "    kappa (thermal conductivity): " << kappa << std::endl;
        }
        
        if (geometry_type == "cylinder") {
            std::cout << "\n  Cylinder geometry:" << std::endl;
            std::cout << "    Height: " << height << std::endl;
            std::cout << "    Length: " << length << std::endl;
            std::cout << "    Object diameter: " << object_diameter << std::endl;
            std::cout << "    Object position: " << object_position << std::endl;
        }
        else if (geometry_type == "sphere_channel") {
            std::cout << "\n  Sphere channel geometry:" << std::endl;
            std::cout << "    Length before/after: " << length_before_sphere << "/" << length_after_sphere << std::endl;
            std::cout << "    Height below/above: " << height_below_sphere << "/" << height_above_sphere << std::endl;
            std::cout << "    Depth: " << depth << std::endl;
            std::cout << "    Inner radius: " << inner_radius << std::endl;
            std::cout << "    Outer radius: " << outer_radius << std::endl;
            std::cout << "    Number of cells: " << n_cells << std::endl;
        }
        else if (geometry_type == "channel with cylinder") {
            std::cout << "\n  Channel with cylinder geometry:" << std::endl;
            std::cout << "    Length before/after: " << length_before_cylinder << "/" << length_after_cylinder << std::endl;
            std::cout << "    Height below/above: " << height_below_cylinder << "/" << height_above_cylinder << std::endl;
            std::cout << "    Shell region radius: " << cwc_shell_region_radius << std::endl;
            std::cout << "    Number of shells: " << cwc_n_shells << std::endl;
            std::cout << "    Skewness: " << cwc_skewness << std::endl;
            std::cout << "    Use transfinite region: " << cwc_use_transfinite_region << std::endl;
        }
        else if (geometry_type == "airfoil") {
            std::cout << "\n  Airfoil geometry (reference only - mesh from file):" << std::endl;
            std::cout << "    Airfoil type: " << airfoil_type << std::endl;
            std::cout << "    Length (chord): " << airfoil_length << std::endl;
            std::cout << "    Height: " << airfoil_height << std::endl;
        }
        else if (geometry_type == "capsule") {
            std::cout << "\n  Capsule geometry:" << std::endl;
            std::cout << "    Sphere diameter multiplier: " << sphere_diameter_multiplier << std::endl;
            std::cout << "    (Capsule dimensions are hard-coded)" << std::endl;
        }

        else if (geometry_type == "rectangle") {
            std::cout << "\n  Rectangle geometry:" << std::endl;
            std::cout << "    x: [" << rect_x_left << ", " << rect_x_right << "]" << std::endl;
            std::cout << "    y: [" << rect_y_bottom << ", " << rect_y_top << "]" << std::endl;
            std::cout << "    Subdivisions: " << rect_subdivisions_x << " x " << rect_subdivisions_y << std::endl;
        }
        else if (geometry_type == "mesh_file") {
            std::cout << "\n  Mesh-file geometry:" << std::endl;
            std::cout << "    File path: " << mesh_file_path << std::endl;
        }

        if (becker_verification) {
            std::cout << "\n  Becker verification parameters:" << std::endl;
            std::cout << "    velocity galilean: " << becker_velocity_galilean << std::endl;
            std::cout << "    velocity left: " << becker_velocity_left << std::endl;
            std::cout << "    velocity right: " << becker_velocity_right << std::endl;
            std::cout << "    density left: " << becker_density_left << std::endl;
        }

        std::cout << "\n  Initial state (primitive):" << std::endl;
        std::cout << "    rho: " << primitive_state[0] << std::endl;
        std::cout << "    u: " << primitive_state[1] << std::endl;
        std::cout << "    p: " << primitive_state[2] << std::endl;
        std::cout << "    direction: [" << direction[0] << ", " << direction[1] << ", " << direction[2] << "]" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

#endif