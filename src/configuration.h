// configuration.h - Refactored to match ryujin parameter structure
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
    // A - TimeLoop
    std::string basename = "simulation";
    double final_time = 1.0;
    double timer_granularity = 0.01;
    
    // B - Equation
    int dimension = 2;
    std::string equation = "euler";
    double gamma = 1.4;
    double mu = 0.0;      // Dynamic viscosity
    double lambda = 0.0;  // Bulk viscosity
    double kappa = 0.0;   // Scaled thermal conductivity
    
    // C - Discretization
    std::string geometry_type = "cylinder";
    int mesh_refinement = 2;
    
    // Cylinder geometry
    double height = 2.0;
    double length = 4.0;
    double object_diameter = 0.5;
    double object_position = 0.6;
    
    // Sphere channel geometry
    int length_before_sphere = 2;
    int length_after_sphere = 8;
    int height_below_sphere = 2;
    int height_above_sphere = 2;
    int depth = 2;
    double shell_region_radius = 0.75;
    int n_shells = 6;
    
    // Airfoil geometry - Mesh is read from file
    std::string airfoil_type = "ONERA OAT15a";
    std::array<double, 2> airfoil_center = {{-0.0613, 0.0}};
    double airfoil_length = 0.2300;
    std::array<double, 2> psi_center = {{0.03, 0.0}};
    double psi_ratio = 3.00;
    double airfoil_height = 1.0;  // Full domain height
    double width = 0.046;         // For 3D extrusion
    double grading_exponent = 6.0;
    double grading_epsilon = 0.0005;
    double grading_epsilon_trailing = 0.0100;
    int anisotropic_pre_refinement_airfoil = 1;
    int anisotropic_pre_refinement_trailing = 3;
    int subdivisions_z = 9;
    
    // E - InitialValues
    std::array<double, 3> primitive_state = {{1.4, 3.0, 1.0}};  // [rho, u, p]
    std::array<double, 2> direction = {{1.0, 0.0}};
    
    // H - TimeIntegrator
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
                prm.declare_entry("height", "2", dealii::Patterns::Double());
                prm.declare_entry("length", "4", dealii::Patterns::Double());
                prm.declare_entry("object diameter", "0.5", dealii::Patterns::Double());
                prm.declare_entry("object position", "0.6", dealii::Patterns::Double());
            }
            prm.leave_subsection();
        }
        else if (geom_type == "sphere_channel") {
            prm.enter_subsection("sphere_channel");
            {
                prm.declare_entry("length before sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("length after sphere", "8", dealii::Patterns::Integer());
                prm.declare_entry("height below sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("height above sphere", "2", dealii::Patterns::Integer());
                prm.declare_entry("depth", "1", dealii::Patterns::Integer());
                prm.declare_entry("shell region radius", "0.75", dealii::Patterns::Double());
                prm.declare_entry("number of shells", "0", dealii::Patterns::Integer());
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
                length_before_sphere = prm.get_integer("length before sphere");
                length_after_sphere = prm.get_integer("length after sphere");
                height_below_sphere = prm.get_integer("height below sphere");
                height_above_sphere = prm.get_integer("height above sphere");
                depth = prm.get_integer("depth");
                shell_region_radius = prm.get_double("shell region radius");
                n_shells = prm.get_integer("number of shells");
            }
            prm.leave_subsection();
        }
        else if (geom_type == "airfoil") {
            prm.enter_subsection("airfoil");
            {
                airfoil_type = prm.get("airfoil type");
                
                // Parse airfoil center
                std::string center_str = prm.get("airfoil center");
                std::stringstream ss_center(center_str);
                char comma;
                ss_center >> airfoil_center[0] >> comma >> airfoil_center[1];
                
                airfoil_length = prm.get_double("airfoil length");
                
                // Parse psi center
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
    }
    
    void read_parameters_impl(const std::string& filename) {
        // Check file exists
        std::ifstream file_check(filename);
        if (!file_check.good()) {
            throw std::runtime_error("Parameter file not found: " + filename);
        }
        file_check.close();
        
        // Determine geometry type
        std::string geom_type = get_geometry_type_from_file(filename);
        
        dealii::ParameterHandler prm;
        
        // A - TimeLoop
        prm.enter_subsection("A - TimeLoop");
        {
            prm.declare_entry("basename", "simulation", dealii::Patterns::Anything());
            prm.declare_entry("final time", "1.0", dealii::Patterns::Double());
            prm.declare_entry("timer granularity", "0.01", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // B - Equation
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
        
        // C - Discretization
        prm.enter_subsection("C - Discretization");
        {
            prm.declare_entry("geometry", "cylinder", dealii::Patterns::Anything());
            prm.declare_entry("mesh refinement", "2", dealii::Patterns::Integer());
            declare_geometry_parameters(prm, geom_type);
        }
        prm.leave_subsection();
        
        // E - InitialValues
        prm.enter_subsection("E - InitialValues");
        {
            prm.declare_entry("direction", "1, 0", dealii::Patterns::Anything());
            prm.enter_subsection("uniform");
            {
                prm.declare_entry("primitive state", "1.4, 3, 1", dealii::Patterns::Anything());
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        // H - TimeIntegrator
        prm.enter_subsection("H - TimeIntegrator");
        {
            prm.declare_entry("cfl min", "0.90", dealii::Patterns::Double());
            prm.declare_entry("cfl max", "0.90", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // Parse file
        prm.parse_input(filename);
        
        // Read values
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
            // Parse direction
            std::string direction_str = prm.get("direction");
            std::stringstream ss_dir(direction_str);
            char comma;
            ss_dir >> direction[0] >> comma >> direction[1];
            
            prm.enter_subsection("uniform");
            {
                std::string state_str = prm.get("primitive state");
                std::stringstream ss(state_str);
                ss >> primitive_state[0] >> comma >> primitive_state[1] >> comma >> primitive_state[2];
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        prm.enter_subsection("H - TimeIntegrator");
        {
            cfl_min = prm.get_double("cfl min");
            cfl_max = prm.get_double("cfl max");
            cfl_number = cfl_min;  // Use cfl_min as the CFL number
        }
        prm.leave_subsection();
        
        print_configuration(filename);
    }
    
    void print_configuration(const std::string& filename) const {
        std::cout << "Configuration loaded from: " << filename << std::endl;
        std::cout << "  Basename: " << basename << std::endl;
        std::cout << "  Dimension: " << dimension << "D" << std::endl;
        std::cout << "  Geometry: " << geometry_type << std::endl;
        
        if (is_airfoil()) {
            std::cout << "  Airfoil mesh is read from file oat15a_mesh.msh" << std::endl;
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
            std::cout << "    Shell region radius: " << shell_region_radius << std::endl;
            std::cout << "    Number of shells: " << n_shells << std::endl;
        }
        else if (geometry_type == "airfoil") {
            std::cout << "\n  Airfoil geometry (reference only - mesh from file):" << std::endl;
            std::cout << "    Airfoil type: " << airfoil_type << std::endl;
            std::cout << "    Length (chord): " << airfoil_length << std::endl;
            std::cout << "    Height: " << airfoil_height << std::endl;
        }
        
        std::cout << "\n  Initial state (primitive):" << std::endl;
        std::cout << "    rho: " << primitive_state[0] << std::endl;
        std::cout << "    u: " << primitive_state[1] << std::endl;
        std::cout << "    p: " << primitive_state[2] << std::endl;
        std::cout << "    direction: [" << direction[0] << ", " << direction[1] << "]" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

#endif