// configuration.h - Minimal configuration for mesh-reading solver
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <deal.II/base/parameter_handler.h>
#include <string>
#include <array>
#include <sstream>
#include <fstream>
#include <iostream>
#include <stdexcept>

class Configuration {
public:
    // A - TimeLoop
    std::string basename = "simulation";
    double final_time = 1.0;
    double timer_granularity = 0.01;
    
    // B - Equation
    int dimension = 2;
    double gamma = 1.4;
    double mu = 0.0;      // Dynamic viscosity (0 = Euler)
    double lambda = 0.0;  // Bulk viscosity
    double kappa = 0.0;   // Thermal conductivity (0 = Euler)
    
    // C - Discretization
    int mesh_refinement = 0;  // Refinement level for loaded mesh
    
    // E - InitialValues (primitive state: rho, velocity_magnitude, pressure)
    std::array<double, 3> primitive_state = {{1.4, 3.0, 1.0}};
    std::array<double, 2> direction = {{1.0, 0.0}};
    
    // H - TimeIntegrator
    double cfl_min = 0.9;
    double cfl_max = 0.9;
    double cfl_number = 0.9;
    
    // Helpers
    bool is_navier_stokes() const { return mu > 0.0 || kappa > 0.0; }
    
    void read_parameters(const std::string& filename) {
        std::ifstream file_check(filename);
        if (!file_check.good()) {
            throw std::runtime_error("Parameter file not found: " + filename);
        }
        file_check.close();
        
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
            prm.declare_entry("gamma", "1.4", dealii::Patterns::Double());
            prm.declare_entry("mu", "0.0", dealii::Patterns::Double());
            prm.declare_entry("lambda", "0.0", dealii::Patterns::Double());
            prm.declare_entry("kappa", "0.0", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // C - Discretization
        prm.enter_subsection("C - Discretization");
        {
            prm.declare_entry("mesh refinement", "0", dealii::Patterns::Integer());
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
            gamma = prm.get_double("gamma");
            mu = prm.get_double("mu");
            lambda = prm.get_double("lambda");
            kappa = prm.get_double("kappa");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("C - Discretization");
        {
            mesh_refinement = prm.get_integer("mesh refinement");
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
            cfl_number = cfl_min;
        }
        prm.leave_subsection();
        
        print_configuration();
    }
    
private:
    void print_configuration() const {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Configuration" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "  Basename: " << basename << std::endl;
        std::cout << "  Dimension: " << dimension << "D" << std::endl;
        std::cout << "  Physics: " << (is_navier_stokes() ? "Navier-Stokes" : "Euler") << std::endl;
        std::cout << "  Final time: " << final_time << std::endl;
        std::cout << "  Output interval: " << timer_granularity << std::endl;
        std::cout << "  CFL: " << cfl_number << std::endl;
        std::cout << "  Mesh refinement: " << mesh_refinement << std::endl;
        
        std::cout << "\n  Physics parameters:" << std::endl;
        std::cout << "    gamma: " << gamma << std::endl;
        if (is_navier_stokes()) {
            std::cout << "    mu: " << mu << std::endl;
            std::cout << "    kappa: " << kappa << std::endl;
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