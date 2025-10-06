// configuration.h
#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <deal.II/base/parameter_handler.h>
#include <string>
#include <array>
#include <sstream>

class Configuration {
public:
    // Simulation parameters
    std::string basename = "mach3-cylinder-2d";
    double final_time = 10.0;
    double cfl_number = 0.9;
    
    // Physics parameters
    static constexpr double gamma = 1.4;
    std::array<double, 3> primitive_state = {{1.4, 3.0, 1.0}}; // [rho, u, p]
    
    // Geometry parameters
    double height = 2.0;
    double length = 4.0;
    double object_diameter = 0.5;
    double object_position = 0.6;
    int mesh_refinement = 2;
    
    // Scheme parameters
    double relaxation_factor = 1.0;
    
    // Navier-Stokes parameters
    double reynolds_number = 1000.0;
    double prandtl_number = 0.75;
    double mach_number = 3.0;
    double mu_reference = 1.0e-3;
    double cv_inverse_kappa_reference = 1.866666666666666e-2;
    bool use_navier_stokes = true;  // Default to Euler
    
    void read_parameters(const std::string& filename) {
        dealii::ParameterHandler prm;
        
        // A - TimeLoop
        prm.enter_subsection("A - TimeLoop");
        {
            prm.declare_entry("basename", "mach3-cylinder-2d", dealii::Patterns::Anything());
            prm.declare_entry("enable output full", "true", dealii::Patterns::Bool());
            prm.declare_entry("final time", "10.0", dealii::Patterns::Double());
            prm.declare_entry("timer granularity", "0.1", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // B - Equation
        prm.enter_subsection("B - Equation");
        {
            prm.declare_entry("dimension", "2", dealii::Patterns::Integer());
            prm.declare_entry("equation", "euler", dealii::Patterns::Anything());
            prm.declare_entry("gamma", "1.4", dealii::Patterns::Double());
            prm.declare_entry("reynolds number", "1000", dealii::Patterns::Double());
            prm.declare_entry("prandtl number", "0.75", dealii::Patterns::Double());
            prm.declare_entry("mach number", "3.0", dealii::Patterns::Double());
            prm.declare_entry("mu reference", "1.0e-3", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // C - Discretization
        prm.enter_subsection("C - Discretization");
        {
            prm.declare_entry("geometry", "cylinder", dealii::Patterns::Anything());
            prm.declare_entry("mesh refinement", "2", dealii::Patterns::Integer());
            
            prm.enter_subsection("cylinder");
            {
                prm.declare_entry("height", "2", dealii::Patterns::Double());
                prm.declare_entry("length", "4", dealii::Patterns::Double());
                prm.declare_entry("object diameter", "0.5", dealii::Patterns::Double());
                prm.declare_entry("object position", "0.6", dealii::Patterns::Double());
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        // E - InitialValues
        prm.enter_subsection("E - InitialValues");
        {
            prm.declare_entry("configuration", "uniform", dealii::Patterns::Anything());
            prm.declare_entry("direction", "1, 0", dealii::Patterns::Anything());
            prm.declare_entry("position", "1, 0", dealii::Patterns::Anything());
            prm.declare_entry("perturbation", "0", dealii::Patterns::Double());
            
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
            prm.declare_entry("cfl recovery strategy", "none", dealii::Patterns::Anything());
            prm.declare_entry("time stepping scheme", "erk 33", dealii::Patterns::Anything());
        }
        prm.leave_subsection();
        
        // J - VTUOutput
        prm.enter_subsection("J - VTUOutput");
        {
            prm.declare_entry("use mpi io", "true", dealii::Patterns::Bool());
            prm.declare_entry("schlieren beta", "10", dealii::Patterns::Double());
        }
        prm.leave_subsection();
        
        // Parse input file
        prm.parse_input(filename);
        
        // Read values
        prm.enter_subsection("A - TimeLoop");
        {
            basename = prm.get("basename");
            final_time = prm.get_double("final time");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("B - Equation");
        {
            std::string equation_type = prm.get("equation");
            use_navier_stokes = (equation_type == "navier_stokes");
            reynolds_number = prm.get_double("reynolds number");
            prandtl_number = prm.get_double("prandtl number");
            mach_number = prm.get_double("mach number");
            mu_reference = prm.get_double("mu reference");
        }
        prm.leave_subsection();
        
        prm.enter_subsection("C - Discretization");
        {
            mesh_refinement = prm.get_integer("mesh refinement");
            prm.enter_subsection("cylinder");
            {
                height = prm.get_double("height");
                length = prm.get_double("length");
                object_diameter = prm.get_double("object diameter");
                object_position = prm.get_double("object position");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        prm.enter_subsection("E - InitialValues");
        {
            prm.enter_subsection("uniform");
            {
                std::string state_string = prm.get("primitive state");
                std::stringstream ss(state_string);
                char comma;
                ss >> primitive_state[0] >> comma >> primitive_state[1] >> comma >> primitive_state[2];
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();
        
        prm.enter_subsection("H - TimeIntegrator");
        {
            cfl_number = prm.get_double("cfl max");
        }
        prm.leave_subsection();
        
        // Compute derived viscosity parameters
        if (use_navier_stokes) {
            // Reference length = cylinder diameter
            double L_ref = object_diameter;
            // Reference velocity = Mach * sqrt(gamma * p_inf / rho_inf)
            double a_inf = std::sqrt(gamma * primitive_state[2] / primitive_state[0]);
            double U_ref = mach_number * a_inf;
            // mu = rho_inf * U_ref * L_ref / Re
            mu_reference = primitive_state[0] * U_ref * L_ref / reynolds_number;
            // cv_inverse_kappa = kappa / cv = kappa * gamma / (gamma - 1)
            cv_inverse_kappa_reference = mu_reference * gamma / ((gamma - 1.0) * prandtl_number) * gamma / (gamma - 1.0);
        }
        
        std::cout << "Parameters successfully read from: " << filename << std::endl;
        std::cout << "  Equation type: " << (use_navier_stokes ? "Navier-Stokes" : "Euler") << std::endl;
        std::cout << "  Final time: " << final_time << std::endl;
        std::cout << "  Mesh refinement: " << mesh_refinement << std::endl;
        if (use_navier_stokes) {
            std::cout << "  Reynolds number: " << reynolds_number << std::endl;
            std::cout << "  Prandtl number: " << prandtl_number << std::endl;
            std::cout << "  Dynamic viscosity: " << mu_reference << std::endl;
            std::cout << "  cv_inverse_kappa: " << cv_inverse_kappa_reference << std::endl;
        }
    }
};

#endif