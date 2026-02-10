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

    std::string mesh_file_path = "";
    std::string boundary_mapping = "";
    std::string default_boundary_condition = "do_nothing";
    std::array<double, 3> primitive_state = {{1.4, 3.0, 1.0}};
    std::array<double, 3> direction = {{1.0, 0.0, 0.0}};

    double cfl_min = 0.9;
    double cfl_max = 0.9;
    double cfl_number = 0.9;
    std::string time_scheme = "erk33";

    bool is_navier_stokes() const {
        return (equation == "navier stokes" || equation == "navier_stokes");
    }

    void read_parameters(const std::string& filename) {
        read_parameters_impl(filename);
    }

private:
    void read_parameters_impl(const std::string& filename) {
        std::ifstream file_check(filename);
        if (!file_check.good()) {
            throw std::runtime_error("Parameter file not found: " + filename);
        }
        file_check.close();

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
            prm.enter_subsection("mesh_file");
            {
                prm.declare_entry("file path", "", dealii::Patterns::Anything(),
                    "Path to the .msh mesh file (relative or absolute)");
                prm.declare_entry("boundary mapping", "", dealii::Patterns::Anything(),
                    "Boundary condition mapping: 'phys_id1:bc_type1, phys_id2:bc_type2, ...'\n"
                    "Available BC types: do_nothing, periodic, slip, no_slip, dirichlet, dynamic");
                prm.declare_entry("default boundary condition", "do_nothing", dealii::Patterns::Anything(),
                    "Default BC for physical IDs not in the mapping");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();

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

        prm.enter_subsection("H - TimeIntegrator");
        {
            prm.declare_entry("cfl min", "0.90", dealii::Patterns::Double());
            prm.declare_entry("cfl max", "0.90", dealii::Patterns::Double());
            prm.declare_entry("time scheme", "erk33", dealii::Patterns::Anything());
        }
        prm.leave_subsection();

        std::string ext = filename.substr(filename.find_last_of(".") + 1);
        if (ext == "cfg") {
            std::ifstream file(filename);
            std::stringstream buffer;
            buffer << file.rdbuf();
            prm.parse_input_from_string(buffer.str(), filename);
        } else {
            prm.parse_input(filename);
        }

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
            prm.enter_subsection("mesh_file");
            {
                mesh_file_path = prm.get("file path");
                boundary_mapping = prm.get("boundary mapping");
                default_boundary_condition = prm.get("default boundary condition");
            }
            prm.leave_subsection();
        }
        prm.leave_subsection();

        prm.enter_subsection("E - InitialValues");
        {
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
        }
        prm.leave_subsection();

        prm.enter_subsection("H - TimeIntegrator");
        {
            cfl_min = prm.get_double("cfl min");
            cfl_max = prm.get_double("cfl max");
            cfl_number = cfl_min;
            time_scheme = prm.get("time scheme");
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
        std::cout << "  Equation: " << equation << std::endl;
        std::cout << "  Final time: " << final_time << std::endl;
        std::cout << "  CFL: " << cfl_number << " (min: " << cfl_min << ", max: " << cfl_max << ")" << std::endl;
        std::cout << "  Time scheme: " << time_scheme << std::endl;

        std::cout << "\n  Physics parameters:" << std::endl;
        std::cout << "    gamma: " << gamma << std::endl;
        if (is_navier_stokes()) {
            std::cout << "    mu (dynamic viscosity): " << mu << std::endl;
            std::cout << "    lambda (bulk viscosity): " << lambda << std::endl;
            std::cout << "    kappa (thermal conductivity): " << kappa << std::endl;
        }

        std::cout << "\n  External mesh file:" << std::endl;
        std::cout << "    File path: " << mesh_file_path << std::endl;
        std::cout << "    Boundary mapping: " << (boundary_mapping.empty() ? "(none)" : boundary_mapping) << std::endl;
        std::cout << "    Default BC: " << default_boundary_condition << std::endl;

        std::cout << "\n  Initial state (primitive):" << std::endl;
        std::cout << "    rho: " << primitive_state[0] << std::endl;
        std::cout << "    u: " << primitive_state[1] << std::endl;
        std::cout << "    p: " << primitive_state[2] << std::endl;
        std::cout << "    direction: [" << direction[0] << ", " << direction[1] << ", " << direction[2] << "]" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

#endif
