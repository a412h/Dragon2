#ifndef INITIAL_CONDITION_H
#define INITIAL_CONDITION_H

#include <vector>
#include <array>
#include <random>
#include <cmath>

class InitialCondition {
public:
    using State = std::array<double, 4>;
    
    static void set_uniform(std::vector<State>& U, 
                           const std::array<double, 3>& primitive) {
        // Primitive variables: [rho, u, p]
        const double rho = primitive[0];  // density = 1.4
        const double u = primitive[1];    // x-velocity = 3.0 (Mach 3)
        const double v = 0.0;              // y-velocity = 0
        const double p = primitive[2];     // pressure = 1.0
        const double gamma = 1.4;
        
        // Convert to conservative variables [rho, rho*u, rho*v, E]
        const double E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);
        
        State conservative = {{rho, rho*u, rho*v, E}};
        
        // Verify the state
        std::cout << "Initial condition: uniform flow" << std::endl;
        std::cout << "  Primitive: [rho=" << rho << ", u=" << u << ", p=" << p << "]" << std::endl;
        std::cout << "  Conservative: [" << conservative[0] << ", " 
                  << conservative[1] << ", " << conservative[2] << ", " 
                  << conservative[3] << "]" << std::endl;
        
        // Check Mach number
        const double c = std::sqrt(gamma * p / rho);
        const double mach = u / c;
        std::cout << "  Speed of sound: " << c << std::endl;
        std::cout << "  Mach number: " << mach << std::endl;
        
        // Set all nodes to this state
        for (auto& state : U) {
            state = conservative;
        }
    }
};

#endif