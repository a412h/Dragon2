#ifndef INITIAL_CONDITION_H
#define INITIAL_CONDITION_H

#include <iostream>
#include <vector>
#include <array>
#include <cmath>


class InitialCondition {
public:
    using State = std::array<double, 4>;

    static void set_uniform(std::vector<State>& U,
                           const std::array<double, 3>& primitive,
                           const std::array<double, 2>& direction = {{1.0, 0.0}},
                           const double gamma = 1.4) {
        const double rho = primitive[0];
        const double vel_mag = primitive[1];
        const double p = primitive[2];

        const double dir_norm = std::sqrt(direction[0]*direction[0] +
                                         direction[1]*direction[1]);
        const double u = vel_mag * direction[0] / dir_norm;
        const double v = vel_mag * direction[1] / dir_norm;

        const double E = p / (gamma - 1.0) + 0.5 * rho * (u * u + v * v);

        State conservative = {{rho, rho*u, rho*v, E}};

        std::cout << "\nInitial condition: uniform flow" << std::endl;
        std::cout << "  Primitive: [rho=" << rho << ", |v|=" << vel_mag
                  << ", p=" << p << "]" << std::endl;
        std::cout << "  Velocity: [u=" << u << ", v=" << v << "]" << std::endl;

        const double angle_deg = std::atan2(direction[1], direction[0]) * 180.0 / M_PI;
        if (std::abs(angle_deg) > 0.01) {
            std::cout << "  Angle of attack: " << angle_deg << " deg" << std::endl;
        }

        std::cout << "  Conservative: [" << conservative[0] << ", "
                  << conservative[1] << ", " << conservative[2] << ", "
                  << conservative[3] << "]" << std::endl;

        const double c = std::sqrt(gamma * p / rho);
        const double mach = vel_mag / c;
        std::cout << "  Speed of sound: " << c << " m/s" << std::endl;
        std::cout << "  Mach number: " << mach << std::endl;

        for (auto& state : U) {
            state = conservative;
        }
    }
};

#endif