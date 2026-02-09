#ifndef LIMITER_CUH
#define LIMITER_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"

template<int dim, typename Number>
struct Limiter {
    static constexpr Number gamma = Number(1.4);
    static constexpr Number gamma_plus_one = Number(2.4);
    static constexpr Number gamma_minus_one = Number(0.4);

    static constexpr Number newton_tolerance = Number(1e-10);
    static constexpr int newton_max_iterations = 2;
    static constexpr Number relaxation_factor = Number(1.0);

    using PF = PhysicsFunctions<dim, Number>;

    struct Bounds {
        Number rho_min;
        Number rho_max;
        Number s_min;
    };

    Number U_i[dim + 2];
    Number p_i;
    Bounds bounds_;
    Number rho_relaxation_numerator;
    Number rho_relaxation_denominator;
    Number s_interp_max;

    __device__ __forceinline__
    void reset(const Number U_i_state[dim + 2], Number pressure_i, Number specific_entropy_i) {
        #pragma unroll
        for (int k = 0; k < dim + 2; ++k) {
            U_i[k] = U_i_state[k];
        }
        p_i = pressure_i;

        bounds_.rho_min = U_i[0];
        bounds_.rho_max = U_i[0];
        bounds_.s_min = specific_entropy_i;
        s_interp_max = specific_entropy_i;

        rho_relaxation_numerator = Number(0);
        rho_relaxation_denominator = Number(0);
    }

    __device__ __forceinline__
    void accumulate(const Number U_j[dim + 2], const Number* scaled_c_ij, Number pressure_j) {
        const Number rho_i = U_i[0];
        const Number rho_j = U_j[0];

        Number momentum_term = Number(0);
        for (int d = 0; d < dim; ++d) {
            const Number m_i_d = U_i[1 + d];
            const Number m_j_d = U_j[1 + d];
            momentum_term += (m_i_d - m_j_d) * scaled_c_ij[d];
        }
        const Number rho_ij_bar = Number(0.5) * (rho_i + rho_j + momentum_term);

        bounds_.rho_min = min(bounds_.rho_min, rho_ij_bar);
        bounds_.rho_max = max(bounds_.rho_max, rho_ij_bar);

        const Number s_j = PF::specific_entropy_local(U_j);
        bounds_.s_min = min(bounds_.s_min, s_j);

        const Number beta_ij = Number(1);
        rho_relaxation_numerator += beta_ij * (rho_i + rho_j);
        rho_relaxation_denominator += abs(beta_ij);

        Number U_avg[dim + 2];
        for (int k = 0; k < dim + 2; ++k) {
            U_avg[k] = (U_i[k] + U_j[k]) * Number(0.5);
        }
        const Number s_interp = PF::specific_entropy_local(U_avg);
        s_interp_max = max(s_interp_max, s_interp);
    }

    __device__ __forceinline__
    Bounds fully_relax_bounds(const Bounds& bounds, Number hd_i) {
        Bounds relaxed = bounds;

        Number r = sqrt(hd_i);
        r = sqrt(r);
        r = r * r * r;
        r *= relaxation_factor;

        const Number eps = Number(1e-14);

        relaxed.rho_min *= max(Number(1) - r, eps);
        relaxed.rho_max *= (Number(1) + r);
        relaxed.s_min *= max(Number(1) - r, eps);

        return relaxed;
    }

    __device__ __forceinline__
    Bounds get_bounds(Number hd_i) {
        Bounds relaxed = fully_relax_bounds(bounds_, hd_i);

        const Number eps = Number(1e-14);

        if (abs(rho_relaxation_denominator) > eps) {
            const Number rho_relaxation =
                Number(2) * relaxation_factor *
                abs(rho_relaxation_numerator) /
                (abs(rho_relaxation_denominator) + eps);

            relaxed.rho_min = max(relaxed.rho_min, bounds_.rho_min - rho_relaxation);
            relaxed.rho_max = min(relaxed.rho_max, bounds_.rho_max + rho_relaxation);
        }

        const Number entropy_relaxation =
            relaxation_factor * (s_interp_max - bounds_.s_min);
        relaxed.s_min = max(relaxed.s_min, bounds_.s_min - entropy_relaxation);

        return relaxed;
    }

    __device__ __forceinline__
    static void quadratic_newton_step(
        Number& p_1, Number& p_2,
        const Number phi_p_1, const Number phi_p_2,
        const Number dphi_p_1, const Number dphi_p_2,
        const Number sign = Number(1))
    {
        constexpr Number eps = Number(1e-14);

        const Number scaling = Number(1) / (p_2 - p_1 + eps);
        const Number dd_11 = dphi_p_1;
        const Number dd_12 = (phi_p_2 - phi_p_1) * scaling;
        const Number dd_22 = dphi_p_2;
        const Number dd_112 = (dd_12 - dd_11) * scaling;
        const Number dd_122 = (dd_22 - dd_12) * scaling;

        const Number discriminant_1 =
            abs(dphi_p_1 * dphi_p_1 - Number(4) * phi_p_1 * dd_112);
        const Number discriminant_2 =
            abs(dphi_p_2 * dphi_p_2 - Number(4) * phi_p_2 * dd_122);

        const Number denominator_1 = dphi_p_1 + sign * sqrt(discriminant_1);
        const Number denominator_2 = dphi_p_2 + sign * sqrt(discriminant_2);

        Number t_1 = (abs(denominator_1) < eps) ? p_1 :
                     p_1 - Number(2) * phi_p_1 / denominator_1;

        Number t_2 = (abs(denominator_2) < eps) ? p_2 :
                     p_2 - Number(2) * phi_p_2 / denominator_2;

        t_1 = max(p_1, t_1);
        t_1 = min(p_2, t_1);
        t_2 = max(p_1, t_2);
        t_2 = min(p_2, t_2);

        p_1 = min(t_1, t_2);
        p_2 = max(t_1, t_2);
    }

    __device__ __forceinline__
    static Number limit(
        const Bounds& bounds,
        const Number U[dim + 2],
        const Number P[dim + 2],
        const Number t_min,
        const Number t_max)
    {
        Number t_r = t_max;

        const Number eps = Number(1e-14);
        const Number small = PF::vacuum_state_relaxation_small();
        const Number large = PF::vacuum_state_relaxation_large();
        const Number relax_small = Number(1) + small * eps;
        const Number relax = Number(1) + large * eps;

        const Number rho_U = U[0];
        const Number rho_P = P[0];
        const Number rho_min = bounds.rho_min;
        const Number rho_max = bounds.rho_max;

        const Number test_min = PF::filter_vacuum_density(max(Number(0), rho_U - relax * rho_max));
        const Number test_max = PF::filter_vacuum_density(max(Number(0), rho_min - relax * rho_U));

        const Number denominator = Number(1) / (abs(rho_P) + eps * rho_max);
        if (rho_max < rho_U + t_r * rho_P)
            t_r = (rho_max - rho_U) * denominator;
        if (rho_U + t_r * rho_P < rho_min)
            t_r = (rho_U - rho_min) * denominator;
        t_r = min(t_r, t_max);
        t_r = max(t_r, t_min);

        Number t_l = t_min;
        const Number gp1 = gamma_plus_one;
        const Number s_min = bounds.s_min;

        #pragma unroll
        for (int n = 0; n < newton_max_iterations; ++n) {

            Number U_r[dim + 2];
            #pragma unroll
            for (int k = 0; k < dim + 2; ++k) {
                U_r[k] = U[k] + t_r * P[k];
            }

            const Number rho_r = U_r[0];
            const Number rho_r_gamma = __powf(rho_r, gamma);
            const Number rho_e_r = PF::internal_energy_local(U_r);
            const Number psi_r = relax_small * rho_r * rho_e_r - s_min * rho_r * rho_r_gamma;

            if (psi_r > Number(0)) {
                t_l = t_r;
                break;
            }

            Number U_l[dim + 2];
            #pragma unroll
            for (int k = 0; k < dim + 2; ++k) {
                U_l[k] = U[k] + t_l * P[k];
            }

            const Number rho_l = PF::density_local(U_l);
            const Number rho_l_gamma = __powf(rho_l, gamma);
            const Number rho_e_l = PF::internal_energy_local(U_l);
            const Number psi_l = relax_small * rho_l * rho_e_l - s_min * rho_l * rho_l_gamma;

            if (max(Number(0), t_r - t_l - newton_tolerance) == Number(0))
                break;

            const Number drho = PF::density_local(P);
            Number ied_U_l[dim + 2];
            Number ied_U_r[dim + 2];
            PF::internal_energy_derivative_local(U_l, ied_U_l);
            PF::internal_energy_derivative_local(U_r, ied_U_r);

            Number drho_e_l = Number(0);
            Number drho_e_r = Number(0);
            #pragma unroll
            for (int k = 0; k < dim + 2; ++k) {
                drho_e_l += ied_U_l[k] * P[k];
                drho_e_r += ied_U_r[k] * P[k];
            }

            const Number dpsi_l = rho_l * drho_e_l + (rho_e_l - gp1 * s_min * rho_l_gamma) * drho;
            const Number dpsi_r = rho_r * drho_e_r + (rho_e_r - gp1 * s_min * rho_r_gamma) * drho;

            quadratic_newton_step(t_l, t_r, psi_l, psi_r, dpsi_l, dpsi_r, Number(-1));
        }

        return t_l;
    }
};

#endif