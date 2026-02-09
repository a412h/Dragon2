#ifndef RIEMANN_SOLVER_CUH
#define RIEMANN_SOLVER_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"
#include "phy_func.cuh"

template<int dim, typename Number>
struct RiemannSolver {
    static constexpr Number gamma = Number(1.4);
    static constexpr Number gamma_plus_one = gamma + 1.;
    static constexpr Number gamma_minus_one = gamma - 1.;
    static constexpr Number gamma_inverse = 1. / gamma;
    static constexpr Number gamma_plus_one_inverse = 1. / gamma_plus_one;
    static constexpr Number gamma_minus_one_inverse = 1. / gamma_minus_one;

    using PF = PhysicsFunctions<dim, Number>;
    using PT = PrimitiveType<Number>;

    __device__
    static Number p_star_two_rarefaction(
        const PT& primitive_type_i,
        const PT& primitive_type_j)
    {
        const auto& riemann_data_i = primitive_type_i;
        const auto& riemann_data_j = primitive_type_j;
        const Number rho_i = riemann_data_i.data[0];
        const Number u_i   = riemann_data_i.data[1];
        const Number p_i   = riemann_data_i.data[2];
        const Number a_i   = riemann_data_i.data[3];
        const Number rho_j = riemann_data_j.data[0];
        const Number u_j   = riemann_data_j.data[1];
        const Number p_j   = riemann_data_j.data[2];
        const Number a_j   = riemann_data_j.data[3];

        const auto inv_p_j = Number(1.) / p_j;
        const auto factor = (gamma - Number(1.)) * Number(0.5);
        const Number numerator = positive_part(a_i + a_j - factor * (u_j - u_i));
        const Number denominator = a_i * pow(p_i * inv_p_j, -factor * gamma_inverse) + a_j;
        const auto exponent = Number(2.) * gamma * gamma_minus_one_inverse;
        const auto p_1_tilde = p_j * pow(numerator / denominator, exponent);

        return p_1_tilde;
    }

    __device__
    static Number p_star_failsafe(
        const PT& primitive_type_i,
        const PT& primitive_type_j)
    {
        const Number rho_i = primitive_type_i.data[0];
        const Number u_i   = primitive_type_i.data[1];
        const Number p_i   = primitive_type_i.data[2];
        const Number rho_j = primitive_type_j.data[0];
        const Number u_j   = primitive_type_j.data[1];
        const Number p_j   = primitive_type_j.data[2];

        const Number p_max = fmax(p_i, p_j);

        Number radicand_i = Number(2.) * p_max;
        radicand_i /= rho_i * (gamma_plus_one * p_max + gamma_minus_one * p_i);

        const Number x_i = sqrt(radicand_i);

        Number radicand_j = Number(2.) * p_max;
        radicand_j /= rho_j * (gamma_plus_one * p_max + gamma_minus_one * p_j);

        const Number x_j = sqrt(radicand_j);

        const Number a = x_i + x_j;
        const Number b = u_j - u_i;
        const Number c = -p_i * x_i - p_j * x_j;

        const Number base = (-b + sqrt(b * b - Number(4.) * a * c)) / (Number(2.) * a);

        const Number p_2_tilde = base * base;

        return p_2_tilde;
    }

    __device__
    static Number phi_of_p_max(
        const PT& primitive_type_i,
        const PT& primitive_type_j)
    {
        const Number rho_i = primitive_type_i.data[0];
        const Number u_i   = primitive_type_i.data[1];
        const Number p_i   = primitive_type_i.data[2];
        const Number rho_j = primitive_type_j.data[0];
        const Number u_j   = primitive_type_j.data[1];
        const Number p_j   = primitive_type_j.data[2];

        const Number p_max = fmax(p_i, p_j);

        const Number radicand_inverse_i = Number(0.5) * rho_i *
            (gamma_plus_one * p_max + gamma_minus_one * p_i);

        const Number value_i = (p_max - p_i) / sqrt(radicand_inverse_i);

        const Number radicand_inverse_j = Number(0.5) * rho_j *
            (gamma_plus_one * p_max + gamma_minus_one * p_j);

        const Number value_j = (p_max - p_j) / sqrt(radicand_inverse_j);

        return value_i + value_j + u_j - u_i;
    }

    __device__
    static Number positive_part(Number x) {
        return fmax(x, Number(0));
    }

    __device__
    static Number negative_part(Number x) {
        return fmax(-x, Number(0));
    }

    __device__
    static Number lambda1_minus(
        const PT& primitive_type,
        const Number p_star)
    {
        const Number factor = gamma_plus_one * Number(0.5) * gamma_inverse;

        const Number rho = primitive_type.data[0];
        const Number u = primitive_type.data[1];
        const Number p = primitive_type.data[2];
        const Number a = primitive_type.data[3];
        const Number inv_p = Number(1.) / p;

        const Number tmp = positive_part((p_star - p) * inv_p);

        return u - a * sqrt(Number(1.) + factor * tmp);
    }

    __device__
    static Number lambda3_plus(
        const PT& primitive_type,
        const Number p_star)
    {
        const Number factor = gamma_plus_one * Number(0.5) * gamma_inverse;

        const Number rho = primitive_type.data[0];
        const Number u = primitive_type.data[1];
        const Number p = primitive_type.data[2];
        const Number a = primitive_type.data[3];
        const Number inv_p = Number(1.) / p;

        const Number tmp = positive_part((p_star - p) * inv_p);

        return u + a * sqrt(Number(1.) + factor * tmp);
    }

    __device__
    static Number compute_lambda(
        const PT& primitive_type_i,
        const PT& primitive_type_j,
        const Number p_star)
    {
        const Number nu_11 = lambda1_minus(primitive_type_i, p_star);
        const Number nu_32 = lambda3_plus(primitive_type_j, p_star);

        return fmax(positive_part(nu_32), negative_part(nu_11));
    }

    __device__
    static Number compute_(
        const PT& primitive_type_i,
        const PT& primitive_type_j)
    {
        const auto& riemann_data_i = primitive_type_i;
        const auto& riemann_data_j = primitive_type_j;
        const Number rho_i = riemann_data_i.data[0];
        const Number u_i   = riemann_data_i.data[1];
        const Number p_i   = riemann_data_i.data[2];
        const Number a_i   = riemann_data_i.data[3];
        const Number rho_j = riemann_data_j.data[0];
        const Number u_j   = riemann_data_j.data[1];
        const Number p_j   = riemann_data_j.data[2];
        const Number a_j   = riemann_data_j.data[3];

        const Number p_max = fmax(p_i, p_j);
        const Number rarefaction = p_star_two_rarefaction(primitive_type_i, primitive_type_j);
        const Number failsafe = p_star_failsafe(primitive_type_i, primitive_type_j);
        const Number p_star_tilde = fmin(rarefaction, failsafe);

        const Number phi_p_max = phi_of_p_max(primitive_type_i, primitive_type_j);

        Number p_2;
        if (phi_p_max < Number(0))
            p_2 = p_star_tilde;
        else
            p_2 = fmin(p_max, p_star_tilde);

        auto lambda_max = compute_lambda(primitive_type_i, primitive_type_j, p_2);

        return lambda_max;
    }

    __device__
    static PT riemann_data_from_state_local(
        const Number U_local[dim+2],
        const Number* n_ij)
    {
        const auto rho = U_local[0];
        const auto rho_inverse = Number(1.) / rho;

        Number proj_m = 0.;
        for (int k = 0; k < dim; ++k)
            proj_m += n_ij[k] * U_local[1+k];

        Number perp_sq = Number(0);
        for (int k = 0; k < dim; ++k) {
            Number perp_k = U_local[1+k] - proj_m * n_ij[k];
            perp_sq += perp_k * perp_k;
        }

        const auto E = U_local[dim+1] - Number(0.5) * perp_sq * rho_inverse;

        const auto p = gamma_minus_one * (E - Number(0.5) * proj_m * proj_m * rho_inverse);
        const auto a = sqrt(gamma * p * rho_inverse);

        PT result;
        result.data[0] = rho;
        result.data[1] = proj_m * rho_inverse;
        result.data[2] = p;
        result.data[3] = a;

        return result;
    }

    __device__
    static Number compute_local(
        const Number U_i[dim+2],
        const Number U_j[dim+2],
        const Number* n_ij)
    {
        const auto primitive_type_i = riemann_data_from_state_local(U_i, n_ij);
        const auto primitive_type_j = riemann_data_from_state_local(U_j, n_ij);

        return compute_(primitive_type_i, primitive_type_j);
    }
};

#endif