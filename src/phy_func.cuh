#ifndef PHY_FUNC_CUH
#define PHY_FUNC_CUH

#include <cuda_runtime.h>
#include "data_struct.cuh"

// Device physics functions
template <int dim, typename Number = float>
struct PhysicsFunctions {
    static constexpr int problem_dimension = dim + 2;
    using StateType = State<dim, Number>;
    using FluxType = Flux<dim, Number>;
    using MomentumType = Momentum<dim, Number>;
    
    static constexpr Number gamma = Number(1.4);
    static constexpr Number gamma_plus_one = Number(2.4);
    static constexpr Number gamma_minus_one = Number(0.4);
    
    __device__ static Number gamma_plus_one_inverse() {
        return Number(1.0) / gamma_plus_one;
    }
    
    __device__ static Number norm_square_dim(const Number* x) {
        Number norm_sq = Number(0.);
        for (int i = 0; i < dim; ++i)
            norm_sq += x[i] * x[i];
        return norm_sq;
    }

    __device__ static Number norm_dim(const Number* x) {
        Number norm = Number(0.);
        for (int i = 0; i < dim; ++i)
            norm += x[i] * x[i];
        return sqrt(norm);
    }
    
    __device__ static Number density(const StateType& U, int i) {
        return U.rho[i];
    }
    
    __device__ static MomentumType momentum(const StateType& U, int i) {
        MomentumType result;
        result[0] = U.momentum_x[i];
        if constexpr (dim >= 2) result[1] = U.momentum_y[i];
        if constexpr (dim == 3) result[2] = U.momentum_z[i];
        return result;
    }
    
    __device__ static Number total_energy(const StateType& U, int i) {
        return U.energy[i];
    }
    
    __device__ static Number internal_energy(const StateType& U, int i) {
        const Number rho = U.rho[i];
        const Number rho_inverse = Number(1.0) / rho;
        Number m_sq = U.momentum_x[i] * U.momentum_x[i];
        if constexpr (dim >= 2) m_sq += U.momentum_y[i] * U.momentum_y[i];
        if constexpr (dim == 3) m_sq += U.momentum_z[i] * U.momentum_z[i];
        return U.energy[i] - Number(0.5) * m_sq * rho_inverse;
    }
    
    __device__ static Number pressure(const StateType& U, int i) {
        return gamma_minus_one * internal_energy(U, i);
    }
    
    __device__ static Number speed_of_sound(const StateType& U, int i) {
        const Number rho_inverse = Number(1.) / U.rho[i];
        const Number p = pressure(U, i);
        return sqrt(gamma * p * rho_inverse);
    }
    
    __device__ static Number specific_entropy(const StateType& U, int i) {
        const Number rho = U.rho[i];
        const Number rho_inverse = Number(1.0) / rho;
        const Number rho_e = internal_energy(U, i);
        return rho_e * pow(rho_inverse, gamma);
    }
    
    __device__ static Number harten_entropy(const StateType& U, int i) {
        const Number rho = U.rho[i];
        const Number E = U.energy[i];
        Number m_sq = U.momentum_x[i] * U.momentum_x[i];
        if constexpr (dim >= 2) m_sq += U.momentum_y[i] * U.momentum_y[i];
        if constexpr (dim == 3) m_sq += U.momentum_z[i] * U.momentum_z[i];
        const Number rho_rho_e = rho * E - Number(0.5) * m_sq;
        return pow(rho_rho_e, gamma_plus_one_inverse());
    }
    
    // Local array computations
    __device__ static Number density_local(const Number U_local[dim + 2]) {
        return U_local[0];
    }
    
    __device__ static MomentumType momentum_local(const Number U_local[dim + 2]) {
        MomentumType result;
        for (int i = 0; i < dim; ++i)
            result[i] = U_local[i + 1];
        return result;
    }
    
    __device__ static Number total_energy_local(const Number U_local[dim + 2]) {
        return U_local[dim + 1];
    }
    
    __device__ static Number internal_energy_local(const Number U_local[dim + 2]) {
        const Number rho_inverse = Number(1.0) / U_local[0];
        Number m_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            m_sq += U_local[1+d] * U_local[1+d];
        }
        return U_local[dim+1] - Number(0.5) * m_sq * rho_inverse;
    }
    
    __device__ static Number pressure_local(const Number U_local[dim + 2]) {
        return gamma_minus_one * internal_energy_local(U_local);
    }
    
    __device__ static Number specific_entropy_local(const Number U_local[dim + 2]) {
        const Number rho = U_local[0];
        const Number rho_inverse = Number(1.0) / rho;
        const Number rho_e = internal_energy_local(U_local);
        return rho_e * pow(rho_inverse, gamma);
    }
    
    __device__ static Number harten_entropy_local(const Number U_local[dim + 2]) {
        const Number rho = U_local[0];
        const Number E = U_local[dim+1];
        Number m_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            m_sq += U_local[1+d] * U_local[1+d];
        }
        const Number rho_rho_e = rho * E - Number(0.5) * m_sq;
        return pow(rho_rho_e, gamma_plus_one_inverse());
    }
    
    __device__ static void harten_entropy_derivative_local(const Number U_local[dim + 2], Number result[dim + 2]) {
        const Number rho = U_local[0];
        const Number E = U_local[dim+1];
        Number m_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            m_sq += U_local[1+d] * U_local[1+d];
        }
        const Number rho_rho_e = rho * E - Number(0.5) * m_sq;
        const auto factor = gamma_plus_one_inverse() * pow(rho_rho_e, -gamma * gamma_plus_one_inverse());
        
        result[0] = factor * E;
        for (int i = 0; i < dim; ++i)
            result[1 + i] = -factor * U_local[1 + i];
        result[dim + 1] = factor * rho;
    }
    
    __device__ static void internal_energy_derivative_local(const Number U_local[dim + 2], Number result[dim + 2]) {
        const Number rho_inverse = Number(1.) / U_local[0];
        Number u[dim];
        for (int i = 0; i < dim; ++i)
            u[i] = U_local[1 + i] * rho_inverse;
        
        result[0] = Number(0.5) * norm_square_dim(u);
        for (int i = 0; i < dim; ++i)
            result[1 + i] = -u[i];
        result[dim + 1] = Number(1.);
    }
    
    __device__ static FluxType f_local(const Number U_local[dim + 2]) {
        const auto rho_inverse = Number(1.) / U_local[0];
        const auto p = pressure_local(U_local);
        const auto E = U_local[dim+1];
        
        FluxType result;
        
        for (int i = 0; i < dim; ++i)
            result(0, i) = U_local[1 + i];
        
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j)
                result(1 + i, j) = U_local[1+j] * (U_local[1 + i] * rho_inverse);
            result(1 + i, i) += p;
        }
        
        for (int i = 0; i < dim; ++i)
            result(dim + 1, i) = U_local[1 + i] * (rho_inverse * (E + p));
        
        return result;
    }
    
    __device__ static Number vacuum_state_relaxation_small() {
        return Number(1.e2);
    }
    
    __device__ static Number vacuum_state_relaxation_large() {
        return Number(1.e4);
    }
    
    __device__ static Number reference_density() {
        return Number(1.);
    }
    
    __device__ static Number filter_vacuum_density(const Number& rho) {
       constexpr Number eps = (sizeof(Number) == sizeof(float)) ? Number(1e-7f) : Number(1e-14);
       const Number rho_cutoff_large = reference_density() * vacuum_state_relaxation_large() * eps;
       return (fabs(rho) < rho_cutoff_large) ? Number(0.0) : rho;
    }
};

// Type aliases
using PhysicsFunctions2D = PhysicsFunctions<2, double>;
using PhysicsFunctions3D = PhysicsFunctions<3, double>;
using PhysicsFunctions2Df = PhysicsFunctions<2, float>;
using PhysicsFunctions3Df = PhysicsFunctions<3, float>;

#endif // PHY_FUNC_CUH