

#ifndef BECKER_SOLUTION_H
#define BECKER_SOLUTION_H

#include <cmath>
#include <tuple>
#include <functional>
#include <array>



struct BeckerParams {
    double gamma;
    double velocity_left;
    double velocity_right;
    double density_left;
    double velocity_galilean;

    double position;

    double velocity_origin;
    double factor;
    double x_left_cutoff;
    double x_right_cutoff;
    double tol_norm;
    double c_l;
    double c_r;
    double log_origin_left;
    double log_origin_right;
};

#ifdef __CUDACC__



__device__ inline void becker_psi_device(
    const BeckerParams& p, double x, double v,
    double& value, double& derivative)
{
    const double log_l = log(p.velocity_left - v) - p.log_origin_left;
    const double log_r = log(v - p.velocity_right) - p.log_origin_right;

    value = p.factor * (p.c_l * log_l - p.c_r * log_r) - x;
    derivative = p.factor * (-p.c_l / (p.velocity_left - v) -
                              p.c_r / (v - p.velocity_right));
}


__device__ inline double becker_find_velocity_device(
    const BeckerParams& p, double x, double v_initial = 0.0)
{
    if (x <= p.x_left_cutoff)
        return p.velocity_left;
    if (x >= p.x_right_cutoff)
        return p.velocity_right;

    double v;
    if (v_initial > p.velocity_right && v_initial < p.velocity_left) {
        v = v_initial;
    } else {

        const double center = 0.5 * (p.x_right_cutoff + p.x_left_cutoff);
        const double width = p.x_right_cutoff - p.x_left_cutoff;
        const double nu = 0.5 * tanh(10.0 * (x - center) / width);
        v = p.velocity_left * (0.5 - nu) + p.velocity_right * (nu + 0.5);
    }

    double f, df;
    becker_psi_device(p, x, v, f, df);

    for (int iter = 0; iter < 100; ++iter) {
        if (fabs(f) <= p.tol_norm) break;

        double v_next = v - f / df;

        if (fabs(v_next - v) <
            1.0e-12 * 0.5 * (p.velocity_right + p.velocity_left)) {
            v = v_next;
            break;
        }

        if (v_next < p.velocity_right)
            v = 0.5 * (p.velocity_right + v);
        else if (v_next > p.velocity_left)
            v = 0.5 * (p.velocity_left + v);
        else
            v = v_next;

        becker_psi_device(p, x, v, f, df);
    }

    return v;
}


template<int dim, typename Number>
__device__ inline void becker_compute_state_device(
    const BeckerParams& p,
    double x_position, double t,
    Number& rho_out, Number& mx_out, Number& my_out, Number& energy_out,
    double v_warm = 0.0, double* v_out = nullptr)
{
    const double R_infty = (p.gamma + 1.0) / (p.gamma - 1.0);

    const double x = x_position - p.position - p.velocity_galilean * t;
    const double v = becker_find_velocity_device(p, x, v_warm);
    if (v_out != nullptr) *v_out = v;
    const double rho = p.density_left * p.velocity_left / v;
    const double e = 1.0 / (2.0 * p.gamma) *
                     (R_infty * p.velocity_left * p.velocity_right - v * v);
    const double vel_x = p.velocity_galilean + v;

    rho_out = Number(rho);
    mx_out = Number(rho * vel_x);
    my_out = Number(0);
    energy_out = Number(rho * (e + 0.5 * vel_x * vel_x));
}


template<int dim, typename Number>
__global__ void apply_becker_dirichlet_kernel(
    State<dim, Number> U,
    const int* bc_type,
    const Number* dof_positions_x,
    BeckerParams becker_params,
    Number current_time,
    int n_dofs,
    double* v_cache = nullptr)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_dofs) return;

    const int bid = bc_type[idx];
    if (bid != 4) return;  

    Number rho, mx, my, energy;
    double v_warm = (v_cache != nullptr) ? v_cache[idx] : 0.0;
    double v_final;
    becker_compute_state_device<dim, Number>(
        becker_params,
        double(dof_positions_x[idx]),
        double(current_time),
        rho, mx, my, energy,
        v_warm, &v_final);
    if (v_cache != nullptr) v_cache[idx] = v_final;

    U.rho[idx] = rho;
    U.momentum_x[idx] = mx;
    if constexpr (dim >= 2) U.momentum_y[idx] = my;
    if constexpr (dim == 3) U.momentum_z[idx] = Number(0);
    U.energy[idx] = energy;
}


template<int dim, typename Number>
__global__ void apply_becker_dirichlet_entry_kernel(
    State<dim, Number> U,
    const int* boundary_dofs,
    const int* boundary_ids,
    const Number* dof_positions_x,
    BeckerParams becker_params,
    Number current_time,
    int n_boundary_dofs,
    double* v_cache = nullptr)
{
    const int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if (eid >= n_boundary_dofs) return;
    if (boundary_ids[eid] != 4) return;  

    const int idx = boundary_dofs[eid];

    Number rho, mx, my, energy;
    double v_warm = (v_cache != nullptr) ? v_cache[idx] : 0.0;
    double v_final;
    becker_compute_state_device<dim, Number>(
        becker_params,
        double(dof_positions_x[idx]),
        double(current_time),
        rho, mx, my, energy,
        v_warm, &v_final);
    if (v_cache != nullptr) v_cache[idx] = v_final;

    U.rho[idx] = rho;
    U.momentum_x[idx] = mx;
    if constexpr (dim >= 2) U.momentum_y[idx] = my;
    if constexpr (dim == 3) U.momentum_z[idx] = Number(0);
    U.energy[idx] = energy;
}


template<int dim, typename Number>
__global__ void refresh_precomputed_at_dirichlet_kernel(
    const State<dim, Number> U,
    Number* d_pressure,
    Number* d_speed_of_sound,
    Number* d_precomputed,
    const int* bc_type,
    int n_dofs)
{
    using PF = PhysicsFunctions<dim, Number>;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_dofs) return;
    if (bc_type[idx] != 4) return;  

    d_precomputed[idx * 2 + 0] = PF::specific_entropy(U, idx);
    d_precomputed[idx * 2 + 1] = PF::harten_entropy(U, idx);

    d_pressure[idx] = PF::pressure(U, idx);
    const Number rho = U.rho[idx];
    d_speed_of_sound[idx] = sqrt(PF::gamma * d_pressure[idx] / rho);
}

#endif 




class BeckerSolutionCPU {
public:
    double gamma_;
    double velocity_left_;
    double velocity_right_;
    double density_left_;
    double velocity_galilean_;
    double mu_;
    double position_ = 0.0;  

    double velocity_origin_;
    double factor_;
    double x_left_cutoff_;
    double x_right_cutoff_;
    double tol_norm_;

    BeckerSolutionCPU() = default;

    BeckerSolutionCPU(double gamma, double velocity_left, double velocity_right,
                      double density_left, double velocity_galilean, double mu,
                      double position = 0.0)
        : gamma_(gamma), velocity_left_(velocity_left),
          velocity_right_(velocity_right), density_left_(density_left),
          velocity_galilean_(velocity_galilean), mu_(mu), position_(position)
    {
        velocity_origin_ = std::sqrt(velocity_left_ * velocity_right_);

        const double Pr = 0.75;
        factor_ = 2.0 * gamma_ / (gamma_ + 1.0) *
                  mu_ / (density_left_ * velocity_left_ * Pr);

        constexpr double tol = 1.0e-12;

        auto psi_val = [&](double x, double v) -> double {
            const double c_l = velocity_left_ / (velocity_left_ - velocity_right_);
            const double c_r = velocity_right_ / (velocity_left_ - velocity_right_);
            const double log_l = std::log(velocity_left_ - v) -
                                 std::log(velocity_left_ - velocity_origin_);
            const double log_r = std::log(v - velocity_right_) -
                                 std::log(velocity_origin_ - velocity_right_);
            return factor_ * (c_l * log_l - c_r * log_r) - x;
        };

        x_left_cutoff_ = psi_val(0.0,
            (1.0 - tol) * velocity_left_ + tol * velocity_right_);
        x_right_cutoff_ = psi_val(0.0,
            tol * velocity_left_ + (1.0 - tol) * velocity_right_);
        tol_norm_ = (x_right_cutoff_ - x_left_cutoff_) * tol;
    }

    std::pair<double, double> psi(double x, double v) const {
        const double c_l = velocity_left_ / (velocity_left_ - velocity_right_);
        const double c_r = velocity_right_ / (velocity_left_ - velocity_right_);
        const double log_l = std::log(velocity_left_ - v) -
                             std::log(velocity_left_ - velocity_origin_);
        const double log_r = std::log(v - velocity_right_) -
                             std::log(velocity_origin_ - velocity_right_);

        double value = factor_ * (c_l * log_l - c_r * log_r) - x;
        double derivative = factor_ * (-c_l / (velocity_left_ - v) -
                                        c_r / (v - velocity_right_));
        return {value, derivative};
    }

    double find_velocity(double x) const {
        if (x <= x_left_cutoff_) return velocity_left_;
        if (x >= x_right_cutoff_) return velocity_right_;

        constexpr double tol = 1.0e-12;

        const double center = 0.5 * (x_right_cutoff_ + x_left_cutoff_);
        const double width = x_right_cutoff_ - x_left_cutoff_;
        const double nu = 0.5 * std::tanh(10.0 * (x - center) / width);
        double v = velocity_left_ * (0.5 - nu) + velocity_right_ * (nu + 0.5);

        auto [f, df] = psi(x, v);

        for (int iter = 0; iter < 200; ++iter) {
            if (std::abs(f) <= tol_norm_) break;

            double v_next = v - f / df;

            if (std::abs(v_next - v) <
                tol * 0.5 * (velocity_right_ + velocity_left_)) {
                v = v_next;
                break;
            }

            if (v_next < velocity_right_)
                v = 0.5 * (velocity_right_ + v);
            else if (v_next > velocity_left_)
                v = 0.5 * (velocity_left_ + v);
            else
                v = v_next;

            auto [new_f, new_df] = psi(x, v);
            f = new_f;
            df = new_df;
        }

        return v;
    }

    std::array<double, 4> compute(double x_position, double t) const {
        const double R_infty = (gamma_ + 1.0) / (gamma_ - 1.0);

        const double x = x_position - position_ - velocity_galilean_ * t;
        const double v = find_velocity(x);
        const double rho = density_left_ * velocity_left_ / v;
        const double e = 1.0 / (2.0 * gamma_) *
                         (R_infty * velocity_left_ * velocity_right_ - v * v);
        const double vel_x = velocity_galilean_ + v;

        return {rho,
                rho * vel_x,
                0.0,
                rho * (e + 0.5 * vel_x * vel_x)};
    }

    BeckerParams make_gpu_params() const {
        BeckerParams p;
        p.gamma = gamma_;
        p.velocity_left = velocity_left_;
        p.velocity_right = velocity_right_;
        p.density_left = density_left_;
        p.velocity_galilean = velocity_galilean_;
        p.position = position_;
        p.velocity_origin = velocity_origin_;
        p.factor = factor_;
        p.x_left_cutoff = x_left_cutoff_;
        p.x_right_cutoff = x_right_cutoff_;
        p.tol_norm = tol_norm_;
        p.c_l = velocity_left_ / (velocity_left_ - velocity_right_);
        p.c_r = velocity_right_ / (velocity_left_ - velocity_right_);
        p.log_origin_left = std::log(velocity_left_ - velocity_origin_);
        p.log_origin_right = std::log(velocity_origin_ - velocity_right_);
        return p;
    }
};

#endif 
