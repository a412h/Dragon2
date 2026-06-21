#ifndef INTERNAL_ENERGY_SOLVER_CUH
#define INTERNAL_ENERGY_SOLVER_CUH

#include <cuda_runtime.h>
#include <type_traits>
#include "mf_operators.cuh"   


template<int dim, typename Number>
__global__ void energy_operator_bc_kernel(
    const Number* energy_in,
    Number* energy_out,
    const int* bc_type,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    if (bc_type[i] == 4)  
        energy_out[i] = energy_in[i];
}


template<int dim, typename Number>
__global__ void compute_energy_diagonal_preconditioner_kernel(
    Number* diagonal_inv,
    const Number* density,
    const Number* lumped_mass,
    const int* cij_row_offsets,
    const Number* cij_values,
    const Number* mij_values,
    Number kappa,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    Number diag_value = lumped_mass[i] * density[i];

    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];
    Number laplacian_diag = Number(0);
    for (int idx = row_start; idx < row_end; ++idx) {
        const Number m_ij = mij_values[idx];
        Number c_ij_norm_sq = Number(0);
        for (int d = 0; d < dim; ++d) {
            const Number c_d = cij_values[idx * dim + d];
            c_ij_norm_sq += c_d * c_d;
        }
        if (m_ij > Number(1e-30))
            laplacian_diag += c_ij_norm_sq / m_ij;
    }
    diag_value += tau * kappa * laplacian_diag;

    diagonal_inv[i] = (diag_value > Number(1e-30)) ? Number(1) / diag_value : Number(0);
}


template<int dim, typename Number>
struct EnergyOperator {
    const Number* density = nullptr;
    const Number* lumped_mass = nullptr;
    const int* conn = nullptr;
    const Number* jac = nullptr;     
    const Number* Sv = nullptr;      
    const Number* Sg = nullptr;      
    int n_elements = 0;
    const int* bc_type = nullptr;
    Number kappa = 0, tau = 0;
    int n_dofs = 0;
    cudaStream_t stream = 0;

    int size() const { return n_dofs; }

    void vmult(Number* dst, const Number* src) const {
        const int t = 256;
        const int b_dofs = (n_dofs + t - 1) / t;
        const int b_elem = (n_elements + t - 1) / t;

        mfops::mf_mass_diagonal_kernel<Number, 1><<<b_dofs, t, 0, stream>>>(
            src, dst, lumped_mass, density, n_dofs);

        if (kappa != Number(0)) {
            mfops::LaplaceFlux<dim, Number> flux;
            mfops::mf_cell_gradient_kernel_sf<dim, Number, fe_degree, 1, mfops::LaplaceFlux<dim, Number>>
                <<<b_elem, t, 0, stream>>>(src, dst, conn, jac, Sv, Sg, flux, tau * kappa, n_elements);
        }

        energy_operator_bc_kernel<dim, Number><<<b_dofs, t, 0, stream>>>(
            src, dst, bc_type, n_dofs);
    }
};


template<int dim, typename Number>
class InternalEnergySolver {
private:
    cudaStream_t stream;
    const int n_dofs;

    mfcg::PCG<Number> pcg;
    EnergyOperator<dim, Number> op;

    Number* d_diag_inv = nullptr;  
    Number* d_Sv = nullptr;        
    Number* d_Sg = nullptr;

    const int* d_cij_row_offsets = nullptr;
    const Number* d_cij_values = nullptr;
    const Number* d_mij_values = nullptr;

public:
    InternalEnergySolver(int _n_dofs, cudaStream_t _stream = 0)
        : stream(_stream), n_dofs(_n_dofs), pcg(_n_dofs, _stream)
    {
        cudaMalloc(&d_diag_inv, n_dofs * sizeof(Number));
        op.n_dofs = n_dofs;
        op.stream = stream;

        std::vector<Number> Sv, Sg;
        mfops::build_sf_1d_tables<fe_degree, Number>(Sv, Sg);
        cudaMalloc(&d_Sv, Sv.size() * sizeof(Number));
        cudaMalloc(&d_Sg, Sg.size() * sizeof(Number));
        cudaMemcpy(d_Sv, Sv.data(), Sv.size() * sizeof(Number), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Sg, Sg.data(), Sg.size() * sizeof(Number), cudaMemcpyHostToDevice);
        op.Sv = d_Sv;
        op.Sg = d_Sg;
    }

    ~InternalEnergySolver() { cudaFree(d_diag_inv); cudaFree(d_Sv); cudaFree(d_Sg); }

    void set_system_matrices(
        const Number* density,
        const Number* lumped_mass,
        const int* cij_row_offsets,
        const int* ,
        const Number* cij_values,
        const Number* mij_values,
        Number _kappa)
    {
        op.density = density;
        op.lumped_mass = lumped_mass;
        op.kappa = _kappa;
        d_cij_row_offsets = cij_row_offsets;
        d_cij_values = cij_values;
        d_mij_values = mij_values;
    }

    void set_bc_type(const int* bc_type) { op.bc_type = bc_type; }

    void set_element_data(const int* element_nodes, const Number* jacobian_data, int _n_elements) {
        op.conn = element_nodes;
        op.jac = jacobian_data;
        op.n_elements = _n_elements;
    }

    void apply_operator(Number* dst, const Number* src, Number tau) {
        op.tau = tau;
        op.vmult(dst, src);
    }

    int solve(Number* energy_solution,
              const Number* energy_rhs,
              Number tau,
              Number tolerance,
              int max_iterations)
    {
        op.tau = tau;

        const int t = 256, blocks = (n_dofs + t - 1) / t;
        compute_energy_diagonal_preconditioner_kernel<dim><<<blocks, t, 0, stream>>>(
            d_diag_inv, op.density, op.lumped_mass, d_cij_row_offsets,
            d_cij_values, d_mij_values, op.kappa, tau, n_dofs);

        mfcg::DiagonalPreconditioner<Number> prec{d_diag_inv, n_dofs, stream};

        return pcg.solve(op, prec, energy_solution, energy_rhs, tolerance, max_iterations);
    }
};

#endif 
