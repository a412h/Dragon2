#ifndef VELOCITY_SOLVER_CUH
#define VELOCITY_SOLVER_CUH

#include <cuda_runtime.h>
#include <type_traits>
#include "mf_operators.cuh"   


template<int dim, typename Number>
__global__ void velocity_operator_bc_kernel(
    const Number* velocity_in,
    Number* velocity_out,
    const int* bc_type,
    const int* bc_index,
    const Number* boundary_normals,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    const int id = bc_type[i];

    if (id == 4 || id == 3) {  
        for (int d = 0; d < dim; ++d)
            velocity_out[i * dim + d] = velocity_in[i * dim + d];
    }
    else if (id == 2) {  
        const int b = bc_index[i];
        Number n_vec[dim];
        for (int d = 0; d < dim; ++d) n_vec[d] = boundary_normals[b * dim + d];

        Number dst_n = Number(0), src_n = Number(0);
        for (int d = 0; d < dim; ++d) {
            dst_n += velocity_out[i * dim + d] * n_vec[d];
            src_n += velocity_in[i * dim + d] * n_vec[d];
        }
        for (int d = 0; d < dim; ++d)
            velocity_out[i * dim + d] += (src_n - dst_n) * n_vec[d];
    }
}


template<int dim, typename Number>
__global__ void compute_diagonal_preconditioner_kernel(
    Number* diagonal_inv,
    const Number* density,
    const Number* lumped_mass_matrix,
    const int* cij_row_offsets,
    const Number* cij_values,
    Number mu,
    Number lambda,
    Number tau,
    int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;

    Number diag_value = lumped_mass_matrix[i] * density[i];

    const int row_start = cij_row_offsets[i];
    const int row_end = cij_row_offsets[i + 1];
    Number visc_diag = Number(0);
    for (int idx = row_start; idx < row_end; ++idx)
        for (int d = 0; d < dim; ++d) {
            Number c_d = cij_values[idx * dim + d];
            visc_diag += c_d * c_d;
        }
    diag_value += tau * (Number(2) * mu + lambda) * visc_diag;

    const Number inv = (diag_value > Number(1e-30)) ? Number(1) / diag_value : Number(0);
    for (int d = 0; d < dim; ++d)
        diagonal_inv[i * dim + d] = inv;
}


template<int dim, typename Number>
struct VelocityOperator {

    const Number* density = nullptr;
    const Number* lumped_mass = nullptr;

    const int* conn = nullptr;
    const Number* jac = nullptr;     
    const Number* Sv = nullptr;      
    const Number* Sg = nullptr;      
    int n_elements = 0;

    const int* bc_type = nullptr;
    const int* bc_index = nullptr;
    const Number* normals = nullptr;

    Number mu = 0, lambda = 0, tau = 0;

    int n_dofs = 0;
    cudaStream_t stream = 0;

    int size() const { return n_dofs * dim; }

    void vmult(Number* dst, const Number* src) const {
        const int t = 256;
        const int b_dofs = (n_dofs + t - 1) / t;
        const int b_elem = (n_elements + t - 1) / t;

        mfops::mf_mass_diagonal_kernel<Number, dim><<<b_dofs, t, 0, stream>>>(
            src, dst, lumped_mass, density, n_dofs);

        if (mu != Number(0) || lambda != Number(0)) {
            mfops::StressFlux<dim, Number> flux{Number(2) * mu, lambda - Number(2.0 / 3.0) * mu};
            mfops::mf_cell_gradient_kernel_sf<dim, Number, fe_degree, dim, mfops::StressFlux<dim, Number>>
                <<<b_elem, t, 0, stream>>>(src, dst, conn, jac, Sv, Sg, flux, tau, n_elements);
        }

        velocity_operator_bc_kernel<dim, Number><<<b_dofs, t, 0, stream>>>(
            src, dst, bc_type, bc_index, normals, n_dofs);
    }
};


template<int dim, typename Number>
class VelocitySolver {
private:
    cudaStream_t stream;
    const int n_dofs;

    mfcg::PCG<Number> pcg;
    VelocityOperator<dim, Number> op;

    Number* d_diag_inv = nullptr;  
    Number* d_Sv = nullptr;        
    Number* d_Sg = nullptr;

    const int* d_cij_row_offsets = nullptr;
    const Number* d_cij_values = nullptr;

public:
    VelocitySolver(int _n_dofs, cudaStream_t _stream = 0)
        : stream(_stream), n_dofs(_n_dofs), pcg(_n_dofs * dim, _stream)
    {
        cudaMalloc(&d_diag_inv, n_dofs * dim * sizeof(Number));
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

    ~VelocitySolver() { cudaFree(d_diag_inv); cudaFree(d_Sv); cudaFree(d_Sg); }

    void set_system_matrices(
        const Number* density,
        const Number* lumped_mass_matrix,
        const int* cij_row_offsets,
        const int* ,   
        const Number* cij_values,
        const Number* ,     
        Number _mu,
        Number _lambda)
    {
        op.density = density;
        op.lumped_mass = lumped_mass_matrix;
        op.mu = _mu;
        op.lambda = _lambda;
        d_cij_row_offsets = cij_row_offsets;
        d_cij_values = cij_values;
    }

    void set_boundary_data(
        const int* bc_type,
        const int* bc_index,
        const Number* boundary_normals)
    {
        op.bc_type = bc_type;
        op.bc_index = bc_index;
        op.normals = boundary_normals;
    }

    void set_element_data(
        const int* element_nodes,
        const Number* jacobian_data,
        int _n_elements)
    {
        op.conn = element_nodes;
        op.jac = jacobian_data;
        op.n_elements = _n_elements;
    }

    void apply_operator(Number* dst, const Number* src, Number tau)
    {
        op.tau = tau;
        op.vmult(dst, src);
    }

    int solve(Number* velocity_solution,
              const Number* velocity_rhs,
              Number tau,
              Number tolerance,
              int max_iterations)
    {
        op.tau = tau;

        const int t = 256, blocks = (n_dofs + t - 1) / t;
        compute_diagonal_preconditioner_kernel<dim><<<blocks, t, 0, stream>>>(
            d_diag_inv, op.density, op.lumped_mass, d_cij_row_offsets,
            d_cij_values, op.mu, op.lambda, tau, n_dofs);

        mfcg::DiagonalPreconditioner<Number> prec{d_diag_inv, n_dofs * dim, stream};

        return pcg.solve(op, prec, velocity_solution, velocity_rhs, tolerance, max_iterations);
    }
};

#endif 
