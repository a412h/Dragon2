
#ifndef CELL_BASED_OPERATOR_CUH
#define CELL_BASED_OPERATOR_CUH

#include <cuda_runtime.h>
#include <vector>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include "fe_config.h"


template<int dim, typename Number, int degree>
__global__ void compute_viscous_heating_kernel_sf(
    const Number* __restrict__ velocity,
    Number* __restrict__ internal_energy_rhs,
    const int* __restrict__ conn,
    const Number* __restrict__ geom,
    const Number* __restrict__ Sv,
    const Number* __restrict__ Sg,
    Number mu, Number lambda, int n_elem)
{
    constexpr int P = degree + 1;
    constexpr int NQ = (dim == 2) ? P * P : P * P * P;
    constexpr int ND = NQ;
    constexpr int GS = dim * dim + 1;

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;
    const Number lambda_bar = lambda - Number(2.0 / 3.0) * mu;
    const Number two_mu = Number(2) * mu;

    int node[ND];
    Number u[ND][dim];
    for (int i = 0; i < ND; ++i) {
        node[i] = conn[e * ND + i];
        for (int c = 0; c < dim; ++c) u[i][c] = velocity[node[i] * dim + c];
    }
    Number sv[P * P], sg[P * P];
    for (int i = 0; i < P * P; ++i) { sv[i] = Sv[i]; sg[i] = Sg[i]; }

    auto stride = [](int ax) { return ax == 0 ? 1 : (ax == 1 ? P : P * P); };
    auto contract = [&](const Number* M, int ax, bool tr, const Number* s, Number* d) {
        const int st = stride(ax);
        for (int o = 0; o < NQ; ++o) {
            const int oax = (o / st) % P;
            const int base = o - oax * st;
            Number acc = Number(0);
            for (int k = 0; k < P; ++k) { const Number m = tr ? M[k * P + oax] : M[oax * P + k]; acc += m * s[base + k * st]; }
            d[o] = acc;
        }
    };
    Number bufA[NQ], bufB[NQ];
    auto sweep = [&](const Number* src, int dir, bool tr, Number* res) {
        const Number* s = src;
        for (int ax = 0; ax < dim; ++ax) { Number* d = (ax % 2 == 0) ? bufA : bufB; contract((ax == dir) ? sg : sv, ax, tr, s, d); s = d; }
        for (int i = 0; i < NQ; ++i) res[i] = s[i];
    };

    Number gref[dim][dim][NQ], ucomp[NQ];
    for (int c = 0; c < dim; ++c) {
        for (int i = 0; i < NQ; ++i) ucomp[i] = u[i][c];
        for (int d = 0; d < dim; ++d) sweep(ucomp, d, false, gref[c][d]);
    }

    Number hq[NQ];
    for (int q = 0; q < NQ; ++q) {
        const Number* Jinv = geom + (e * NQ + q) * GS;
        const Number JxW = geom[(e * NQ + q) * GS + dim * dim];
        Number gv[dim][dim];
        for (int c = 0; c < dim; ++c)
            for (int ee = 0; ee < dim; ++ee) {
                Number s = Number(0);
                for (int d = 0; d < dim; ++d) s += Jinv[d * dim + ee] * gref[c][d][q];
                gv[c][ee] = s;
            }
        Number div = Number(0);
        for (int i = 0; i < dim; ++i) div += gv[i][i];
        Number heating = Number(0);
        for (int i = 0; i < dim; ++i)
            for (int j = 0; j < dim; ++j) {
                const Number eps_ij = Number(0.5) * (gv[i][j] + gv[j][i]);
                Number S_ij = two_mu * eps_ij;
                if (i == j) S_ij += lambda_bar * div;
                heating += eps_ij * S_ij;
            }
        hq[q] = heating * JxW;
    }

    Number nc[ND];
    sweep(hq, -1, true, nc);
    for (int n = 0; n < ND; ++n) atomicAdd(&internal_energy_rhs[node[n]], nc[n]);
}


template<int dim, typename Number>
class ElementConnectivity {
public:
    int* d_element_nodes;
    Number* d_jacobian_data;
    int n_elements;
    int n_nodes;

    ElementConnectivity() 
        : d_element_nodes(nullptr), d_jacobian_data(nullptr), 
          n_elements(0), n_nodes(0) {
    }
    
    ~ElementConnectivity() {
        if (d_element_nodes != nullptr) cudaFree(d_element_nodes);
        if (d_jacobian_data != nullptr) cudaFree(d_jacobian_data);
    }
    
    void build_from_triangulation(const OfflineData<dim, double>& offline_data) {
        n_nodes = offline_data.dof_handler.n_dofs();
        n_elements = offline_data.dof_handler.get_triangulation().n_active_cells();
        
        std::cout << "  Building element connectivity from triangulation..." << std::endl;
        std::cout << "    Nodes: " << n_nodes << std::endl;
        std::cout << "    Elements: " << n_elements << std::endl;
        
        constexpr int P = fe_degree + 1;
        constexpr int nodes_per_elem = (dim == 2) ? P * P : P * P * P;  
        constexpr int n_q_points     = nodes_per_elem;                 
        constexpr int jac_data_per_quad = dim * dim + 1;               
        constexpr int jac_data_per_elem = n_q_points * jac_data_per_quad;
        
        std::vector<int> h_connectivity(n_elements * nodes_per_elem);
        std::vector<Number> h_jacobians(n_elements * jac_data_per_elem);

        dealii::QGauss<dim> quadrature(fe_degree + 1);
        dealii::FEValues<dim> fe_values(
            offline_data.finite_element,
            quadrature,
            dealii::update_inverse_jacobians | dealii::update_JxW_values);

        const std::vector<unsigned int> h2l =
            dealii::FETools::hierarchic_to_lexicographic_numbering<dim>(fe_degree);
        
        const unsigned int dofs_per_cell = offline_data.finite_element.dofs_per_cell;
        std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
        
        std::cout << "    DoFs per cell: " << dofs_per_cell << std::endl;
        
        if (dofs_per_cell != nodes_per_elem) {
            std::cerr << "ERROR: Element has " << dofs_per_cell 
                      << " DoFs but expected " << nodes_per_elem << std::endl;
            throw std::runtime_error("Mismatch in element DoF count");
        }
        
        int elem_id = 0;
        for (const auto& cell : offline_data.dof_handler.active_cell_iterators()) {

            cell->get_dof_indices(local_dof_indices);

            for (unsigned int h = 0; h < static_cast<unsigned>(nodes_per_elem); ++h) {
                const unsigned int master = offline_data.periodic_master[local_dof_indices[h]];
                h_connectivity[elem_id * nodes_per_elem + h2l[h]] = master;  
            }

            fe_values.reinit(cell);
            for (unsigned int q = 0; q < static_cast<unsigned>(n_q_points); ++q) {
                const auto& J_inv = fe_values.inverse_jacobian(q);  
                const int offset = elem_id * jac_data_per_elem + q * jac_data_per_quad;
                for (int i = 0; i < dim; ++i)
                    for (int j = 0; j < dim; ++j)
                        h_jacobians[offset + i * dim + j] = static_cast<Number>(J_inv[i][j]);
                h_jacobians[offset + dim * dim] = static_cast<Number>(fe_values.JxW(q));
            }
            
            elem_id++;
        }
        
        std::cout << "    Processed " << elem_id << " elements" << std::endl;

        cudaError_t err1 = cudaMalloc(&d_element_nodes, h_connectivity.size() * sizeof(int));
        cudaError_t err2 = cudaMalloc(&d_jacobian_data, h_jacobians.size() * sizeof(Number));
        
        if (err1 != cudaSuccess || err2 != cudaSuccess) {
            std::cerr << "ERROR: CUDA malloc failed!" << std::endl;
            std::cerr << "  Element nodes: " << cudaGetErrorString(err1) << std::endl;
            std::cerr << "  Jacobian data: " << cudaGetErrorString(err2) << std::endl;
            throw std::runtime_error("CUDA allocation failure");
        }
        
        std::cout << "    Allocated GPU memory" << std::endl;
        std::cout << "      Element nodes: " << h_connectivity.size() * sizeof(int) << " bytes" << std::endl;
        std::cout << "      Jacobian data: " << h_jacobians.size() * sizeof(Number) << " bytes" << std::endl;
        
        cudaMemcpy(d_element_nodes, h_connectivity.data(),
                   h_connectivity.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_jacobian_data, h_jacobians.data(),
                   h_jacobians.size() * sizeof(Number), cudaMemcpyHostToDevice);
        
        std::cout << "    Element connectivity transferred to GPU successfully" << std::endl;
    }
};

#endif