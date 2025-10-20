// For Step 2: Build internal energy right hand side
#ifndef CELL_BASED_OPERATOR_CUH
#define CELL_BASED_OPERATOR_CUH

#include <cuda_runtime.h>
#include <vector>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>

// ============================================================================
// Constant memory for quadrature data (optimization)
// ============================================================================
namespace QuadratureData {
    // 2D quadrature points (Gauss)
    __constant__ float c_quad_pts_2d[4][2] = {
        {-0.57735026918962576451f, -0.57735026918962576451f},
        { 0.57735026918962576451f, -0.57735026918962576451f},
        { 0.57735026918962576451f,  0.57735026918962576451f},
        {-0.57735026918962576451f,  0.57735026918962576451f}
    };
    // 3D quadrature points (Gauss)
    __constant__ float c_quad_pts_3d[8][3] = {
        {-0.57735026918962576451f, -0.57735026918962576451f, -0.57735026918962576451f},
        { 0.57735026918962576451f, -0.57735026918962576451f, -0.57735026918962576451f},
        { 0.57735026918962576451f,  0.57735026918962576451f, -0.57735026918962576451f},
        {-0.57735026918962576451f,  0.57735026918962576451f, -0.57735026918962576451f},
        {-0.57735026918962576451f, -0.57735026918962576451f,  0.57735026918962576451f},
        { 0.57735026918962576451f, -0.57735026918962576451f,  0.57735026918962576451f},
        { 0.57735026918962576451f,  0.57735026918962576451f,  0.57735026918962576451f},
        {-0.57735026918962576451f,  0.57735026918962576451f,  0.57735026918962576451f}
    };
    // Constant 1 / sqrt(3)
    __constant__ float c_gp = 0.57735026918962576451f;
}

// ============================================================================
// Kernels
// ============================================================================
template<int dim, typename Number, int nodes_per_elem = (dim == 2) ? 4 : 8>
__global__ void compute_viscous_heating_kernel(
    const Number* velocity,              // Input: nodal velocities [n_nodes * dim]
    Number* internal_energy_rhs,         // Output: accumulated m_i * K_i [n_nodes]
    const int* element_connectivity,     // Element->node mapping [n_elements * nodes_per_elem]
    const Number* jacobian_data,         // Jacobians [n_elements * dim * dim * 2]
    const Number* lumped_mass_matrix,    // Lumped mass matrix [n_nodes]
    Number mu,
    Number lambda,
    int n_elements,
    int n_nodes)
{
    const int elem_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (elem_id >= n_elements) return;
    
    // Shared memory for velocity data (optimization)
    __shared__ Number s_vel_elem[128][nodes_per_elem][dim];
    const int tid = threadIdx.x;
    
    // Load element connectivity
    int nodes[nodes_per_elem];
    #pragma unroll nodes_per_elem
    for (int n = 0; n < nodes_per_elem; ++n) {
        nodes[n] = element_connectivity[elem_id * nodes_per_elem + n];
    }
    
    // Load velocity data (optimization)
    #pragma unroll nodes_per_elem
    for (int n = 0; n < nodes_per_elem; ++n) {
        for (int d = 0; d < dim; ++d) {
            __ldg(&velocity[nodes[n] * dim + d]);
        }
    }
    
    // Load element velocities (optimization)
    #pragma unroll nodes_per_elem
    for (int n = 0; n < nodes_per_elem; ++n) {
        #pragma unroll
        for (int d = 0; d < dim; ++d) {
            s_vel_elem[tid][n][d] = velocity[nodes[n] * dim + d];
        }
    }
    __syncthreads();
    
    // Create local reference
    Number (&vel_elem)[nodes_per_elem][dim] = 
        reinterpret_cast<Number(&)[nodes_per_elem][dim]>(s_vel_elem[tid]);
    
    // Load Jacobian and inverse
    const int jac_offset = elem_id * dim * dim * 2;
    Number J[dim][dim], J_inv[dim][dim], det_J;
    
    #pragma unroll
    for (int i = 0; i < dim; ++i) {
        #pragma unroll
        for (int j = 0; j < dim; ++j) {
            J[i][j] = jacobian_data[jac_offset + i * dim + j];
            J_inv[i][j] = jacobian_data[jac_offset + dim * dim + i * dim + j];
        }
    }
    
    if constexpr (dim == 2) {
        det_J = J[0][0] * J[1][1] - J[0][1] * J[1][0];
    } else {
        det_J = J[0][0] * (J[1][1] * J[2][2] - J[1][2] * J[2][1]) -
                J[0][1] * (J[1][0] * J[2][2] - J[1][2] * J[2][0]) +
                J[0][2] * (J[1][0] * J[2][1] - J[1][1] * J[2][0]);
    }
    
    // Initialize node contributions with explicit unroll
    Number node_contributions[nodes_per_elem];
    #pragma unroll nodes_per_elem
    for (int n = 0; n < nodes_per_elem; ++n) {
        node_contributions[n] = Number(0);
    }
    
    // Loop over quadrature points
    if constexpr (dim == 2) {
        // OPTIMIZATION 2: Use constant memory for quadrature points
        const Number gp = Number(QuadratureData::c_gp);
        
        #pragma unroll 4
        for (int q = 0; q < 4; ++q) {
            const Number xi = Number(QuadratureData::c_quad_pts_2d[q][0]);
            const Number eta = Number(QuadratureData::c_quad_pts_2d[q][1]);
            
            // Shape functions at quadrature point
            Number phi[4];
            phi[0] = Number(0.25) * (1 - xi) * (1 - eta);
            phi[1] = Number(0.25) * (1 + xi) * (1 - eta);
            phi[2] = Number(0.25) * (1 + xi) * (1 + eta);
            phi[3] = Number(0.25) * (1 - xi) * (1 + eta);
            
            // Shape function gradients on reference element
            Number grad_phi_ref[4][2];
            grad_phi_ref[0][0] = -Number(0.25) * (1 - eta);
            grad_phi_ref[0][1] = -Number(0.25) * (1 - xi);
            grad_phi_ref[1][0] =  Number(0.25) * (1 - eta);
            grad_phi_ref[1][1] = -Number(0.25) * (1 + xi);
            grad_phi_ref[2][0] =  Number(0.25) * (1 + eta);
            grad_phi_ref[2][1] =  Number(0.25) * (1 + xi);
            grad_phi_ref[3][0] = -Number(0.25) * (1 + eta);
            grad_phi_ref[3][1] =  Number(0.25) * (1 - xi);
            
            // Transform gradients to physical element
            Number grad_phi[4][2];
            #pragma unroll 4
            for (int n = 0; n < 4; ++n) {
                #pragma unroll 2
                for (int i = 0; i < 2; ++i) {
                    grad_phi[n][i] = J_inv[0][i] * grad_phi_ref[n][0] + 
                                     J_inv[1][i] * grad_phi_ref[n][1];
                }
            }
            
            // Compute velocity gradient at quadrature point
            Number grad_v[2][2] = {{0, 0}, {0, 0}};
            #pragma unroll 4
            for (int n = 0; n < 4; ++n) {
                #pragma unroll 2
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll 2
                    for (int j = 0; j < 2; ++j) {
                        grad_v[i][j] += vel_elem[n][i] * grad_phi[n][j];
                    }
                }
            }
            
            // Symmetric gradient
            Number eps[2][2];
            eps[0][0] = grad_v[0][0];
            eps[0][1] = Number(0.5) * (grad_v[0][1] + grad_v[1][0]);
            eps[1][0] = eps[0][1];
            eps[1][1] = grad_v[1][1];
            
            // Divergence
            const Number div_v = eps[0][0] + eps[1][1];
            
            // Stress tensor S
            const Number lambda_bar = lambda - Number(2.0/3.0) * mu;
            const Number two_mu = Number(2) * mu;
            Number S[2][2];
            S[0][0] = two_mu * eps[0][0] + lambda_bar * div_v;
            S[0][1] = two_mu * eps[0][1];
            S[1][0] = S[0][1];
            S[1][1] = two_mu * eps[1][1] + lambda_bar * div_v;
            
            // Viscous heating: eps : S
            const Number heating_q = eps[0][0] * S[0][0] + 
                                    Number(2) * eps[0][1] * S[0][1] + 
                                    eps[1][1] * S[1][1];
            
            // Integration weight
            const Number weight = abs(det_J);
            
            // Distribute to nodes using shape functions
            #pragma unroll 4
            for (int n = 0; n < 4; ++n) {
                node_contributions[n] += weight * phi[n] * heating_q;
            }
        }
    } else {
        // 3D case - hexahedron with 8 nodes and 8 quadrature points
        
        #pragma unroll 8
        for (int q = 0; q < 8; ++q) {
            // Optimization
            const Number xi = Number(QuadratureData::c_quad_pts_3d[q][0]);
            const Number eta = Number(QuadratureData::c_quad_pts_3d[q][1]); 
            const Number zeta = Number(QuadratureData::c_quad_pts_3d[q][2]);
            
            // Shape functions for Q1 hexahedron at quadrature point
            Number phi[8];
            phi[0] = Number(0.125) * (1 - xi) * (1 - eta) * (1 - zeta);
            phi[1] = Number(0.125) * (1 + xi) * (1 - eta) * (1 - zeta);
            phi[2] = Number(0.125) * (1 + xi) * (1 + eta) * (1 - zeta);
            phi[3] = Number(0.125) * (1 - xi) * (1 + eta) * (1 - zeta);
            phi[4] = Number(0.125) * (1 - xi) * (1 - eta) * (1 + zeta);
            phi[5] = Number(0.125) * (1 + xi) * (1 - eta) * (1 + zeta);
            phi[6] = Number(0.125) * (1 + xi) * (1 + eta) * (1 + zeta);
            phi[7] = Number(0.125) * (1 - xi) * (1 + eta) * (1 + zeta);
            
            // Shape function gradients on reference element
            Number grad_phi_ref[8][3];
            grad_phi_ref[0][0] = -Number(0.125) * (1 - eta) * (1 - zeta);
            grad_phi_ref[0][1] = -Number(0.125) * (1 - xi) * (1 - zeta);
            grad_phi_ref[0][2] = -Number(0.125) * (1 - xi) * (1 - eta);
            
            grad_phi_ref[1][0] =  Number(0.125) * (1 - eta) * (1 - zeta);
            grad_phi_ref[1][1] = -Number(0.125) * (1 + xi) * (1 - zeta);
            grad_phi_ref[1][2] = -Number(0.125) * (1 + xi) * (1 - eta);
            
            grad_phi_ref[2][0] =  Number(0.125) * (1 + eta) * (1 - zeta);
            grad_phi_ref[2][1] =  Number(0.125) * (1 + xi) * (1 - zeta);
            grad_phi_ref[2][2] = -Number(0.125) * (1 + xi) * (1 + eta);
            
            grad_phi_ref[3][0] = -Number(0.125) * (1 + eta) * (1 - zeta);
            grad_phi_ref[3][1] =  Number(0.125) * (1 - xi) * (1 - zeta);
            grad_phi_ref[3][2] = -Number(0.125) * (1 - xi) * (1 + eta);
            
            grad_phi_ref[4][0] = -Number(0.125) * (1 - eta) * (1 + zeta);
            grad_phi_ref[4][1] = -Number(0.125) * (1 - xi) * (1 + zeta);
            grad_phi_ref[4][2] =  Number(0.125) * (1 - xi) * (1 - eta);
            
            grad_phi_ref[5][0] =  Number(0.125) * (1 - eta) * (1 + zeta);
            grad_phi_ref[5][1] = -Number(0.125) * (1 + xi) * (1 + zeta);
            grad_phi_ref[5][2] =  Number(0.125) * (1 + xi) * (1 - eta);
            
            grad_phi_ref[6][0] =  Number(0.125) * (1 + eta) * (1 + zeta);
            grad_phi_ref[6][1] =  Number(0.125) * (1 + xi) * (1 + zeta);
            grad_phi_ref[6][2] =  Number(0.125) * (1 + xi) * (1 + eta);
            
            grad_phi_ref[7][0] = -Number(0.125) * (1 + eta) * (1 + zeta);
            grad_phi_ref[7][1] =  Number(0.125) * (1 - xi) * (1 + zeta);
            grad_phi_ref[7][2] =  Number(0.125) * (1 - xi) * (1 + eta);
            
            // Transform gradients to physical element
            Number grad_phi[8][3];
            #pragma unroll 8
            for (int n = 0; n < 8; ++n) {
                #pragma unroll 3
                for (int i = 0; i < 3; ++i) {
                    grad_phi[n][i] = 0;
                    #pragma unroll 3
                    for (int j = 0; j < 3; ++j) {
                        grad_phi[n][i] += J_inv[j][i] * grad_phi_ref[n][j];
                    }
                }
            }
            
            // Compute velocity gradient at quadrature point
            Number grad_v[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
            #pragma unroll 8
            for (int n = 0; n < 8; ++n) {
                #pragma unroll 3
                for (int i = 0; i < 3; ++i) {
                    #pragma unroll 3
                    for (int j = 0; j < 3; ++j) {
                        grad_v[i][j] += vel_elem[n][i] * grad_phi[n][j];
                    }
                }
            }
            
            // Symmetric gradient (strain rate tensor)
            Number eps[3][3];
            #pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                #pragma unroll 3
                for (int j = 0; j < 3; ++j) {
                    eps[i][j] = Number(0.5) * (grad_v[i][j] + grad_v[j][i]);
                }
            }
            
            // Divergence
            const Number div_v = eps[0][0] + eps[1][1] + eps[2][2];
            
            // Stress tensor S = 2*mu*eps + (lambda - 2/3*mu)*div*I
            const Number lambda_bar = lambda - Number(2.0/3.0) * mu;
            const Number two_mu = Number(2) * mu;
            Number S[3][3];
            #pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                #pragma unroll 3
                for (int j = 0; j < 3; ++j) {
                    S[i][j] = two_mu * eps[i][j];
                }
                S[i][i] += lambda_bar * div_v;
            }
            
            // Viscous heating: eps : S
            Number heating_q = 0;
            #pragma unroll 3
            for (int i = 0; i < 3; ++i) {
                #pragma unroll 3
                for (int j = 0; j < 3; ++j) {
                    heating_q += eps[i][j] * S[i][j];
                }
            }
            
            // Integration weight
            const Number weight = abs(det_J);
            
            // Distribute to nodes using shape functions
            #pragma unroll 8
            for (int n = 0; n < 8; ++n) {
                node_contributions[n] += weight * phi[n] * heating_q;
            }
        }
    }
    
    // Distribute to global nodes with atomic add
    #pragma unroll nodes_per_elem
    for (int n = 0; n < nodes_per_elem; ++n) {
        atomicAdd(&internal_energy_rhs[nodes[n]], node_contributions[n]);
    }
}

// Element connectivity class (unchanged)
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
        
        constexpr int nodes_per_elem = (dim == 2) ? 4 : 8;
        constexpr int jac_data_per_elem = (dim == 2) ? 8 : 18;
        
        std::vector<int> h_connectivity(n_elements * nodes_per_elem);
        std::vector<Number> h_jacobians(n_elements * jac_data_per_elem);
        
        // Setup for extracting Jacobian
        dealii::QGauss<dim> quadrature(1);
        dealii::FEValues<dim> fe_values(
            offline_data.finite_element, 
            quadrature,
            dealii::update_jacobians | dealii::update_inverse_jacobians);
        
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
            // Connectivity
            cell->get_dof_indices(local_dof_indices);
            
            for (unsigned int i = 0; i < nodes_per_elem; ++i) {
                h_connectivity[elem_id * nodes_per_elem + i] = local_dof_indices[i];
            }
            // Jacobians
            fe_values.reinit(cell);
            const auto& J = fe_values.jacobian(0);
            const auto& J_inv = fe_values.inverse_jacobian(0);
            
            if constexpr (dim == 2) {
                h_jacobians[elem_id * 8 + 0] = static_cast<Number>(J[0][0]);
                h_jacobians[elem_id * 8 + 1] = static_cast<Number>(J[0][1]);
                h_jacobians[elem_id * 8 + 2] = static_cast<Number>(J[1][0]);
                h_jacobians[elem_id * 8 + 3] = static_cast<Number>(J[1][1]);
                
                h_jacobians[elem_id * 8 + 4] = static_cast<Number>(J_inv[0][0]);
                h_jacobians[elem_id * 8 + 5] = static_cast<Number>(J_inv[0][1]);
                h_jacobians[elem_id * 8 + 6] = static_cast<Number>(J_inv[1][0]);
                h_jacobians[elem_id * 8 + 7] = static_cast<Number>(J_inv[1][1]);
            } else {
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        h_jacobians[elem_id * 18 + i * 3 + j] = static_cast<Number>(J[i][j]);
                        h_jacobians[elem_id * 18 + 9 + i * 3 + j] = static_cast<Number>(J_inv[i][j]);
                    }
                }
            }
            
            elem_id++;
        }
        
        std::cout << "    Processed " << elem_id << " elements" << std::endl;
        
        // Transfer data to GPU
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