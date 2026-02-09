
#ifndef DATA_STRUCT_CUH
#define DATA_STRUCT_CUH

#include <cuda_runtime.h>
#include <driver_types.h>
#include <vector>
#include <array>
#include "configuration.h"
#include "output.h"

enum class TimeScheme { ERK33_CN, SSPRK33_CN };
#include "offline_data.h"

#define CUDA_CHECK(call) { cudaError_t error = call; if (error != cudaSuccess) { fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); exit(1);} }

template<int dim, typename Number>
struct State {
    Number* rho;
    Number* momentum_x;
    Number* momentum_y;
    Number* momentum_z;
    Number* energy;

    __device__ __host__ Number* momentum(int d) {
        if (d == 0) return momentum_x;
        else if (d == 1) return momentum_y;
        else return momentum_z;
    }

    __device__ __host__ const Number* momentum(int d) const {
        if (d == 0) return momentum_x;
        else if (d == 1) return momentum_y;
        else return momentum_z;
    }
};

template<int dim, typename Number>
struct Flux {
    static constexpr int problem_dimension = dim + 2;
    Number F[problem_dimension * dim];
    __host__ __device__ Number& operator()(int component, int dimension);
    __host__ __device__ const Number& operator()(int component, int dimension) const;
};

template<int dim, typename Number>
struct Momentum {
    Number data[dim];
    __device__ Number& operator[](int i) { return data[i]; }
    __device__ const Number& operator[](int i) const { return data[i]; }
};

template<typename Number>
struct MijMatrix {
    int* row_offsets;
    int* col_indices;
    Number* values;
};

template<typename Number>
struct MiMatrix {
    Number* values;
};

template<typename Number>
struct MiMatrixInverse {
    Number* values;
};

template<int dim, typename Number>
struct CijMatrix {
    int* row_offsets;
    int* col_indices;
    Number* values;
};

template<int dim, typename Number>
struct BoundaryData {
    int* boundary_dofs;
    int* boundary_ids;
    Number* boundary_normals;
    int n_boundary_dofs;

    int* bc_type;
    int* bc_index;
};

struct CouplingPairs {
    int* internal_pairs;
    int* boundary_pairs;
    int n_internal_pairs;
    int n_boundary_pairs;
};

struct Sparsity {
    int* row_offsets;
    int* col_indices;
    int* transpose_indices;
};

template<int dim, typename Number>
struct Pij {
    Number* p_rho;
    Number* p_momentum_x;
    Number* p_momentum_y;
    Number* p_momentum_z;
    Number* p_energy;

    __device__ __host__ Number* p_momentum(int d) {
        if (d == 0) return p_momentum_x;
        else if (d == 1) return p_momentum_y;
        else return p_momentum_z;
    }
};

template<int dim, typename Number>
struct Ri {
    Number* r_rho;
    Number* r_momentum_x;
    Number* r_momentum_y;
    Number* r_momentum_z;
    Number* r_energy;

    __device__ __host__ Number* r_momentum(int d) {
        if (d == 0) return r_momentum_x;
        else if (d == 1) return r_momentum_y;
        else return r_momentum_z;
    }
};

template<int dim, typename Number>
struct ERK33Params {
    int n_steps_per_batch;
    State<dim, Number> d_U;
    State<dim, Number> d_U_n;
    State<dim, Number> d_temp_0;
    State<dim, Number> d_temp_1;
    State<dim, Number> d_new_U;
    Number* d_pressure;
    Number* d_speed_of_sound;
    Number* d_precomputed;
    Number* d_alpha_i;
    Number* d_dij;
    Pij<dim, Number> d_pij;
    Ri<dim, Number> d_ri;
    Number* d_bounds;
    Number* d_lij;
    Number* d_lij_next;
    MijMatrix<Number> d_mij;
    MiMatrix<Number> d_mi;
    MiMatrixInverse<Number> d_mi_inv;
    CijMatrix<dim, Number> d_cij;
    BoundaryData<dim, Number> d_boundary_data;
    CouplingPairs d_coupling_pairs;
    Number inflow_rho;
    Number inflow_momentum_x;
    Number inflow_momentum_y;
    Number inflow_momentum_z;
    Number inflow_energy;
    Number measure_of_omega;
    Number cfl;
    Number evc_factor;
    Number tau_max;
    Number* d_block_tau_mins;
    int n_blocks;
    int n_dofs;
    int nnz;
    Sparsity d_sparsity;
};

template<typename Number>
struct PrimitiveType {
    static constexpr int riemann_data_size = 4;
    Number data[riemann_data_size];
};


template<int dim, typename Number>
__host__ __device__ Number& Flux<dim, Number>::operator()(int component, int dimension) {
    return F[component * dim + dimension];
}

template<int dim, typename Number>
__host__ __device__ const Number& Flux<dim, Number>::operator()(int component, int dimension) const {
    return F[component * dim + dimension];
}



template<int dim, typename Number>
void allocate_state(State<dim, Number>& state, int n_dofs) {
    CUDA_CHECK(cudaMalloc(&state.rho, n_dofs * sizeof(Number)));
    CUDA_CHECK(cudaMalloc(&state.momentum_x, n_dofs * sizeof(Number)));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMalloc(&state.momentum_y, n_dofs * sizeof(Number)));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMalloc(&state.momentum_z, n_dofs * sizeof(Number)));
    }
    CUDA_CHECK(cudaMalloc(&state.energy, n_dofs * sizeof(Number)));
}

template<int dim, typename Number>
void free_state(State<dim, Number>& state) {
    CUDA_CHECK(cudaFree(state.rho));
    CUDA_CHECK(cudaFree(state.momentum_x));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaFree(state.momentum_y));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaFree(state.momentum_z));
    }
    CUDA_CHECK(cudaFree(state.energy));
}

template<int dim, typename Number>
void allocate_pij(Pij<dim, Number>& pij, int nnz) {
    CUDA_CHECK(cudaMalloc(&pij.p_rho, nnz * sizeof(Number)));
    CUDA_CHECK(cudaMalloc(&pij.p_momentum_x, nnz * sizeof(Number)));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMalloc(&pij.p_momentum_y, nnz * sizeof(Number)));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMalloc(&pij.p_momentum_z, nnz * sizeof(Number)));
    }
    CUDA_CHECK(cudaMalloc(&pij.p_energy, nnz * sizeof(Number)));
}

template<int dim, typename Number>
void free_pij(Pij<dim, Number>& pij) {
    CUDA_CHECK(cudaFree(pij.p_rho));
    CUDA_CHECK(cudaFree(pij.p_momentum_x));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaFree(pij.p_momentum_y));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaFree(pij.p_momentum_z));
    }
    CUDA_CHECK(cudaFree(pij.p_energy));
}

template<int dim, typename Number>
void allocate_ri(Ri<dim, Number>& ri, int n_dofs) {
    CUDA_CHECK(cudaMalloc(&ri.r_rho, n_dofs * sizeof(Number)));
    CUDA_CHECK(cudaMalloc(&ri.r_momentum_x, n_dofs * sizeof(Number)));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaMalloc(&ri.r_momentum_y, n_dofs * sizeof(Number)));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaMalloc(&ri.r_momentum_z, n_dofs * sizeof(Number)));
    }
    CUDA_CHECK(cudaMalloc(&ri.r_energy, n_dofs * sizeof(Number)));
}

template<int dim, typename Number>
void free_ri(Ri<dim, Number>& ri) {
    CUDA_CHECK(cudaFree(ri.r_rho));
    CUDA_CHECK(cudaFree(ri.r_momentum_x));
    if constexpr (dim >= 2) {
        CUDA_CHECK(cudaFree(ri.r_momentum_y));
    }
    if constexpr (dim == 3) {
        CUDA_CHECK(cudaFree(ri.r_momentum_z));
    }
    CUDA_CHECK(cudaFree(ri.r_energy));
}

template<int dim, typename Number_cu, TimeScheme scheme>
Number_cu cuda_time_loop(
    const MijMatrix<Number_cu>& d_mij_matrix,
    const MiMatrix<Number_cu>& d_mi_matrix,
    const MiMatrixInverse<Number_cu>& d_mi_inv_matrix,
    const CijMatrix<dim, Number_cu>& d_cij_matrix,
    const Sparsity& d_sparsity,
    State<dim, Number_cu>& d_U,
    const BoundaryData<dim, Number_cu>& d_boundary_data,
    const CouplingPairs& d_coupling_pairs,
    Number_cu measure_of_omega,
    int n_dofs,
    int nnz_mij,
    int nnz_cij,
    const Configuration& config,
    const OfflineData<dim, double>& offline_data,
    VTUOutput<dim>* output_handler);

#endif