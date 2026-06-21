#ifndef MF_OPERATORS_CUH
#define MF_OPERATORS_CUH

#include <cuda_runtime.h>
#include <cmath>
#include <vector>
#include <deal.II/base/quadrature_lib.h>
#include "fe_config.h"
#include "mf_pcg.cuh"

namespace mfops {

template <typename Number, int n_comp>
__global__ void mf_mass_diagonal_kernel(const Number* __restrict__ in,
                                        Number* __restrict__ out,
                                        const Number* __restrict__ lumped_mass,
                                        const Number* __restrict__ density,
                                        int n_dofs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_dofs) return;
    const Number mr = lumped_mass[i] * density[i];

    #pragma unroll
    for (int c = 0; c < n_comp; ++c)
        out[i * n_comp + c] = mr * in[i * n_comp + c];
}

template <int dim, typename Number>
struct LaplaceFlux {
    __device__ __forceinline__
    void operator()(const Number grad[][dim], Number S[][dim]) const {
#pragma unroll
        for (int i = 0; i < dim; ++i) S[0][i] = grad[0][i];
    }
};

template <int dim, typename Number>
struct StressFlux {
    Number two_mu;
    Number lambda_bar;
    __device__ __forceinline__
    void operator()(const Number grad[][dim], Number S[][dim]) const {
        Number div = Number(0);
#pragma unroll
        for (int i = 0; i < dim; ++i) div += grad[i][i];
#pragma unroll
        for (int i = 0; i < dim; ++i) {
#pragma unroll
            for (int j = 0; j < dim; ++j)
                S[i][j] = two_mu * Number(0.5) * (grad[i][j] + grad[j][i]);
            S[i][i] += lambda_bar * div;
        }
    }
};

template <int dim, typename Number, int degree, int n_comp, class Flux>
__global__ void mf_cell_gradient_kernel_sf(
    const Number* __restrict__ in,
    Number* __restrict__ out,
    const int* __restrict__ conn,        
    const Number* __restrict__ geom,     
    const Number* __restrict__ Sv,       
    const Number* __restrict__ Sg,       
    Flux flux, Number scale, int n_elem)
{
    constexpr int P  = degree + 1;
    constexpr int NQ = (dim == 2) ? P * P : P * P * P;
    constexpr int ND = NQ;
    constexpr int GS = dim * dim + 1;

    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_elem) return;

    int node[ND];
    Number u[ND][n_comp];
    for (int i = 0; i < ND; ++i) {
        node[i] = conn[e * ND + i];
        for (int c = 0; c < n_comp; ++c) u[i][c] = in[node[i] * n_comp + c];
    }
    Number sv[P * P], sg[P * P];
    for (int i = 0; i < P * P; ++i) { sv[i] = Sv[i]; sg[i] = Sg[i]; }

    auto stride = [](int ax) { return ax == 0 ? 1 : (ax == 1 ? P : P * P); };
    auto contract = [&](const Number* M, int ax, bool transpose,
                        const Number* src, Number* dst) {
        const int st = stride(ax);
        for (int o = 0; o < NQ; ++o) {
            const int oax = (o / st) % P;
            const int base = o - oax * st;
            Number acc = Number(0);
            for (int k = 0; k < P; ++k) {
                const Number m = transpose ? M[k * P + oax] : M[oax * P + k];
                acc += m * src[base + k * st];
            }
            dst[o] = acc;
        }
    };
    Number bufA[NQ], bufB[NQ];
    auto sweep = [&](const Number* source, int dir, bool transpose, Number* result) {
        const Number* src = source;
        for (int ax = 0; ax < dim; ++ax) {
            Number* dst = (ax % 2 == 0) ? bufA : bufB;
            contract((ax == dir) ? sg : sv, ax, transpose, src, dst);
            src = dst;
        }
        for (int i = 0; i < NQ; ++i) result[i] = src[i];
    };

    Number gref[n_comp][dim][NQ], ucomp[NQ];
    for (int c = 0; c < n_comp; ++c) {
        for (int i = 0; i < NQ; ++i) ucomp[i] = u[i][c];
        for (int d = 0; d < dim; ++d) sweep(ucomp, d, false, gref[c][d]);
    }

    Number rflux[n_comp][dim][NQ];
    for (int q = 0; q < NQ; ++q) {
        const Number* Jinv = geom + (e * NQ + q) * GS;   
        const Number JxW   = geom[(e * NQ + q) * GS + dim * dim];
        Number gu[n_comp][dim];
        for (int c = 0; c < n_comp; ++c)
            for (int ee = 0; ee < dim; ++ee) {
                Number s = Number(0);
                for (int d = 0; d < dim; ++d) s += Jinv[d * dim + ee] * gref[c][d][q];
                gu[c][ee] = s;
            }
        Number S[n_comp][dim];
        flux(gu, S);
        for (int c = 0; c < n_comp; ++c)
            for (int d = 0; d < dim; ++d) {
                Number s = Number(0);
                for (int ee = 0; ee < dim; ++ee) s += Jinv[d * dim + ee] * S[c][ee];
                rflux[c][d][q] = s * JxW;
            }
    }

    Number contrib[ND][n_comp], tmp[NQ];
    for (int i = 0; i < ND; ++i)
        for (int c = 0; c < n_comp; ++c) contrib[i][c] = Number(0);
    for (int c = 0; c < n_comp; ++c)
        for (int d = 0; d < dim; ++d) {
            sweep(rflux[c][d], d, true, tmp);
            for (int i = 0; i < ND; ++i) contrib[i][c] += tmp[i];
        }
    for (int i = 0; i < ND; ++i)
        for (int c = 0; c < n_comp; ++c)
            atomicAdd(&out[node[i] * n_comp + c], scale * contrib[i][c]);
}

template <int degree, typename Number>
inline void build_sf_1d_tables(std::vector<Number>& Sv, std::vector<Number>& Sg)
{
    constexpr int P = degree + 1;
    double nd[P];
    for (int i = 0; i < P; ++i) nd[i] = double(i) / degree;
    dealii::QGauss<1> g(P);
    Sv.assign(P * P, Number(0));
    Sg.assign(P * P, Number(0));
    for (int q = 0; q < P; ++q) {
        const double x = g.point(q)[0];
        for (int i = 0; i < P; ++i) {
            double val = 1.0, der = 0.0;
            for (int m = 0; m < P; ++m) { if (m == i) continue; val *= (x - nd[m]) / (nd[i] - nd[m]); }
            for (int j = 0; j < P; ++j) {
                if (j == i) continue;
                double t = 1.0 / (nd[i] - nd[j]);
                for (int m = 0; m < P; ++m) { if (m == i || m == j) continue; t *= (x - nd[m]) / (nd[i] - nd[m]); }
                der += t;
            }
            Sv[q * P + i] = Number(val);
            Sg[q * P + i] = Number(der);
        }
    }
}

}  

#endif  
