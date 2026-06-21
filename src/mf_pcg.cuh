#ifndef MF_PCG_CUH
#define MF_PCG_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <type_traits>
#include <cmath>

namespace mfcg {

template <typename Number>
struct blas1;

template <>  
struct blas1<float> { 
    static void dot(cublasHandle_t h, int n, const float* x, const float* y, float* res)  
    {
        cublasSdot(h, n, x, 1, y, 1, res);
    }
    static void axpy(cublasHandle_t h, int n, const float* a, const float* x, float* y)  
    {
        cublasSaxpy(h, n, a, x, 1, y, 1);
    }
    static void nrm2(cublasHandle_t h, int n, const float* x, float* res)  
    {
        cublasSnrm2(h, n, x, 1, res);
    }
};

template <>  
struct blas1<double> {
    static void dot(cublasHandle_t h, int n, const double* x, const double* y, double* res)
    {
        cublasDdot(h, n, x, 1, y, 1, res);
    }
    static void axpy(cublasHandle_t h, int n, const double* a, const double* x, double* y)
    {
        cublasDaxpy(h, n, a, x, 1, y, 1);
    }
    static void nrm2(cublasHandle_t h, int n, const double* x, double* res)
    {
        cublasDnrm2(h, n, x, 1, res);
    }
};

template <typename Number>
__global__ void k_alpha(Number* alpha, Number* negative, const Number* numerator, const Number* denominator)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        const Number v = (*denominator != Number(0)) ? (*numerator) / (*denominator) : Number(0);
        *alpha = v;
        *negative = -v;
    }
}

template <typename Number>
__global__ void k_beta(Number* beta, const Number* numerator, const Number* denominator)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
        *beta = (*denominator != Number(0)) ? (*numerator) / (*denominator) : Number(0);
}

template <typename Number>
__global__ void k_xpby(Number* p, const Number* z, const Number* beta, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = z[i] + (*beta) * p[i];
}

template <typename Number>
__global__ void k_vmul(const Number* inv, const Number* src, Number* dst, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = inv[i] * src[i];
}

template <typename Number>
struct DiagonalPreconditioner
{
    const Number* inv = nullptr;  
    int n = 0;
    cudaStream_t stream = 0;
    void vmult(Number* dst, const Number* src) const {
        const int block_size = 256;  
        const int grid_size = (n + block_size - 1) / block_size;  
        k_vmul<Number><<<grid_size, block_size, 0, stream>>>(inv, src, dst, n);
    }
};

template <typename Number>
struct IdentityPreconditioner {  
    int n = 0;
    cudaStream_t stream = 0;
    void vmult(Number* dst, const Number* src) const {
        cudaMemcpyAsync(dst, src, n * sizeof(Number), cudaMemcpyDeviceToDevice, stream);
    }
};

template <typename Number>
class PCG {
 public:
    PCG(int n, cudaStream_t stream = 0) : n_(n), stream_(stream)
    {
        cublasCreate(&handle_);
        cublasSetStream(handle_, stream_);
        cublasSetPointerMode(handle_, CUBLAS_POINTER_MODE_DEVICE);  
        cudaMalloc(&r_,  n_ * sizeof(Number));      
        cudaMalloc(&z_,  n_ * sizeof(Number));      
        cudaMalloc(&p_,  n_ * sizeof(Number));      
        cudaMalloc(&Ap_, n_ * sizeof(Number));      
        cudaMalloc(&sc_, NSCALAR * sizeof(Number));   
        const Number neg1 = Number(-1);             
        cudaMemcpy(&sc_[NEGONE], &neg1, sizeof(Number), cudaMemcpyHostToDevice);
    }

    ~PCG() {
        cudaFree(r_); cudaFree(z_); cudaFree(p_); cudaFree(Ap_); cudaFree(sc_);
        cublasDestroy(handle_);
    }

    int size() const { return n_; }

    template <class Operator, class Preconditioner>
    int solve(const Operator& A,
              const Preconditioner& M,
              Number* x,
              const Number* b,
              Number rel_tol,
              int max_it)
    {
        using B = blas1<Number>;

        const int block_size = 256;  
        const int grid_size = (n_ + block_size - 1) / block_size;  

        const Number abs_tol = std::is_same_v<Number, float> ? Number(1e-8) : Number(1e-15);

        A.vmult(Ap_, x);
        cudaMemcpyAsync(r_, b, n_ * sizeof(Number), cudaMemcpyDeviceToDevice, stream_);
        B::axpy(handle_, n_, &sc_[NEGONE], Ap_, r_);          

        B::nrm2(handle_, n_, r_, &sc_[RR]);
        const Number r0 = fetch(&sc_[RR]);
        if (r0 < abs_tol) return 0;
        const Number target = (rel_tol * r0 < abs_tol) ? abs_tol : rel_tol * r0;

        M.vmult(z_, r_);
        cudaMemcpyAsync(p_, z_, n_ * sizeof(Number), cudaMemcpyDeviceToDevice, stream_);
        B::dot(handle_, n_, r_, z_, &sc_[RHO]);

        for (int it = 0; it < max_it; ++it) {
            A.vmult(Ap_, p_);
            B::dot(handle_, n_, p_, Ap_, &sc_[PAP]);
            k_alpha<Number><<<1, 1, 0, stream_>>>(&sc_[ALPHA], &sc_[NEG_ALPHA], &sc_[RHO], &sc_[PAP]);
            B::axpy(handle_, n_, &sc_[ALPHA],     p_,  x);      
            B::axpy(handle_, n_, &sc_[NEG_ALPHA], Ap_, r_);     

            B::nrm2(handle_, n_, r_, &sc_[RR]);
            if (fetch(&sc_[RR]) < target) return it + 1;

            M.vmult(z_, r_);
            B::dot(handle_, n_, r_, z_, &sc_[RHO_NEW]);
            k_beta<Number><<<1, 1, 0, stream_>>>(&sc_[BETA], &sc_[RHO_NEW], &sc_[RHO]);
            k_xpby<Number><<<grid_size, block_size, 0, stream_>>>(p_, z_, &sc_[BETA], n_);   
            cudaMemcpyAsync(&sc_[RHO], &sc_[RHO_NEW], sizeof(Number), cudaMemcpyDeviceToDevice, stream_);
        }
        return -1;  
    }

 private:

    Number fetch(const Number* dptr) {
        Number h;
        cudaMemcpyAsync(&h, dptr, sizeof(Number), cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);
        return h;
    }

    enum { RHO = 0, RHO_NEW, PAP, ALPHA, NEG_ALPHA, BETA, RR, NEGONE, NSCALAR };  

    int n_;
    cudaStream_t stream_;
    cublasHandle_t handle_;
    Number *r_ = nullptr, *z_ = nullptr, *p_ = nullptr, *Ap_ = nullptr;
    Number *sc_ = nullptr;  
};

}  

#endif  
