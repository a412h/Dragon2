#ifndef ATOMIC_OPERATIONS_CUH
#define ATOMIC_OPERATIONS_CUH

#include <cuda_runtime.h>

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
                         __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));
    return old;
}

// Atomic min wrapper for both float and double
__device__ __forceinline__ void atomicMinNumber(float* addr, float value) {
    atomicMinFloat(addr, value);
}

__device__ __forceinline__ void atomicMinNumber(double* addr, double value) {
    // Not implemented yet for double
}

__device__ __forceinline__ void atomicMin_custom(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
            __float_as_int(fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
}

__device__ __forceinline__ void atomicMin_custom(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

#endif // ATOMIC_OPERATIONS_CUH