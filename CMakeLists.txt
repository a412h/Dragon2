cmake_minimum_required(VERSION 3.18)

SET(PROJECT_NAME solver_ns)
SET(SRCS ../src/)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wfatal-errors")

# SPECIFIC ARCHITECTURE
set(CMAKE_CUDA_ARCHITECTURES 89)
message(STATUS "Compiling for architecture 89")

find_package(deal.II 9.3 REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(
    ${PROJECT_NAME}
    ${SRCS}main.cpp
    ${SRCS}cuda_time_loop.cu 
    ${SRCS}hyperbolic_kernels.cu
)

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Optimized for RTX 40xx
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --gpu-architecture=sm_89  # RTX 40xx architecture
        --threads 256             # 256 threads per block
        --maxrregcount=64         
        -Xptxas=-O3
        -Xcompiler=-O3,-march=native,-ffast-math,-funroll-loops
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3
        -march=native
        -ffast-math
        -funroll-loops
        -fopenmp
    >
)

target_link_libraries(
    ${PROJECT_NAME}
    ${DEAL_II_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX
)

deal_ii_setup_target(${PROJECT_NAME})
