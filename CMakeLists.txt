cmake_minimum_required(VERSION 3.18)

# Project configuration
set(PROJECT_NAME solver_ns)
set(SRCS ../src/)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.3/bin/nvcc" CACHE FILEPATH "CUDA compiler")
set(CUDAToolkit_ROOT "/usr/local/cuda-13.3" CACHE PATH "CUDA Toolkit" FORCE)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Build configuration
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wfatal-errors")

# RTX 5000 configuration
set(CMAKE_CUDA_ARCHITECTURES 89)
message(STATUS "Compiling for architecture 89")

# Force CUDA 13.3 toolkit path
set(CUDAToolkit_ROOT "/usr/local/cuda-13.3" CACHE PATH "CUDA Toolkit" FORCE)

# deal.II configuration
set(DEAL_II_DIR "Write your path to dealii/install" CACHE PATH "deal.II directory" FORCE)
set(CMAKE_PREFIX_PATH "" CACHE PATH "Prefix path" FORCE)
set(CMAKE_SYSTEM_PREFIX_PATH "" CACHE STRING "System prefix" FORCE)

find_package(deal.II 9.3 REQUIRED 
    HINTS ${DEAL_II_DIR}
    NO_DEFAULT_PATH NO_CMAKE_SYSTEM_PATH NO_SYSTEM_ENVIRONMENT_PATH)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

# Executable
add_executable(${PROJECT_NAME}
    ${SRCS}main.cpp
    ${SRCS}cuda_time_loop.cu 
    ${SRCS}hyperbolic_kernels.cu)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_RUNTIME_LIBRARY Shared)

# Optimization flags
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3 --use_fast_math
        #--gpu-architecture=sm_100
        --gpu-architecture=sm_89
        --threads 256
        --maxrregcount=64
        -Xptxas=-O3
        --ptxas-options=-allow-expensive-optimizations=true
        -Xcompiler=-O3,-march=native,-ffast-math,-funroll-loops,-ftree-vectorize
        --extra-device-vectorization
        -DCUDA_ARCH_BLACKWELL>
    $<$<COMPILE_LANGUAGE:CXX>:
        -O3 -march=native -mtune=native
        -ffast-math -funroll-loops -ftree-vectorize
        -fopenmp -DNDEBUG>)

# Linking - explicitly link CUDA 13.3 libraries
target_link_libraries(${PROJECT_NAME}
    ${DEAL_II_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    OpenMP::OpenMP_CXX)

# Explicitly set RPATH to use CUDA 13.3 libraries
set_target_properties(${PROJECT_NAME} PROPERTIES
    BUILD_RPATH "/usr/local/cuda-13.3/lib64"
    INSTALL_RPATH "/usr/local/cuda-13.3/lib64")

deal_ii_setup_target(${PROJECT_NAME})

# Note - You do not need CUDA 13.3 necessarily, it will work with a precedent version (but not too precedent)