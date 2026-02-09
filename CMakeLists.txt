cmake_minimum_required(VERSION 3.18)
SET(PROJECT_NAME solver_ns)
SET(SRCS ${CMAKE_SOURCE_DIR}/src/)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfatal-errors")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wfatal-errors")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "" OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
    set(CMAKE_CUDA_ARCHITECTURES 86 89)
endif()
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

set(DEAL_II_DIR "" CACHE PATH "Path to deal.II installation directory")
if(DEAL_II_DIR STREQUAL "")
    message(FATAL_ERROR "DEAL_II_DIR is not set. Please set it to your deal.II installation path, e.g.:\n  cmake -DDEAL_II_DIR=/path/to/dealii/install ..")
endif()
set(CMAKE_PREFIX_PATH "${DEAL_II_DIR}" CACHE PATH "Prefix path for finding packages" FORCE)
message(STATUS "Using deal.II from: ${DEAL_II_DIR}")

find_package(deal.II 9.3 REQUIRED PATHS ${DEAL_II_DIR} NO_DEFAULT_PATH)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(
    ${PROJECT_NAME}
    ${SRCS}main.cpp
    ${SRCS}cuda_time_loop.cu
    ${SRCS}hyperbolic_kernels.cu
)

target_include_directories(${PROJECT_NAME} PRIVATE ${SRCS})

set_target_properties(
    ${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --threads 256
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
