cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES "61;75;80;86")

# Usare GCC 11 per CUDA 11.8
set(CMAKE_CUDA_HOST_COMPILER gcc-11)

project(gpu_router_cpu_ref LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

include_directories(include)

add_executable(gpu_router
    source/main.cpp
    source/packet_generator.cpp
    source/forward_cpu.cpp
    source/aos_to_soa.cpp
    source/bloom_filter.cpp
    source/gpu_runtime.cu
    source/gpu_forward.cu
    source/gpu_runtime_bloom.cu
    source/gpu_forward_bloom.cu
)

# Permetti GCC 13 con CUDA 11 (a tuo rischio)
target_compile_options(gpu_router PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--allow-unsupported-compiler>
)

target_link_libraries(gpu_router PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(gpu_router PRIVATE CUDA::cudart)
target_include_directories(gpu_router PRIVATE ${CUDAToolkit_INCLUDE_DIRS})