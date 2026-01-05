cmake_minimum_required(VERSION 3.10)
project(gpu_router_cpu_ref LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

include_directories(include)

add_executable(gpu_router
    source/main.cpp
    source/packet_generator.cpp
    source/forward_cpu.cpp
    source/aos_to_soa.cpp
)

target_link_libraries(gpu_router PUBLIC OpenMP::OpenMP_CXX)
