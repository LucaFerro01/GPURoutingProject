cmake_minimum_required(VERSION 3.10)
project(gpu_router_cpu_ref LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_executable(gpu_router
    src/main.cpp
    src/packet_generator.cpp
    src/forward_cpu.cpp
)
