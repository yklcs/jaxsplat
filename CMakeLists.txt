cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_FLAGS -std=c++17)
project(_jaxsplat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(pybind11 CONFIG REQUIRED)
find_package(glm REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(
    _jaxsplat
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/ops.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/forward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/backward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/other.cu
)

target_compile_options(_jaxsplat PUBLIC -lineinfo -g3)

install(TARGETS _jaxsplat LIBRARY DESTINATION jaxsplat)
