cmake_minimum_required(VERSION 3.23)
project(jaxsplatlib LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES native)

find_package(pybind11 CONFIG REQUIRED)

include_directories(${CMAKE_CURRENT_LIST_DIR}/lib)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

pybind11_add_module(
    jaxsplatlib
    ${CMAKE_CURRENT_LIST_DIR}/lib/ffi.cxx
    ${CMAKE_CURRENT_LIST_DIR}/lib/ops.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/forward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/backward.cu
    ${CMAKE_CURRENT_LIST_DIR}/lib/kernels/other.cu
)

install(TARGETS jaxsplatlib LIBRARY DESTINATION .)
