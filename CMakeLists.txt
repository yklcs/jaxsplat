cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_FLAGS -std=c++17) # must be set before CUDA detection
project(_jaxsplat LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

set(PYBIND11_NEWPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)

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

if(DEFINED SKBUILD)
    set(JAXSPLAT_INSTALL_DEST ${SKBUILD_PLATLIB_DIR})
else()
    set(JAXSPLAT_INSTALL_DEST ${CMAKE_CURRENT_LIST_DIR})
endif()

message(STATUS "Installing _jaxsplat to ${JAXSPLAT_INSTALL_DEST}")
install(TARGETS _jaxsplat LIBRARY DESTINATION ${JAXSPLAT_INSTALL_DEST})
