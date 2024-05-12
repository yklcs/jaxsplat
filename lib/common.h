#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <iostream>
#include <stdexcept>

constexpr unsigned MAX_GRID_DIM = 256;

#define CUDA_THROW_IF_ERR(err)                                                 \
    do {                                                                       \
        cuda_throw_if_err((err), __FILE__, __LINE__);                          \
    } while (false)

inline void cuda_throw_if_err(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        std::cerr << "Encountered CUDA error in " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        throw std::runtime_error(cudaGetErrorString(error));
    }
}

template <typename T> struct mat3 {
    T m[3][3];

    // Fills diags with val, rest with 0.
    inline __device__ mat3<T>(T m) {
        this->m[0][0] = m;
        this->m[0][1] = static_cast<T>(0.f);
        this->m[0][2] = static_cast<T>(0.f);
        this->m[1][0] = static_cast<T>(0.f);
        this->m[1][1] = m;
        this->m[1][2] = static_cast<T>(0.f);
        this->m[2][0] = static_cast<T>(0.f);
        this->m[2][1] = static_cast<T>(0.f);
        this->m[2][2] = m;
    }

    inline __device__
    mat3<T>(T m00, T m10, T m20, T m01, T m11, T m21, T m02, T m12, T m22) {
        this->m[0][0] = m00;
        this->m[0][1] = m10;
        this->m[0][2] = m20;
        this->m[1][0] = m01;
        this->m[1][1] = m11;
        this->m[1][2] = m21;
        this->m[2][0] = m02;
        this->m[2][1] = m12;
        this->m[2][2] = m22;
    }

    inline __device__ T *operator[](const size_t idx) { return m[idx]; }
    inline __device__ T const *operator[](const size_t idx) const {
        return m[idx];
    }

    inline __device__ mat3 operator*(mat3 const &rhs) const {
        return mat3<T>(
            m[0][0] * rhs[0][0] + m[1][0] * rhs[0][1] + m[2][0] * rhs[0][2],
            m[0][1] * rhs[0][0] + m[1][1] * rhs[0][1] + m[2][1] * rhs[0][2],
            m[0][2] * rhs[0][0] + m[1][2] * rhs[0][1] + m[2][2] * rhs[0][2],
            m[0][0] * rhs[1][0] + m[1][0] * rhs[1][1] + m[2][0] * rhs[1][2],
            m[0][1] * rhs[1][0] + m[1][1] * rhs[1][1] + m[2][1] * rhs[1][2],
            m[0][2] * rhs[1][0] + m[1][2] * rhs[1][1] + m[2][2] * rhs[1][2],
            m[0][0] * rhs[2][0] + m[1][0] * rhs[2][1] + m[2][0] * rhs[2][2],
            m[0][1] * rhs[2][0] + m[1][1] * rhs[2][1] + m[2][1] * rhs[2][2],
            m[0][2] * rhs[2][0] + m[1][2] * rhs[2][1] + m[2][2] * rhs[2][2]
        );
    }

    inline __device__ mat3 operator+(mat3 const &rhs) const {
        return mat3<T>(
            m[0][0] + rhs[0][0],
            m[0][1] + rhs[0][1],
            m[0][2] + rhs[0][2],
            m[1][0] + rhs[1][0],
            m[1][1] + rhs[1][1],
            m[1][2] + rhs[1][2],
            m[2][0] + rhs[2][0],
            m[2][1] + rhs[2][1],
            m[2][2] + rhs[2][2]
        );
    }

    inline __device__ mat3 transpose() const {
        return mat3<T>(
            m[0][0],
            m[1][0],
            m[2][0],
            m[0][1],
            m[1][1],
            m[2][1],
            m[0][2],
            m[1][2],
            m[2][2]
        );
    }
};

template <typename T> struct mat2 {
    T m[2][2];

    // Fills diags with val, rest with 0.
    inline __device__ mat2<T>(T val) {
        m[0][0] = val;
        m[0][1] = static_cast<T>(0.f);
        m[1][0] = static_cast<T>(0.f);
        m[1][1] = val;
    }

    inline __device__ mat2<T>(T m00, T m10, T m01, T m11) {
        this->m[0][0] = m00;
        this->m[0][1] = m10;
        this->m[1][0] = m01;
        this->m[1][1] = m11;
    }

    inline __device__ T *operator[](const size_t idx) { return m[idx]; }
    inline __device__ T const *operator[](const size_t idx) const {
        return m[idx];
    }

    inline __device__ mat2 operator-() const {
        return mat2(-m[0][0], -m[0][1], -m[1][0], -m[1][1]);
    }

    inline __device__ mat2 operator*(mat2 const &rhs) const {
        return mat2<T>(
            m[0][0] * rhs[0][0] + m[1][0] * rhs[0][1],
            m[0][1] * rhs[0][0] + m[1][1] * rhs[0][1],
            m[0][0] * rhs[1][0] + m[1][0] * rhs[1][1],
            m[0][1] * rhs[1][0] + m[1][1] * rhs[1][1]
        );
    }
};
