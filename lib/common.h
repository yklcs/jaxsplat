#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <stdexcept>

constexpr unsigned MAX_GRID_DIM = 512;

inline void throw_if_error(cudaError_t error) {
  if (error != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(error));
  }
}

template <typename T> struct mat3 {
  T m[3][3];

  inline __device__ mat3<T>(T m) {
    this->m[0][0] = m;
    this->m[1][0] = m;
    this->m[2][0] = m;
    this->m[0][1] = m;
    this->m[1][1] = m;
    this->m[2][1] = m;
    this->m[0][2] = m;
    this->m[1][2] = m;
    this->m[2][2] = m;
  }

  inline __device__ mat3<T>(T m00, T m10, T m20, T m01, T m11, T m21, T m02,
                            T m12, T m22) {
    this->m[0][0] = m00;
    this->m[1][0] = m10;
    this->m[2][0] = m20;
    this->m[0][1] = m01;
    this->m[1][1] = m11;
    this->m[2][1] = m21;
    this->m[0][2] = m02;
    this->m[1][2] = m12;
    this->m[2][2] = m22;
  }

  inline __device__ T *operator[](const size_t idx) { return m[idx]; }
  inline __device__ T const *operator[](const size_t idx) const {
    return m[idx];
  }

  inline __device__ mat3 matmul(mat3 const &rhs) const {
    return mat3<T>(
        m[0][0] * rhs[0][0] + m[0][1] * rhs[1][0] + m[0][2] * rhs[2][0],
        m[1][0] * rhs[0][0] + m[1][1] * rhs[1][0] + m[1][2] * rhs[2][0],
        m[2][0] * rhs[0][0] + m[2][1] * rhs[1][0] + m[2][2] * rhs[2][0],
        m[0][0] * rhs[0][1] + m[0][1] * rhs[1][1] + m[0][2] * rhs[2][1],
        m[1][0] * rhs[1][1] + m[1][1] * rhs[1][1] + m[1][2] * rhs[2][1],
        m[2][0] * rhs[2][1] + m[2][1] * rhs[1][1] + m[2][2] * rhs[2][1],
        m[0][0] * rhs[0][2] + m[0][1] * rhs[1][2] + m[0][2] * rhs[2][2],
        m[1][0] * rhs[0][2] + m[1][1] * rhs[1][2] + m[1][2] * rhs[2][2],
        m[2][0] * rhs[0][2] + m[2][1] * rhs[1][2] + m[2][2] * rhs[2][2]);
  }

  inline __device__ mat3 transpose() const {
    return mat3<T>(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2],
                   m[2][0], m[2][1], m[2][2]);
  }
};

template <typename T> struct mat2 {
  T m[2][2];

  inline __device__ mat2<T>(T val) {
    m[0][0] = val;
    m[1][0] = val;
    m[0][1] = val;
    m[1][1] = val;
  }

  inline __device__ mat2<T>(T m00, T m10, T m01, T m11) {
    this->m[0][0] = m00;
    this->m[1][0] = m10;
    this->m[0][1] = m01;
    this->m[1][1] = m11;
  }

  inline __device__ T *operator[](const size_t idx) { return m[idx]; }
  inline __device__ T const *operator[](const size_t idx) const {
    return m[idx];
  }

  inline __device__ mat2 operator-() const {
    return mat2(-m[0][0], -m[1][0], -m[0][1], -m[1][1]);
  }

  inline __device__ mat2 matmul(mat2 const &rhs) const {
    return mat2<T>(m[0][0] * rhs[0][0] + m[0][1] * rhs[1][0],
                   m[1][0] * rhs[0][0] + m[1][1] * rhs[1][0],
                   m[0][0] * rhs[0][1] + m[0][1] * rhs[1][1],
                   m[1][0] * rhs[0][1] + m[1][1] * rhs[1][1]);
  }
};
