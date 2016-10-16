#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

namespace Blas {

void Setup();
void Release();

template <typename T>
void SetArray(T *y, int n, float value);

template <typename T>
void SetArrayRepeat(T *y, int offy, int n, int value_size, const T *value);

template <typename T>
void PowArray(const T *x, int n, float alpha, T *y);

template <typename T>
void ScaleArray(const T *x, int n, float alpha, T *y);

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x);

template <typename T>
void BlasScopy(int n, const T *x, int incx, T *y, int offy, int incy);

template <typename T>
void BlasSasum(int n, const T *x, float *y);

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, const T *x,
               float beta, T *y);

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               const T *B, float beta, T *C, int offc);

}  // namespace Blas

#endif  // SHADOW_UTIL_BLAS_HPP
