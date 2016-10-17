#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

namespace Blas {

void Setup();
void Release();

template <typename T>
void Set(int n, float val, T *y);

template <typename T>
void SetRepeat(int n, const T *val, int val_size, T *y, int offy);

template <typename T>
void Add(int n, const T *a, const T *b, T *y);

template <typename T>
void Sub(int n, const T *a, const T *b, T *y);

template <typename T>
void Mul(int n, const T *a, const T *b, T *y);

template <typename T>
void Div(int n, const T *a, const T *b, T *y);

template <typename T>
void Pow(int n, const T *a, float alpha, T *y);

template <typename T>
void Scale(int n, float alpha, const T *x, T *y);

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
