#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

namespace Blas {

void Setup();
void Release();

template <typename T>
void Set(int n, float val, T *y, int offy);

template <typename T>
void SetRepeat(int n, const T *val, int val_size, T *y, int offy);

template <typename T>
void Add(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);

template <typename T>
void Sub(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);

template <typename T>
void Mul(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);

template <typename T>
void Div(int n, const T *a, int offa, const T *b, int offb, T *y, int offy);

template <typename T>
void Square(int n, const T *a, int offa, T *y, int offy);

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy);

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy);

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx);

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy);

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y);

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy);

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC);

}  // namespace Blas

#endif  // SHADOW_UTIL_BLAS_HPP
