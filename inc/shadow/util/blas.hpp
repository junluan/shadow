#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

namespace Blas {

void Setup();
void Release();

template <typename T>
void SetArray(T *data, int count, float value);

template <typename T>
void SetArrayRepeat(T *data, int offset, int N, int value_size, const T *value);

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy);

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *A,
               int lda, const T *B, int ldb, float BETA, T *C, int offset,
               int ldc);

}  // namespace Blas

#endif  // SHADOW_UTIL_BLAS_HPP
