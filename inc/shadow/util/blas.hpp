#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

namespace Blas {

template <typename T>
void SetArray(int N, float value, T *out_data);

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset);

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy);

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *A,
               int lda, const T *B, int ldb, float BETA, T *C, int offset,
               int ldc);

}  // namespace Blas

#endif  // SHADOW_UTIL_BLAS_HPP
