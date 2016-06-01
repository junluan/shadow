#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

class Blas {
public:
  static void BlasCopy(int N, const float *X, int INCX, float *Y, int INCY);
  static void BlasAxpy(int N, float ALPHA, const float *X, int INCX, float *Y,
                       int INCY);
  static void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                        const BType *A, int lda, const BType *B, int ldb,
                        float BETA, BType *C, int offset, int ldc);
};

#endif // SHADOW_UTIL_BLAS_HPP
