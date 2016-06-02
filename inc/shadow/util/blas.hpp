#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

class Blas {
public:
  static void SetArray(int N, float value, BType *out_data);
  static void SetArrayRepeat(int N, const BType *value, int value_size,
                             BType *out_data);
  static void BlasCopy(int N, const BType *X, int incx, BType *Y, int incy);
  static void BlasAxpy(int N, float ALPHA, const float *X, int INCX, float *Y,
                       int INCY);
  static void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                        const BType *A, int lda, const BType *B, int ldb,
                        float BETA, BType *C, int offset, int ldc);
};

#endif // SHADOW_UTIL_BLAS_HPP
