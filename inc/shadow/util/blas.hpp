#ifndef SHADOW_UTIL_BLAS_HPP
#define SHADOW_UTIL_BLAS_HPP

#include "shadow/kernel.hpp"

class Blas {
public:
  static void BlasCopy(int N, const float *X, int INCX, float *Y, int INCY);
  static void BlasAxpy(int N, float ALPHA, const float *X, int INCX, float *Y,
                       int INCY);
  static void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                        const float *A, int lda, const float *B, int ldb,
                        float BETA, float *C, int ldc);

#ifdef USE_CUDA
  static void CUDABlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                            const float *bufA, int lda, const float *bufB,
                            int ldb, float BETA, float *bufC, int offset,
                            int ldc);
#endif

#ifdef USE_CL
  static void CLBlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                          const cl_mem bufA, int lda, const cl_mem bufB,
                          int ldb, float BETA, cl_mem bufC, int offset,
                          int ldc);
#endif
};

#endif // SHADOW_UTIL_BLAS_HPP
