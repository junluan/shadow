#ifndef SHADOW_BLAS_H
#define SHADOW_BLAS_H

#include "kernel.h"

class Blas {
public:
  static void BlasCopy(int N, float *X, int INCX, float *Y, int INCY);
  static void BlasAxpy(int N, float ALPHA, float *X, int INCX, float *Y,
                       int INCY);
  static void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                        float *A, int lda, float *B, int ldb, float BETA,
                        float *C, int ldc);

#ifdef USE_CUDA
  static void CUDABlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                            float *bufA, int lda, float *bufB, int ldb,
                            float BETA, float *bufC, int offset, int ldc);
#endif

#ifdef USE_CL
  static void CLBlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                          const cl_mem bufA, int lda, const cl_mem bufB,
                          int ldb, float BETA, cl_mem bufC, int offset,
                          int ldc);
#endif
};

#endif // SHADOW_BLAS_H
