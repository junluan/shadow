#ifndef SHADOW_BLAS_H
#define SHADOW_BLAS_H

#ifdef USE_CL
#include "cl.h"
#endif

class Blas {
public:
  Blas();
  ~Blas();

  static void BlasCopy(int N, float *X, int INCX, float *Y, int INCY);
  static void BlasAxpy(int N, float ALPHA, float *X, int INCX, float *Y,
                       int INCY);

  static void CLBlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                          float *A, int lda, float *B, int ldb, float BETA,
                          float *C, int ldc);
  static void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                        float *A, int lda, float *B, int ldb, float BETA,
                        float *C, int ldc);

#ifdef USE_CL
  static void CLBlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                          const cl_mem &bufA, int lda, const cl_mem &bufB,
                          int ldb, float BETA, cl_mem &bufC, int offset,
                          int ldc);
#endif

private:
  static void SGemmNN(int M, int N, int K, float ALPHA, float *A, int lda,
                      float *B, int ldb, float *C, int ldc);
  static void SGemmNT(int M, int N, int K, float ALPHA, float *A, int lda,
                      float *B, int ldb, float *C, int ldc);
  static void SGemmTN(int M, int N, int K, float ALPHA, float *A, int lda,
                      float *B, int ldb, float *C, int ldc);
  static void SGemmTT(int M, int N, int K, float ALPHA, float *A, int lda,
                      float *B, int ldb, float *C, int ldc);
};

#endif // SHADOW_BLAS_H
