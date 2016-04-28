#include "blas.hpp"

void Blas::BlasCopy(int N, const float *X, int INCX, float *Y, int INCY) {
  for (int i = 0; i < N; ++i)
    Y[i * INCY] = X[i * INCX];
}

void Blas::BlasAxpy(int N, float ALPHA, const float *X, int INCX, float *Y,
                    int INCY) {
  for (int i = 0; i < N; ++i)
    Y[i * INCY] += ALPHA * X[i * INCX];
}

void SGemmNN(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void SGemmNT(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void SGemmTN(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void SGemmTT(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void Blas::BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                     const float *A, int lda, const float *B, int ldb,
                     float BETA, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  if (!TA && !TB)
    SGemmNN(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (TA && !TB)
    SGemmTN(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else if (!TA && TB)
    SGemmNT(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  else
    SGemmTT(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

#ifdef USE_CUDA
void Blas::CUDABlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                         const float *bufA, int lda, const float *bufB, int ldb,
                         float BETA, float *bufC, int offset, int ldc) {
  cublasOperation_t transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(CUDA::BlasHandle, transA, transB, N, M, K, &ALPHA, bufB, ldb,
              bufA, lda, &BETA, bufC + offset, ldc);
}
#endif

#ifdef USE_CL
#include <clBLAS.h>
void Blas::CLBlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA,
                       const cl_mem bufA, int lda, const cl_mem bufB, int ldb,
                       float BETA, cl_mem bufC, int offset, int ldc) {
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, ALPHA, bufA, 0, lda,
              bufB, 0, ldb, BETA, bufC, offset, ldc, 1, CL::easyCL->queue, 0,
              NULL, NULL);
}
#endif
