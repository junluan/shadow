#include "shadow/util/blas.hpp"

#include <algorithm>

namespace Blas {

#if defined(USE_CUDA)
template <typename T>
void SetArray(int N, float value, T *out_data) {
  Kernel::SetArray(N, value, out_data);
}

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset) {
  Kernel::SetArrayRepeat(N, value, value_size, out_data, offset);
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  cublasScopy(Kernel::cublas_handle_, N, X, incx, Y + offset, incy);
}

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *bufA,
               int lda, const T *bufB, int ldb, float BETA, T *bufC, int offset,
               int ldc) {
  cublasOperation_t transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(Kernel::cublas_handle_, transA, transB, N, M, K, &ALPHA, bufB,
              ldb, bufA, lda, &BETA, bufC + offset, ldc);
}

// Explicit instantiation
template void SetArray<float>(int N, float value, float *out_data);
template void SetArrayRepeat<float>(int N, const float *value, int value_size,
                                    float *out_data, int offset);
template void BlasCopy<float>(int N, const float *X, int incx, float *Y,
                              int offset, int incy);
template void BlasSGemm<float>(int TA, int TB, int M, int N, int K, float ALPHA,
                               const float *A, int lda, const float *B, int ldb,
                               float BETA, float *C, int offset, int ldc);

#elif defined(USE_CL)
template <typename T>
void SetArray(int N, float value, T *out_data) {
  Kernel::SetArray(N, value, out_data);
}

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset) {
  Kernel::SetArrayRepeat(N, value, value_size, out_data, offset);
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  clblasScopy(N, *X, 0, incx, *Y, offset, incy, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
}

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *bufA,
               int lda, const T *bufB, int ldb, float BETA, T *bufC, int offset,
               int ldc) {
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, ALPHA, *bufA, 0, lda,
              *bufB, 0, ldb, BETA, *bufC, offset, ldc, 1, Kernel::easyCL->queue,
              0, nullptr, nullptr);
}

// Explicit instantiation
template void SetArray<cl_mem>(int N, float value, cl_mem *out_data);
template void SetArrayRepeat<cl_mem>(int N, const cl_mem *value, int value_size,
                                     cl_mem *out_data, int offset);
template void BlasCopy<cl_mem>(int N, const cl_mem *X, int incx, cl_mem *Y,
                               int offset, int incy);
template void BlasSGemm<cl_mem>(int TA, int TB, int M, int N, int K,
                                float ALPHA, const cl_mem *A, int lda,
                                const cl_mem *B, int ldb, float BETA, cl_mem *C,
                                int offset, int ldc);

#else
template <typename T>
void SetArray(int N, float value, T *out_data) {
  std::fill(out_data, out_data + N, value);
}

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset) {
  for (int i = 0; i < value_size; ++i) {
    T *out_data_offset = out_data + offset + i * N;
    std::fill(out_data_offset, out_data_offset + N, value[i]);
  }
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  for (int i = 0; i < N; ++i) {
    Y[offset + i * incy] = X[i * incx];
  }
}

inline void SGemmNN(int M, int N, int K, float ALPHA, const float *A, int lda,
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

inline void SGemmNT(int M, int N, int K, float ALPHA, const float *A, int lda,
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

inline void SGemmTN(int M, int N, int K, float ALPHA, const float *A, int lda,
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

inline void SGemmTT(int M, int N, int K, float ALPHA, const float *A, int lda,
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

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *A,
               int lda, const T *B, int ldb, float BETA, T *C, int offset,
               int ldc) {
  float *C_off = C + offset;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_off[i * ldc + j] *= BETA;
    }
  }
  if (!TA && !TB)
    SGemmNN(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else if (TA && !TB)
    SGemmTN(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else if (!TA && TB)
    SGemmNT(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else
    SGemmTT(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
}

// Explicit instantiation
template void SetArray<float>(int N, float value, float *out_data);
template void SetArrayRepeat<float>(int N, const float *value, int value_size,
                                    float *out_data, int offset);
template void BlasCopy<float>(int N, const float *X, int incx, float *Y,
                              int offset, int incy);
template void BlasSGemm<float>(int TA, int TB, int M, int N, int K, float ALPHA,
                               const float *A, int lda, const float *B, int ldb,
                               float BETA, float *C, int offset, int ldc);
#endif

}  // namespace Blas
