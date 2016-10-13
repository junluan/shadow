#include "shadow/util/blas.hpp"

namespace Blas {

#if defined(USE_CUDA)
#include "cublas_v2.h"

static cublasHandle_t cublas_handle_ = nullptr;

void Setup() { cublasCreate(&cublas_handle_); }

void Release() {
  if (cublas_handle_ != nullptr) cublasDestroy(cublas_handle_);
}

__global__ void SetArrayKernel(float *data, int count, float value) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= count) return;

  data[globalid] = value;
}

template <typename T>
void SetArray(T *data, int count, float value) {
  SetArrayKernel<<<Kernel::GridDim(count), BLOCK>>>(data, count, value);
  Kernel::CheckError(cudaPeekAtLastError());
}

__global__ void SetArrayRepeatKernel(float *data, int offset, int N,
                                     int value_size, const float *value) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N * value_size) return;

  int value_index = globalid / N;
  data[offset + globalid] = value[value_index];
}

template <typename T>
void SetArrayRepeat(T *data, int offset, int N, int value_size,
                    const T *value) {
  SetArrayRepeatKernel<<<Kernel::GridDim(N * value_size), BLOCK>>>(
      data, offset, N, value_size, value);
  Kernel::CheckError(cudaPeekAtLastError());
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  cublasScopy(cublas_handle_, N, X, incx, Y + offset, incy);
}

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *bufA,
               int lda, const T *bufB, int ldb, float BETA, T *bufC, int offset,
               int ldc) {
  cublasOperation_t transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(cublas_handle_, transA, transB, N, M, K, &ALPHA, bufB, ldb, bufA,
              lda, &BETA, bufC + offset, ldc);
}

// Explicit instantiation
template void SetArray<float>(float *data, int count, float value);
template void SetArrayRepeat<float>(float *data, int offset, int N,
                                    int value_size, const float *value);
template void BlasCopy<float>(int N, const float *X, int incx, float *Y,
                              int offset, int incy);
template void BlasSGemm<float>(int TA, int TB, int M, int N, int K, float ALPHA,
                               const float *A, int lda, const float *B, int ldb,
                               float BETA, float *C, int offset, int ldc);

#endif

}  // namespace Blas
