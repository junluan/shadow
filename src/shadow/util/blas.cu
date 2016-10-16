#include "shadow/util/blas.hpp"
#include "shadow/util/util.hpp"

namespace Blas {

#if defined(USE_CUDA)
#include "cublas_v2.h"

static cublasHandle_t cublas_handle_ = nullptr;

void Setup() { cublasCreate(&cublas_handle_); }

void Release() {
  if (cublas_handle_ != nullptr) cublasDestroy(cublas_handle_);
}

__global__ void SetArrayKernel(float *y, int n, float value) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[globalid] = value;
}

template <typename T>
void SetArray(T *y, int n, float value) {
  SetArrayKernel<<<Kernel::GridDim(n), BLOCK>>>(y, n, value);
  CheckError(cudaPeekAtLastError());
}

__global__ void SetArrayRepeatKernel(float *y, int offy, int n, int value_size,
                                     const float *value) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n * value_size) return;

  int value_index = globalid / n;
  y[offy + globalid] = value[value_index];
}

template <typename T>
void SetArrayRepeat(T *y, int offy, int n, int value_size, const T *value) {
  SetArrayRepeatKernel<<<Kernel::GridDim(n * value_size), BLOCK>>>(
      y, offy, n, value_size, value);
  CheckError(cudaPeekAtLastError());
}

__global__ void PowArrayKernel(const float *x, int n, float alpha, float *y) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[globalid] = std::pow(x[globalid], alpha);
}

template <typename T>
void PowArray(const T *x, int n, float alpha, T *y) {
  PowArrayKernel<<<Kernel::GridDim(n), BLOCK>>>(x, n, alpha, y);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void ScaleArray(const T *x, int n, float alpha, T *y) {
  BlasScopy(n, x, 1, y, 0, 1);
  BlasSscal(n, alpha, y);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x) {
  cublasSscal(cublas_handle_, n, &alpha, x, 1);
}

template <typename T>
void BlasScopy(int n, const T *x, int incx, T *y, int offy, int incy) {
  cublasScopy(cublas_handle_, n, x, incx, y + offy, incy);
}

template <typename T>
void BlasSasum(int n, const T *x, float *y) {
  cublasSasum(cublas_handle_, n, x, 1, y);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, const T *x,
               float beta, T *y) {
  cublasOperation_t transA = TA ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemv(cublas_handle_, transA, N, M, &alpha, A, N, x, 1, &beta, y, 1);
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               const T *B, float beta, T *C, int offc) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  cublasOperation_t transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(cublas_handle_, transB, transA, N, M, K, &alpha, B, ldb, A, lda,
              &beta, C + offc, N);
}

// Explicit instantiation
template void SetArray<float>(float *y, int n, float value);
template void SetArrayRepeat<float>(float *y, int offy, int n, int value_size,
                                    const float *value);
template void PowArray(const float *x, int n, float alpha, float *y);
template void ScaleArray<float>(const float *x, int n, float alpha, float *y);

// Level 1
template void BlasSscal<float>(int n, float alpha, float *x);
template void BlasScopy<float>(int n, const float *x, int incx, float *y,
                               int offy, int incy);
template void BlasSasum<float>(int n, const float *x, float *y);

// Level 2
template void BlasSgemv<float>(int TA, int M, int N, float alpha,
                               const float *A, const float *x, float beta,
                               float *y);

// Level 3
template void BlasSgemm<float>(int TA, int TB, int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C, int offc);

#endif

}  // namespace Blas
