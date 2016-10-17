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

__global__ void KernelSet(int n, float val, float *y) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[globalid] = val;
}

template <typename T>
void Set(int n, float val, T *y) {
  KernelSet<<<Kernel::GridDim(n), BLOCK>>>(n, val, y);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelSetRepeat(int n, const float *val, int val_size, float *y,
                                int offy) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  int val_index = globalid / (n / val_size);
  y[offy + globalid] = val[val_index];
}

template <typename T>
void SetRepeat(int n, const T *val, int val_size, T *y, int offy) {
  KernelSetRepeat<<<Kernel::GridDim(n), BLOCK>>>(n, val, val_size, y, offy);
  CheckError(cudaPeekAtLastError());
}

#define BINARY_FUNC(name, operation)                                          \
  __global__ void Kernel##name(int n, const float *a, const float *b,         \
                               float *y) {                                    \
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; \
    if (i >= n) return;                                                       \
    operation;                                                                \
  }

BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BINARY_FUNC(Div, y[i] = a[i] / b[i]);

template <typename T>
void Add(int n, const T *a, const T *b, T *y) {
  KernelAdd<<<Kernel::GridDim(n), BLOCK>>>(n, a, b, y);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Sub(int n, const T *a, const T *b, T *y) {
  KernelSub<<<Kernel::GridDim(n), BLOCK>>>(n, a, b, y);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Mul(int n, const T *a, const T *b, T *y) {
  KernelMul<<<Kernel::GridDim(n), BLOCK>>>(n, a, b, y);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Div(int n, const T *a, const T *b, T *y) {
  KernelDiv<<<Kernel::GridDim(n), BLOCK>>>(n, a, b, y);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelPow(int n, const float *a, float alpha, float *y) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[globalid] = std::pow(a[globalid], alpha);
}

template <typename T>
void Pow(int n, const T *a, float alpha, T *y) {
  KernelPow<<<Kernel::GridDim(n), BLOCK>>>(n, a, alpha, y);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Scale(int n, float alpha, const T *x, T *y) {
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
template void Set<float>(int n, float val, float *y);
template void SetRepeat<float>(int n, const float *val, int val_size, float *y,
                               int offy);
template void Add<float>(int n, const float *a, const float *b, float *y);
template void Sub<float>(int n, const float *a, const float *b, float *y);
template void Mul<float>(int n, const float *a, const float *b, float *y);
template void Div<float>(int n, const float *a, const float *b, float *y);
template void Pow<float>(int n, const float *a, float alpha, float *y);
template void Scale<float>(int n, float alpha, const float *x, float *y);

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
