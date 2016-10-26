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

__global__ void KernelSet(int n, float val, float *y, int offy) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[offy + globalid] = val;
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  KernelSet<<<Kernel::GridDim(n), BLOCK>>>(n, val, y, offy);
  CheckError(cudaPeekAtLastError());
}

#define BINARY_FUNC(name, operation)                                           \
  __global__ void Kernel##name(int n, const float *a, int offa,                \
                               const float *b, int offb, float *y, int offy) { \
    int i = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;  \
    if (i >= n) return;                                                        \
    y[offy + i] = a[offa + i] operation b[offb + i];                           \
  }

BINARY_FUNC(Add, +);
BINARY_FUNC(Sub, -);
BINARY_FUNC(Mul, *);
BINARY_FUNC(Div, /);

template <typename T>
void Add(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  KernelAdd<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, b, offb, y, offy);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Sub(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  KernelSub<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, b, offb, y, offy);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Mul(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  KernelMul<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, b, offb, y, offy);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Div(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  KernelDiv<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, b, offb, y, offy);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelSquare(int n, const float *a, int offa, float *y,
                             int offy) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[offy + globalid] = a[offa + globalid] * a[offa + globalid];
}

template <typename T>
void Square(int n, const T *a, int offa, T *y, int offy) {
  KernelSquare<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, y, offy);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelPow(int n, const float *a, int offa, float alpha,
                          float *y, int offy) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= n) return;

  y[offy + globalid] = std::pow(a[offa + globalid], alpha);
}

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  KernelPow<<<Kernel::GridDim(n), BLOCK>>>(n, a, offa, alpha, y, offy);
  CheckError(cudaPeekAtLastError());
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
  BlasScopy(n, x, offx, y, offy);
  BlasSscal(n, alpha, y, offy);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
  cublasSscal(cublas_handle_, n, &alpha, x + offx, 1);
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  cublasScopy(cublas_handle_, n, x + offx, 1, y + offy, 1);
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  cublasSaxpy(cublas_handle_, n, &alpha, x + offx, 1, y + offy, 1);
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  cublasSasum(cublas_handle_, n, x + offx, 1, y);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy) {
  cublasOperation_t transA = TA ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemv(cublas_handle_, transA, N, M, &alpha, A + offA, N, x + offx, 1,
              &beta, y + offy, 1);
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  cublasOperation_t transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(cublas_handle_, transB, transA, N, M, K, &alpha, B + offB, ldb,
              A + offA, lda, &beta, C + offC, N);
}

// Explicit instantiation
template void Set<float>(int n, float val, float *y, int offy);
template void Add<float>(int n, const float *a, int offa, const float *b,
                         int offb, float *y, int offy);
template void Sub<float>(int n, const float *a, int offa, const float *b,
                         int offb, float *y, int offy);
template void Mul<float>(int n, const float *a, int offa, const float *b,
                         int offb, float *y, int offy);
template void Div<float>(int n, const float *a, int offa, const float *b,
                         int offb, float *y, int offy);
template void Square<float>(int n, const float *a, int offa, float *y,
                            int offy);
template void Pow<float>(int n, const float *a, int offa, float alpha, float *y,
                         int offy);
template void Scale<float>(int n, float alpha, const float *x, int offx,
                           float *y, int offy);

// Level 1
template void BlasSscal<float>(int n, float alpha, float *x, int offx);
template void BlasScopy<float>(int n, const float *x, int offx, float *y,
                               int offy);
template void BlasSaxpy<float>(int n, float alpha, const float *x, int offx,
                               float *y, int offy);
template void BlasSasum<float>(int n, const float *x, int offx, float *y);

// Level 2
template void BlasSgemv<float>(int TA, int M, int N, float alpha,
                               const float *A, int offA, const float *x,
                               int offx, float beta, float *y, int offy);

// Level 3
template void BlasSgemm<float>(int TA, int TB, int M, int N, int K, float alpha,
                               const float *A, int offA, const float *B,
                               int offB, float beta, float *C, int offC);

#endif

}  // namespace Blas
