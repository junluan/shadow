#include "blas.hpp"

#include "external.hpp"

#include <cfloat>

namespace Shadow {

namespace Blas {

__global__ void KernelSet(int n, float val, float* y, int offy) {
  CUDA_KERNEL_LOOP(globalid, n) { y[offy + globalid] = val; }
}

template <>
void Set<DeviceType::kGPU, float>(int n, float val, float* y, int offy,
                                  Context* context) {
  KernelSet<<<GetBlocks(n), NumThreads, 0, cudaStream_t(context->stream())>>>(
      n, val, y, offy);
  CUDA_CHECK(cudaPeekAtLastError());
}

#define DEFINE_BLAS_BINARY_FUNC(name, operation)                               \
  __global__ void Kernel##name(int n, const float* a, int offa,                \
                               const float* b, int offb, float* y, int offy) { \
    CUDA_KERNEL_LOOP(i, n) {                                                   \
      a += offa, b += offb, y += offy;                                         \
      operation;                                                               \
    }                                                                          \
  }                                                                            \
  template <>                                                                  \
  void name<DeviceType::kGPU, float>(int n, const float* a, int offa,          \
                                     const float* b, int offb, float* y,       \
                                     int offy, Context* context) {             \
    Kernel##name<<<GetBlocks(n), NumThreads, 0,                                \
                   cudaStream_t(context->stream())>>>(n, a, offa, b, offb, y,  \
                                                      offy);                   \
    CUDA_CHECK(cudaPeekAtLastError());                                         \
  }

DEFINE_BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
DEFINE_BLAS_BINARY_FUNC(Pow, y[i] = powf(a[i], b[i]));
DEFINE_BLAS_BINARY_FUNC(Max, y[i] = fmaxf(a[i], b[i]));
DEFINE_BLAS_BINARY_FUNC(Min, y[i] = fminf(a[i], b[i]));
#undef DEFINE_BLAS_BINARY_FUNC

#define DEFINE_BLAS_BINARY_SCALAR_FUNC(name, operation)                      \
  __global__ void Kernel##name(int n, const float* a, int offa, float alpha, \
                               float* y, int offy) {                         \
    CUDA_KERNEL_LOOP(i, n) {                                                 \
      a += offa, y += offy;                                                  \
      operation;                                                             \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void name<DeviceType::kGPU, float>(int n, const float* a, int offa,        \
                                     float alpha, float* y, int offy,        \
                                     Context* context) {                     \
    Kernel##name<<<GetBlocks(n), NumThreads, 0,                              \
                   cudaStream_t(context->stream())>>>(n, a, offa, alpha, y,  \
                                                      offy);                 \
    CUDA_CHECK(cudaPeekAtLastError());                                       \
  }

DEFINE_BLAS_BINARY_SCALAR_FUNC(Add, y[i] = a[i] + alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Sub, y[i] = a[i] - alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Mul, y[i] = a[i] * alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Div, y[i] = a[i] / alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Pow, y[i] = powf(a[i], alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Max, y[i] = fmaxf(a[i], alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Min, y[i] = fminf(a[i], alpha));
#undef DEFINE_BLAS_BINARY_SCALAR_FUNC

#define DEFINE_BLAS_UNARY_FUNC(name, operation)                              \
  __global__ void Kernel##name(int n, const float* a, int offa, float* y,    \
                               int offy) {                                   \
    CUDA_KERNEL_LOOP(i, n) {                                                 \
      a += offa, y += offy;                                                  \
      operation;                                                             \
    }                                                                        \
  }                                                                          \
  template <>                                                                \
  void name<DeviceType::kGPU, float>(int n, const float* a, int offa,        \
                                     float* y, int offy, Context* context) { \
    Kernel##name<<<GetBlocks(n), NumThreads, 0,                              \
                   cudaStream_t(context->stream())>>>(n, a, offa, y, offy);  \
    CUDA_CHECK(cudaPeekAtLastError());                                       \
  }

DEFINE_BLAS_UNARY_FUNC(Abs, y[i] = fabsf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Square, y[i] = a[i] * a[i]);
DEFINE_BLAS_UNARY_FUNC(Sqrt, y[i] = sqrtf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Log, y[i] = logf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Exp, y[i] = expf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Sin, y[i] = sinf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Cos, y[i] = cosf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Tan, y[i] = tanf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Asin, y[i] = asinf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Acos, y[i] = acosf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Atan, y[i] = atanf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Floor, y[i] = floorf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Ceil, y[i] = ceilf(a[i]));
DEFINE_BLAS_UNARY_FUNC(Neg, y[i] = -a[i]);
DEFINE_BLAS_UNARY_FUNC(Reciprocal, y[i] = 1 / a[i]);
#undef DEFINE_BLAS_UNARY_FUNC

// Level 1
template <>
void BlasSscal<DeviceType::kGPU, float>(int n, float alpha, float* x, int offx,
                                        Context* context) {
  CUBLAS_CHECK(cublasSscal(cublasHandle_t(context->cublas_handle()), n, &alpha,
                           x + offx, 1));
}

template <>
void BlasScopy<DeviceType::kGPU, float>(int n, const float* x, int offx,
                                        float* y, int offy, Context* context) {
  CUBLAS_CHECK(cublasScopy(cublasHandle_t(context->cublas_handle()), n,
                           x + offx, 1, y + offy, 1));
}

template <>
void BlasSaxpy<DeviceType::kGPU, float>(int n, float alpha, const float* x,
                                        int offx, float* y, int offy,
                                        Context* context) {
  CUBLAS_CHECK(cublasSaxpy(cublasHandle_t(context->cublas_handle()), n, &alpha,
                           x + offx, 1, y + offy, 1));
}

template <>
void BlasSasum<DeviceType::kGPU, float>(int n, const float* x, int offx,
                                        float* y, Context* context) {
  CUBLAS_CHECK(
      cublasSasum(cublasHandle_t(context->cublas_handle()), n, x + offx, 1, y));
}

// Level 2
template <>
void BlasSgemv<DeviceType::kGPU, float>(int TA, int M, int N, float alpha,
                                        const float* A, int offA,
                                        const float* x, int offx, float beta,
                                        float* y, int offy, Context* context) {
  auto transA = TA ? CUBLAS_OP_N : CUBLAS_OP_T;
  CUBLAS_CHECK(cublasSgemv(cublasHandle_t(context->cublas_handle()), transA, N,
                           M, &alpha, A + offA, N, x + offx, 1, &beta, y + offy,
                           1));
}

// Level 3
template <>
void BlasSgemm<DeviceType::kGPU, float>(int TA, int TB, int M, int N, int K,
                                        float alpha, const float* A, int offA,
                                        const float* B, int offB, float beta,
                                        float* C, int offC, Context* context) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  auto transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemm(cublasHandle_t(context->cublas_handle()), transB,
                           transA, N, M, K, &alpha, B + offB, ldb, A + offA,
                           lda, &beta, C + offC, N));
}

}  // namespace Blas

}  // namespace Shadow
