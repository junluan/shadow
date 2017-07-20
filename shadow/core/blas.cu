#include "blas.hpp"
#include "kernel.hpp"
#include "util/log.hpp"

#include <cfloat>

namespace Shadow {

namespace Blas {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelChannelMax(int num, int channels, int spatial_dim,
                                 const T *data, T *val_max) {
  CUDA_KERNEL_LOOP(globalid, num * spatial_dim) {
    int n = globalid / spatial_dim;
    int s = globalid % spatial_dim;
    T max_val = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      max_val = fmaxf(data[(n * channels + c) * spatial_dim + s], max_val);
    }
    val_max[globalid] = max_val;
  }
}

template <typename T>
void ChannelMax(int num, int channels, int spatial_dim, const T *data,
                T *val_max) {
  KernelChannelMax<T><<<GetBlocks(num * spatial_dim), NumThreads>>>(
      num, channels, spatial_dim, data, val_max);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelChannelSub(int count, int num, int channels,
                                 int spatial_dim, const T *val_sub, T *data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / channels / spatial_dim;
    int s = globalid % spatial_dim;
    data[globalid] -= val_sub[n * spatial_dim + s];
  }
}

template <typename T>
void ChannelSub(int count, int num, int channels, int spatial_dim,
                const T *val_sub, T *data) {
  KernelChannelSub<T><<<GetBlocks(count), NumThreads>>>(
      count, num, channels, spatial_dim, val_sub, data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelChannelSum(int num, int channels, int spatial_dim,
                                 const T *data, T *val_sum) {
  CUDA_KERNEL_LOOP(globalid, num * spatial_dim) {
    int n = globalid / spatial_dim;
    int s = globalid % spatial_dim;
    T sum = T(0);
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    val_sum[globalid] = sum;
  }
}

template <typename T>
void ChannelSum(int num, int channels, int spatial_dim, const T *data,
                T *val_sum) {
  KernelChannelSum<T><<<GetBlocks(num * spatial_dim), NumThreads>>>(
      num, channels, spatial_dim, data, val_sum);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelChannelDiv(int count, int num, int channels,
                                 int spatial_dim, const T *val_div, T *data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / channels / spatial_dim;
    int s = globalid % spatial_dim;
    data[globalid] /= val_div[n * spatial_dim + s];
  }
}

template <typename T>
void ChannelDiv(int count, int num, int channels, int spatial_dim,
                const T *val_div, T *data) {
  KernelChannelDiv<T><<<GetBlocks(count), NumThreads>>>(
      count, num, channels, spatial_dim, val_div, data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelSet(int n, float val, T *y, int offy) {
  CUDA_KERNEL_LOOP(globalid, n) { y[offy + globalid] = val; }
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  KernelSet<T><<<GetBlocks(n), NumThreads>>>(n, val, y, offy);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelAddScalar(int n, float val, T *y, int offy) {
  CUDA_KERNEL_LOOP(globalid, n) { y[offy + globalid] += val; }
}

template <typename T>
void Add(int n, float val, T *y, int offy) {
  KernelAddScalar<T><<<GetBlocks(n), NumThreads>>>(n, val, y, offy);
  CUDA_CHECK(cudaPeekAtLastError());
}

#define BLAS_BINARY_FUNC(name, operation)                                 \
  template <typename T>                                                   \
  __global__ void Kernel##name(int n, const T *a, int offa, const T *b,   \
                               int offb, T *y, int offy) {                \
    CUDA_KERNEL_LOOP(i, n) {                                              \
      a += offa, b += offb, y += offy;                                    \
      operation;                                                          \
    }                                                                     \
  }                                                                       \
  template <typename T>                                                   \
  void name(int n, const T *a, int offa, const T *b, int offb, T *y,      \
            int offy) {                                                   \
    Kernel##name<T><<<GetBlocks(n), NumThreads>>>(n, a, offa, b, offb, y, \
                                                  offy);                  \
    CUDA_CHECK(cudaPeekAtLastError());                                    \
  }                                                                       \
  template void name(int n, const float *a, int offa, const float *b,     \
                     int offb, float *y, int offy);

#define BLAS_UNARY_FUNC(name, operation)                                      \
  template <typename T>                                                       \
  __global__ void Kernel##name(int n, const T *a, int offa, T *y, int offy) { \
    CUDA_KERNEL_LOOP(i, n) {                                                  \
      a += offa, y += offy;                                                   \
      operation;                                                              \
    }                                                                         \
  }                                                                           \
  template <typename T>                                                       \
  void name(int n, const T *a, int offa, T *y, int offy) {                    \
    Kernel##name<T><<<GetBlocks(n), NumThreads>>>(n, a, offa, y, offy);       \
    CUDA_CHECK(cudaPeekAtLastError());                                        \
  }                                                                           \
  template void name(int n, const float *a, int offa, float *y, int offy);

BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

BLAS_BINARY_FUNC(Max, y[i] = fmaxf(a[i], b[i]));
BLAS_BINARY_FUNC(Min, y[i] = fminf(a[i], b[i]));

BLAS_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
BLAS_UNARY_FUNC(Exp, y[i] = expf(a[i]));
BLAS_UNARY_FUNC(Log, y[i] = logf(a[i]));
BLAS_UNARY_FUNC(Abs, y[i] = fabsf(a[i]));

template <typename T>
__global__ void KernelPow(int n, const T *a, int offa, float alpha, T *y,
                          int offy) {
  CUDA_KERNEL_LOOP(globalid, n) {
    y[offy + globalid] = pow(a[offa + globalid], alpha);
  }
}

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  KernelPow<T><<<GetBlocks(n), NumThreads>>>(n, a, offa, alpha, y, offy);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
  BlasScopy(n, x, offx, y, offy);
  BlasSscal(n, alpha, y, offy);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
  cublasSscal(Kernel::cublas_handle_, n, &alpha, x + offx, 1);
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  cublasScopy(Kernel::cublas_handle_, n, x + offx, 1, y + offy, 1);
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  cublasSaxpy(Kernel::cublas_handle_, n, &alpha, x + offx, 1, y + offy, 1);
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  cublasSasum(Kernel::cublas_handle_, n, x + offx, 1, y);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy) {
  auto transA = TA ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasSgemv(Kernel::cublas_handle_, transA, N, M, &alpha, A + offA, N,
              x + offx, 1, &beta, y + offy, 1);
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  auto transA = TA ? CUBLAS_OP_T : CUBLAS_OP_N;
  auto transB = TB ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(Kernel::cublas_handle_, transB, transA, N, M, K, &alpha, B + offB,
              ldb, A + offA, lda, &beta, C + offC, N);
}

// Explicit instantiation
template void ChannelMax(int num, int channels, int spatial_dim,
                         const float *data, float *val_max);
template void ChannelSub(int count, int num, int channels, int spatial_dim,
                         const float *val_sub, float *data);
template void ChannelSum(int num, int channels, int spatial_dim,
                         const float *data, float *val_sum);
template void ChannelDiv(int count, int num, int channels, int spatial_dim,
                         const float *val_div, float *data);

template void Set(int n, float val, float *y, int offy);
template void Add(int n, float val, float *y, int offy);
template void Pow(int n, const float *a, int offa, float alpha, float *y,
                  int offy);
template void Scale(int n, float alpha, const float *x, int offx, float *y,
                    int offy);

// Level 1
template void BlasSscal(int n, float alpha, float *x, int offx);
template void BlasScopy(int n, const float *x, int offx, float *y, int offy);
template void BlasSaxpy(int n, float alpha, const float *x, int offx, float *y,
                        int offy);
template void BlasSasum(int n, const float *x, int offx, float *y);

// Level 2
template void BlasSgemv(int TA, int M, int N, float alpha, const float *A,
                        int offA, const float *x, int offx, float beta,
                        float *y, int offy);

// Level 3
template void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha,
                        const float *A, int offA, const float *B, int offB,
                        float beta, float *C, int offC);
#endif

}  // namespace Blas

}  // namespace Shadow
