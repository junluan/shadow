#include "shadow/util/blas.hpp"
#include "shadow/kernel.hpp"

#if defined(USE_OpenBLAS)
#include "cblas.h"
#endif

#include <algorithm>
#include <cfloat>

namespace Blas {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void ChannelMax(int num, int channels, int spatial_dim, const T *data,
                T *val_max) {
  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      float max_val = -FLT_MAX;
      for (int c = 0; c < channels; ++c) {
        max_val = std::max(data[(n * channels + c) * spatial_dim + s], max_val);
      }
      val_max[n * spatial_dim + s] = max_val;
    }
  }
}
template <typename T>
void ChannelSub(int count, int num, int channels, int spatial_dim,
                const T *val_sub, T *data) {
  for (int n = 0; n < num; ++n) {
    int offset = n * spatial_dim;
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        data[(n * channels + c) * spatial_dim + s] -= val_sub[offset + s];
      }
    }
  }
}
template <typename T>
void ChannelSum(int num, int channels, int spatial_dim, const T *data,
                T *val_sum) {
  for (int n = 0; n < num; ++n) {
    for (int s = 0; s < spatial_dim; ++s) {
      float sum = 0;
      for (int c = 0; c < channels; ++c) {
        sum += data[(n * channels + c) * spatial_dim + s];
      }
      val_sum[n * spatial_dim + s] = sum;
    }
  }
}
template <typename T>
void ChannelDiv(int count, int num, int channels, int spatial_dim,
                const T *val_div, T *data) {
  for (int n = 0; n < num; ++n) {
    int offset = n * spatial_dim;
    for (int c = 0; c < channels; ++c) {
      for (int s = 0; s < spatial_dim; ++s) {
        data[(n * channels + c) * spatial_dim + s] /= val_div[offset + s];
      }
    }
  }
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  std::fill(y + offy, y + offy + n, val);
}

#define BLAS_BINARY_FUNC(name, operation)                             \
  template <typename T>                                               \
  void name(int n, const T *a, int offa, const T *b, int offb, T *y,  \
            int offy) {                                               \
    a += offa, b += offb, y += offy;                                  \
    for (int i = 0; i < n; ++i) {                                     \
      operation;                                                      \
    }                                                                 \
  }                                                                   \
  template void name(int n, const float *a, int offa, const float *b, \
                     int offb, float *y, int offy);

BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

#define BLAS_UNARY_FUNC(name, operation)                   \
  template <typename T>                                    \
  void name(int n, const T *a, int offa, T *y, int offy) { \
    a += offa, y += offy;                                  \
    for (int i = 0; i < n; ++i) {                          \
      operation;                                           \
    }                                                      \
  }                                                        \
  template void name(int n, const float *a, int offa, float *y, int offy);

BLAS_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
BLAS_UNARY_FUNC(Exp, y[i] = std::exp(a[i]));
BLAS_UNARY_FUNC(Log, y[i] = std::log(a[i]));
BLAS_UNARY_FUNC(Abs, y[i] = std::abs(a[i]));

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i] = std::pow(a[offa + i], alpha);
  }
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i] = alpha * x[offx + i];
  }
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
#if defined(USE_OpenBLAS)
  cblas_sscal(n, alpha, x + offx, 1);
#else
  for (int i = 0; i < n; ++i) {
    x[offx + i] *= alpha;
  }
#endif
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
#if defined(USE_OpenBLAS)
  cblas_scopy(n, x + offx, 1, y + offy, 1);
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] = x[offx + i];
  }
#endif
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
#if defined(USE_OpenBLAS)
  cblas_saxpy(n, alpha, x + offx, 1, y + offy, 1);
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] += alpha * x[offx + i];
  }
#endif
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
#if defined(USE_OpenBLAS)
  *y = cblas_sasum(n, x + offx, 1);
#else
  double asum = 0;
  for (int i = 0; i < n; ++i) {
    asum += std::abs(x[offx + i]);
  }
  *y = static_cast<T>(asum);
#endif
}

// Level 2
inline void SgemvN(int M, int N, float alpha, const float *A, const float *x,
                   float *y) {
  for (int i = 0; i < M; ++i) {
    double sum = 0;
    for (int j = 0; j < N; ++j) {
      sum += alpha * A[i * N + j] * x[j];
    }
    y[i] += static_cast<float>(sum);
  }
}

inline void SgemvT(int M, int N, float alpha, const float *A, const float *x,
                   float *y) {
  for (int i = 0; i < N; ++i) {
    double sum = 0;
    for (int j = 0; j < M; ++j) {
      sum += alpha * A[j * N + i] * x[j];
    }
    y[i] += static_cast<float>(sum);
  }
}

template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy) {
#if defined(USE_OpenBLAS)
  CBLAS_TRANSPOSE transA = TA ? CblasTrans : CblasNoTrans;
  cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A + offA, N, x + offx, 1,
              beta, y + offy, 1);
#else
  for (int i = 0; i < (TA ? N : M); ++i) {
    y[offy + i] *= beta;
  }
  if (!TA) {
    SgemvN(M, N, alpha, A + offA, x + offx, y + offy);
  } else {
    SgemvT(M, N, alpha, A + offA, x + offx, y + offy);
  }
#endif
}

// Level 3
inline void SgemmNN(int M, int N, int K, float alpha, const float *A,
                    const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_part = alpha * A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A_part * B[k * N + j];
      }
    }
  }
}

inline void SgemmTN(int M, int N, int K, float alpha, const float *A,
                    const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_part = alpha * A[k * M + i];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A_part * B[k * N + j];
      }
    }
  }
}

inline void SgemmNT(int M, int N, int K, float alpha, const float *A,
                    const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[i * K + k] * B[j * K + k];
      }
      C[i * N + j] += sum;
    }
  }
}

inline void SgemmTT(int M, int N, int K, float alpha, const float *A,
                    const float *B, float *C) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[k * M + i] * B[j * K + k];
      }
      C[i * N + j] += sum;
    }
  }
}

template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
#if defined(USE_OpenBLAS)
  int lda = TA ? M : K, ldb = TB ? K : N;
  CBLAS_TRANSPOSE transA = TA ? CblasTrans : CblasNoTrans;
  CBLAS_TRANSPOSE transB = TB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A + offA, lda,
              B + offB, ldb, beta, C + offC, N);
#else
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[offC + i * N + j] *= beta;
    }
  }
  if (!TA && !TB) {
    SgemmNN(M, N, K, alpha, A + offA, B + offB, C + offC);
  } else if (TA && !TB) {
    SgemmTN(M, N, K, alpha, A + offA, B + offB, C + offC);
  } else if (!TA && TB) {
    SgemmNT(M, N, K, alpha, A + offA, B + offB, C + offC);
  } else {
    SgemmTT(M, N, K, alpha, A + offA, B + offB, C + offC);
  }
#endif
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

#elif defined(USE_CL)
#include <clBLAS.h>

template <typename T>
void ChannelMax(int num, int channels, int spatial_dim, const T *data,
                T *val_max) {
  size_t global = num * spatial_dim;
  CLCudaAPI::Kernel *kernel = Kernel::cl_channelmax_kernel_;
  kernel->SetArguments(num, channels, spatial_dim, *data, *val_max);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelSub(int count, int num, int channels, int spatial_dim,
                const T *val_sub, T *data) {
  size_t global = count;
  CLCudaAPI::Kernel *kernel = Kernel::cl_channelsub_kernel_;
  kernel->SetArguments(count, num, channels, spatial_dim, *val_sub, *data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelSum(int num, int channels, int spatial_dim, const T *data,
                T *val_sum) {
  size_t global = num * spatial_dim;
  CLCudaAPI::Kernel *kernel = Kernel::cl_channelsum_kernel_;
  kernel->SetArguments(num, channels, spatial_dim, *data, *val_sum);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelDiv(int count, int num, int channels, int spatial_dim,
                const T *val_div, T *data) {
  size_t global = count;
  CLCudaAPI::Kernel *kernel = Kernel::cl_channeldiv_kernel_;
  kernel->SetArguments(count, num, channels, spatial_dim, *val_div, *data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  cl_kernel kernel = (*Kernel::cl_set_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(float), &val);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 3, sizeof(int), &offy);
  size_t global = n;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

#define BLAS_BINARY_FUNC(name, kname)                                       \
  template <typename T>                                                     \
  inline void name(int n, const T *a, int offa, const T *b, int offb, T *y, \
                   int offy) {                                              \
    size_t global = n;                                                      \
    CLCudaAPI::Kernel *kernel = Kernel::cl_##kname;                         \
    kernel->SetArguments(n, *a, offa, *b, offb, *y, offy);                  \
    kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);              \
    Kernel::queue_->Finish();                                               \
  }                                                                         \
  template void name(int n, const cl_mem *a, int offa, const cl_mem *b,     \
                     int offb, cl_mem *y, int offy);

BLAS_BINARY_FUNC(Add, add_kernel_);
BLAS_BINARY_FUNC(Sub, sub_kernel_);
BLAS_BINARY_FUNC(Mul, mul_kernel_);
BLAS_BINARY_FUNC(Div, div_kernel_);

#define BLAS_UNARY_FUNC(name, kname)                              \
  template <typename T>                                           \
  inline void name(int n, const T *a, int offa, T *y, int offy) { \
    size_t global = n;                                            \
    CLCudaAPI::Kernel *kernel = Kernel::cl_##kname;               \
    kernel->SetArguments(n, *a, offa, *y, offy);                  \
    kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);    \
    Kernel::queue_->Finish();                                     \
  }                                                               \
  template void name(int n, const cl_mem *a, int offa, cl_mem *y, int offy);

BLAS_UNARY_FUNC(Sqr, sqr_kernel_);
BLAS_UNARY_FUNC(Exp, exp_kernel_);
BLAS_UNARY_FUNC(Log, log_kernel_);
BLAS_UNARY_FUNC(Abs, abs_kernel_);

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  size_t global = n;
  CLCudaAPI::Kernel *kernel = Kernel::cl_pow_kernel_;
  kernel->SetArguments(n, *a, offa, alpha, *y, offy);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
  BlasScopy(n, x, offx, y, offy);
  BlasSscal(n, alpha, y, offy);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
  clblasSscal(n, alpha, *x, offx, 1, 1, Kernel::queue_->pointer(), 0, nullptr,
              nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  clblasScopy(n, *x, offx, 1, *y, offy, 1, 1, Kernel::queue_->pointer(), 0,
              nullptr, nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  clblasSaxpy(n, alpha, *x, offx, 1, *y, offy, 1, 1, Kernel::queue_->pointer(),
              0, nullptr, nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  cl_mem *y_ = Kernel::MakeBuffer<cl_mem>(1, static_cast<float *>(nullptr));
  cl_mem *temp_ = Kernel::MakeBuffer<cl_mem>(n, static_cast<float *>(nullptr));
  clblasSasum(n, *y_, 0, *x, offx, 1, *temp_, 1, Kernel::queue_->pointer(), 0,
              nullptr, nullptr);
  Kernel::queue_->Finish();
  Kernel::ReadBuffer(1, y_, y);
  Kernel::ReleaseBuffer(y_);
  Kernel::ReleaseBuffer(temp_);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy) {
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasSgemv(clblasRowMajor, transA, M, N, alpha, *A, offA, N, *x, offx, 1,
              beta, *y, offy, 1, 1, Kernel::queue_->pointer(), 0, nullptr,
              nullptr);
  Kernel::queue_->Finish();
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, alpha, *A, offA, lda, *B,
              offB, ldb, beta, *C, offC, N, 1, Kernel::queue_->pointer(), 0,
              nullptr, nullptr);
  Kernel::queue_->Finish();
}

// Explicit instantiation
template void ChannelMax(int num, int channels, int spatial_dim,
                         const cl_mem *data, cl_mem *val_max);
template void ChannelSub(int count, int num, int channels, int spatial_dim,
                         const cl_mem *val_sub, cl_mem *data);
template void ChannelSum(int num, int channels, int spatial_dim,
                         const cl_mem *data, cl_mem *val_sum);
template void ChannelDiv(int count, int num, int channels, int spatial_dim,
                         const cl_mem *val_div, cl_mem *data);

template void Set(int n, float val, cl_mem *y, int offy);
template void Pow(int n, const cl_mem *a, int offa, float alpha, cl_mem *y,
                  int offy);
template void Scale(int n, float alpha, const cl_mem *x, int offx, cl_mem *y,
                    int offy);

// Level 1
template void BlasSscal(int n, float alpha, cl_mem *x, int offx);
template void BlasScopy(int n, const cl_mem *x, int offx, cl_mem *y, int offy);
template void BlasSaxpy(int n, float alpha, const cl_mem *x, int offx,
                        cl_mem *y, int offy);
template void BlasSasum(int n, const cl_mem *x, int offx, float *y);

// Level 2
template void BlasSgemv(int TA, int M, int N, float alpha, const cl_mem *A,
                        int offA, const cl_mem *x, int offx, float beta,
                        cl_mem *y, int offy);

// Level 3
template void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha,
                        const cl_mem *A, int offA, const cl_mem *B, int offB,
                        float beta, cl_mem *C, int offC);
#endif

}  // namespace Blas
