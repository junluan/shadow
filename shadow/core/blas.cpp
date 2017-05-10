#include "blas.hpp"
#include "kernel.hpp"

#if defined(USE_CL)
#include <clBLAS.h>
#endif

#if defined(USE_OpenBLAS)
#include "cblas.h"
#endif

#if defined(USE_Eigen)
#include <Eigen/Eigen>

template <typename T>
using MapVector = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using MapMatrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
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
#if defined(USE_Eigen)
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen.setConstant(static_cast<T>(val));
#else
  std::fill(y + offy, y + offy + n, static_cast<T>(val));
#endif
}

template <typename T>
void Add(int n, float val, T *y, int offy) {
#if defined(USE_Eigen)
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen = y_eigen.array() + static_cast<T>(val);
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] += static_cast<T>(val);
  }
#endif
}

#if defined(USE_Eigen)
#define BLAS_BINARY_FUNC_EIGEN(name, operation)                       \
  template <typename T>                                               \
  void name(int n, const T *a, int offa, const T *b, int offb, T *y,  \
            int offy) {                                               \
    const auto &a_eigen = MapVector<T>(const_cast<T *>(a + offa), n); \
    const auto &b_eigen = MapVector<T>(const_cast<T *>(b + offb), n); \
    auto y_eigen = MapVector<T>(y + offy, n);                         \
    operation;                                                        \
  }                                                                   \
  template void name(int n, const float *a, int offa, const float *b, \
                     int offb, float *y, int offy);

#define BLAS_UNARY_FUNC_EIGEN(name, operation)                        \
  template <typename T>                                               \
  void name(int n, const T *a, int offa, T *y, int offy) {            \
    const auto &a_eigen = MapVector<T>(const_cast<T *>(a + offa), n); \
    auto y_eigen = MapVector<T>(y + offy, n);                         \
    operation;                                                        \
  }                                                                   \
  template void name(int n, const float *a, int offa, float *y, int offy);

BLAS_BINARY_FUNC_EIGEN(Add, y_eigen = a_eigen.array() + b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Sub, y_eigen = a_eigen.array() - b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Mul, y_eigen = a_eigen.array() * b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Div, y_eigen = a_eigen.array() / b_eigen.array());

BLAS_UNARY_FUNC_EIGEN(Sqr, y_eigen = a_eigen.array() * a_eigen.array());
BLAS_UNARY_FUNC_EIGEN(Exp, y_eigen = a_eigen.array().exp());
BLAS_UNARY_FUNC_EIGEN(Log, y_eigen = a_eigen.array().log());
BLAS_UNARY_FUNC_EIGEN(Abs, y_eigen = a_eigen.array().abs());

#else
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

#define BLAS_UNARY_FUNC(name, operation)                   \
  template <typename T>                                    \
  void name(int n, const T *a, int offa, T *y, int offy) { \
    a += offa, y += offy;                                  \
    for (int i = 0; i < n; ++i) {                          \
      operation;                                           \
    }                                                      \
  }                                                        \
  template void name(int n, const float *a, int offa, float *y, int offy);

BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);

BLAS_UNARY_FUNC(Sqr, y[i] = a[i] * a[i]);
BLAS_UNARY_FUNC(Exp, y[i] = std::exp(a[i]));
BLAS_UNARY_FUNC(Log, y[i] = std::log(a[i]));
BLAS_UNARY_FUNC(Abs, y[i] = std::abs(a[i]));
#endif

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
#if defined(USE_Eigen)
  const auto &a_eigen = MapVector<T>(const_cast<T *>(a + offa), n);
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen = a_eigen.array().pow(alpha);
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] = std::pow(a[offa + i], alpha);
  }
#endif
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
#if defined(USE_Eigen)
  const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), n);
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen = static_cast<T>(alpha) * x_eigen;
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] = static_cast<T>(alpha) * x[offx + i];
  }
#endif
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
#if defined(USE_OpenBLAS)
  cblas_sscal(n, alpha, x + offx, 1);
#elif defined(USE_Eigen)
  auto x_eigen = MapVector<T>(x + offx, n);
  x_eigen = alpha * x_eigen;
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
#elif defined(USE_Eigen)
  const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), n);
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen = x_eigen;
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
#elif defined(USE_Eigen)
  const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), n);
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen = alpha * x_eigen + y_eigen;
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
#elif defined(USE_Eigen)
  const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), n);
  *y = static_cast<T>(x_eigen.cwiseAbs().sum());
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
#elif defined(USE_Eigen)
  const auto &A_eigen = MapMatrix<T>(const_cast<T *>(A + offA), N, M);
  if (!TA) {
    const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), N);
    auto y_eigen = MapVector<T>(y + offy, M);
    y_eigen = alpha * A_eigen.transpose() * x_eigen + beta * y_eigen;
  } else {
    const auto &x_eigen = MapVector<T>(const_cast<T *>(x + offx), M);
    auto y_eigen = MapVector<T>(y + offy, N);
    y_eigen = alpha * A_eigen * x_eigen + beta * y_eigen;
  }
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
#elif defined(USE_Eigen)
  auto C_eigen = MapMatrix<T>(C + offC, N, M);
  if (!TA && !TB) {
    const auto &A_eigen = MapMatrix<T>(const_cast<T *>(A + offA), K, M);
    const auto &B_eigen = MapMatrix<T>(const_cast<T *>(B + offB), N, K);
    C_eigen = alpha * B_eigen * A_eigen + beta * C_eigen;
  } else if (TA && !TB) {
    const auto &A_eigen = MapMatrix<T>(const_cast<T *>(A + offA), M, K);
    const auto &B_eigen = MapMatrix<T>(const_cast<T *>(B + offB), N, K);
    C_eigen = alpha * B_eigen * A_eigen.transpose() + beta * C_eigen;
  } else if (!TA && TB) {
    const auto &A_eigen = MapMatrix<T>(const_cast<T *>(A + offA), K, M);
    const auto &B_eigen = MapMatrix<T>(const_cast<T *>(B + offB), K, N);
    C_eigen = alpha * B_eigen.transpose() * A_eigen + beta * C_eigen;
  } else {
    const auto &A_eigen = MapMatrix<T>(const_cast<T *>(A + offA), M, K);
    const auto &B_eigen = MapMatrix<T>(const_cast<T *>(B + offB), K, N);
    C_eigen =
        alpha * B_eigen.transpose() * A_eigen.transpose() + beta * C_eigen;
  }
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

#elif defined(USE_CL)
template <typename T>
void ChannelMax(int num, int channels, int spatial_dim, const T *data,
                T *val_max) {
  size_t global = num * spatial_dim;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["ChannelMax"];
  kernel->SetArguments(num, channels, spatial_dim, *data, *val_max);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelSub(int count, int num, int channels, int spatial_dim,
                const T *val_sub, T *data) {
  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["ChannelSub"];
  kernel->SetArguments(count, num, channels, spatial_dim, *val_sub, *data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelSum(int num, int channels, int spatial_dim, const T *data,
                T *val_sum) {
  size_t global = num * spatial_dim;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["ChannelSum"];
  kernel->SetArguments(num, channels, spatial_dim, *data, *val_sum);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}
template <typename T>
void ChannelDiv(int count, int num, int channels, int spatial_dim,
                const T *val_div, T *data) {
  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["ChannelDiv"];
  kernel->SetArguments(count, num, channels, spatial_dim, *val_div, *data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  size_t global = n;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["Set"];
  kernel->SetArguments(n, val, *y, offy);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Add(int n, float val, T *y, int offy) {
  size_t global = n;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["AddScalar"];
  kernel->SetArguments(n, val, *y, offy);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

#define BLAS_BINARY_FUNC(name, kname)                                       \
  template <typename T>                                                     \
  inline void name(int n, const T *a, int offa, const T *b, int offb, T *y, \
                   int offy) {                                              \
    size_t global = n;                                                      \
    EasyCL::Kernel *kernel = Kernel::cl_kernels_[kname];                    \
    kernel->SetArguments(n, *a, offa, *b, offb, *y, offy);                  \
    kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);              \
    Kernel::queue_->Finish();                                               \
  }                                                                         \
  template void name(int n, const BufferF *a, int offa, const BufferF *b,   \
                     int offb, BufferF *y, int offy);

BLAS_BINARY_FUNC(Add, "Add");
BLAS_BINARY_FUNC(Sub, "Sub");
BLAS_BINARY_FUNC(Mul, "Mul");
BLAS_BINARY_FUNC(Div, "Div");

#define BLAS_UNARY_FUNC(name, kname)                              \
  template <typename T>                                           \
  inline void name(int n, const T *a, int offa, T *y, int offy) { \
    size_t global = n;                                            \
    EasyCL::Kernel *kernel = Kernel::cl_kernels_[kname];          \
    kernel->SetArguments(n, *a, offa, *y, offy);                  \
    kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);    \
    Kernel::queue_->Finish();                                     \
  }                                                               \
  template void name(int n, const BufferF *a, int offa, BufferF *y, int offy);

BLAS_UNARY_FUNC(Sqr, "Sqr");
BLAS_UNARY_FUNC(Exp, "Exp");
BLAS_UNARY_FUNC(Log, "Log");
BLAS_UNARY_FUNC(Abs, "Abs");

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  size_t global = n;
  EasyCL::Kernel *kernel = Kernel::cl_kernels_["Pow"];
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
  clblasSscal(n, alpha, (*x)(), offx, 1, 1, Kernel::queue_->pointer(), 0,
              nullptr, nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  clblasScopy(n, (*x)(), offx, 1, (*y)(), offy, 1, 1, Kernel::queue_->pointer(),
              0, nullptr, nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  clblasSaxpy(n, alpha, (*x)(), offx, 1, (*y)(), offy, 1, 1,
              Kernel::queue_->pointer(), 0, nullptr, nullptr);
  Kernel::queue_->Finish();
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  BufferF y_(*Kernel::context_, 1), temp_(*Kernel::context_, n);
  clblasSasum(n, y_(), 0, (*x)(), offx, 1, temp_(), 1,
              Kernel::queue_->pointer(), 0, nullptr, nullptr);
  Kernel::queue_->Finish();
  y_.Read(*Kernel::queue_, 1, y);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, int offA,
               const T *x, int offx, float beta, T *y, int offy) {
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasSgemv(clblasRowMajor, transA, M, N, alpha, (*A)(), offA, N, (*x)(),
              offx, 1, beta, (*y)(), offy, 1, 1, Kernel::queue_->pointer(), 0,
              nullptr, nullptr);
  Kernel::queue_->Finish();
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, alpha, (*A)(), offA, lda,
              (*B)(), offB, ldb, beta, (*C)(), offC, N, 1,
              Kernel::queue_->pointer(), 0, nullptr, nullptr);
  Kernel::queue_->Finish();
}

// Explicit instantiation
template void ChannelMax(int num, int channels, int spatial_dim,
                         const BufferF *data, BufferF *val_max);
template void ChannelSub(int count, int num, int channels, int spatial_dim,
                         const BufferF *val_sub, BufferF *data);
template void ChannelSum(int num, int channels, int spatial_dim,
                         const BufferF *data, BufferF *val_sum);
template void ChannelDiv(int count, int num, int channels, int spatial_dim,
                         const BufferF *val_div, BufferF *data);

template void Set(int n, float val, BufferF *y, int offy);
template void Add(int n, float val, BufferF *y, int offy);
template void Pow(int n, const BufferF *a, int offa, float alpha, BufferF *y,
                  int offy);
template void Scale(int n, float alpha, const BufferF *x, int offx, BufferF *y,
                    int offy);

// Level 1
template void BlasSscal(int n, float alpha, BufferF *x, int offx);
template void BlasScopy(int n, const BufferF *x, int offx, BufferF *y,
                        int offy);
template void BlasSaxpy(int n, float alpha, const BufferF *x, int offx,
                        BufferF *y, int offy);
template void BlasSasum(int n, const BufferF *x, int offx, float *y);

// Level 2
template void BlasSgemv(int TA, int M, int N, float alpha, const BufferF *A,
                        int offA, const BufferF *x, int offx, float beta,
                        BufferF *y, int offy);

// Level 3
template void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha,
                        const BufferF *A, int offA, const BufferF *B, int offB,
                        float beta, BufferF *C, int offC);
#endif

}  // namespace Blas
