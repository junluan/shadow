#include "blas.hpp"

#if defined(USE_Eigen)
#include "Eigen/Eigen"
template <typename T>
using MapVector = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using MapMatrix = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
#endif

#if defined(USE_OpenBLAS)
#include "cblas.h"
#elif defined(USE_MKL)
#include "mkl_cblas.h"
#endif

#include <algorithm>
#include <cfloat>
#include <cmath>

namespace Shadow {

namespace Blas {

template <>
void Set<DeviceType::kCPU, float>(int n, float val, float* y, int offy,
                                  Context* context) {
#if defined(USE_Eigen)
  auto y_eigen = MapVector<float>(y + offy, n);
  y_eigen.setConstant(val);
#else
  std::fill(y + offy, y + offy + n, val);
#endif
}

#if defined(USE_Eigen)
#define DEFINE_BLAS_BINARY_FUNC(name, operation)                             \
  template <>                                                                \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa,        \
                                     const float* b, int offb, float* y,     \
                                     int offy, Context* context) {           \
    const auto& a_eigen = MapVector<float>(const_cast<float*>(a + offa), n); \
    const auto& b_eigen = MapVector<float>(const_cast<float*>(b + offb), n); \
    auto y_eigen = MapVector<float>(y + offy, n);                            \
    operation;                                                               \
  }

DEFINE_BLAS_BINARY_FUNC(Add, y_eigen = a_eigen.array() + b_eigen.array());
DEFINE_BLAS_BINARY_FUNC(Sub, y_eigen = a_eigen.array() - b_eigen.array());
DEFINE_BLAS_BINARY_FUNC(Mul, y_eigen = a_eigen.array() * b_eigen.array());
DEFINE_BLAS_BINARY_FUNC(Div, y_eigen = a_eigen.array() / b_eigen.array());
DEFINE_BLAS_BINARY_FUNC(Pow, y_eigen = a_eigen.array().pow(b_eigen.array()));
DEFINE_BLAS_BINARY_FUNC(Max, y_eigen = a_eigen.cwiseMax(b_eigen));
DEFINE_BLAS_BINARY_FUNC(Min, y_eigen = a_eigen.cwiseMin(b_eigen));
#undef DEFINE_BLAS_BINARY_FUNC

#define DEFINE_BLAS_BINARY_SCALAR_FUNC(name, operation)                      \
  template <>                                                                \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa,        \
                                     float alpha, float* y, int offy,        \
                                     Context* context) {                     \
    const auto& a_eigen = MapVector<float>(const_cast<float*>(a + offa), n); \
    auto y_eigen = MapVector<float>(y + offy, n);                            \
    operation;                                                               \
  }

DEFINE_BLAS_BINARY_SCALAR_FUNC(Add, y_eigen = a_eigen.array() + alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Sub, y_eigen = a_eigen.array() - alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Mul, y_eigen = a_eigen.array() * alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Div, y_eigen = a_eigen.array() / alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Pow, y_eigen = a_eigen.array().pow(alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Max, y_eigen = a_eigen.cwiseMax(alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Min, y_eigen = a_eigen.cwiseMin(alpha));
#undef DEFINE_BLAS_BINARY_SCALAR_FUNC

#define DEFINE_BLAS_UNARY_FUNC(name, operation)                              \
  template <>                                                                \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa,        \
                                     float* y, int offy, Context* context) { \
    const auto& a_eigen = MapVector<float>(const_cast<float*>(a + offa), n); \
    auto y_eigen = MapVector<float>(y + offy, n);                            \
    operation;                                                               \
  }

DEFINE_BLAS_UNARY_FUNC(Abs, y_eigen = a_eigen.array().abs());
DEFINE_BLAS_UNARY_FUNC(Square, y_eigen = a_eigen.array().square());
DEFINE_BLAS_UNARY_FUNC(Sqrt, y_eigen = a_eigen.array().sqrt());
DEFINE_BLAS_UNARY_FUNC(Log, y_eigen = a_eigen.array().log());
DEFINE_BLAS_UNARY_FUNC(Exp, y_eigen = a_eigen.array().exp());
DEFINE_BLAS_UNARY_FUNC(Sin, y_eigen = a_eigen.array().sin());
DEFINE_BLAS_UNARY_FUNC(Cos, y_eigen = a_eigen.array().cos());
DEFINE_BLAS_UNARY_FUNC(Tan, y_eigen = a_eigen.array().tan());
DEFINE_BLAS_UNARY_FUNC(Asin, y_eigen = a_eigen.array().asin());
DEFINE_BLAS_UNARY_FUNC(Acos, y_eigen = a_eigen.array().acos());
DEFINE_BLAS_UNARY_FUNC(Atan, y_eigen = a_eigen.array().atan());
DEFINE_BLAS_UNARY_FUNC(Floor, y_eigen = a_eigen.array().floor());
DEFINE_BLAS_UNARY_FUNC(Ceil, y_eigen = a_eigen.array().ceil());
DEFINE_BLAS_UNARY_FUNC(Neg, y_eigen = -a_eigen.array());
DEFINE_BLAS_UNARY_FUNC(Reciprocal, y_eigen = a_eigen.array().inverse());
#undef DEFINE_BLAS_UNARY_FUNC

#else
#define DEFINE_BLAS_BINARY_FUNC(name, operation)                         \
  template <>                                                            \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa,    \
                                     const float* b, int offb, float* y, \
                                     int offy, Context* context) {       \
    a += offa, b += offb, y += offy;                                     \
    for (int i = 0; i < n; ++i) {                                        \
      operation;                                                         \
    }                                                                    \
  }

DEFINE_BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
DEFINE_BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
DEFINE_BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
DEFINE_BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
DEFINE_BLAS_BINARY_FUNC(Pow, y[i] = std::pow(a[i], b[i]));
DEFINE_BLAS_BINARY_FUNC(Max, y[i] = std::max(a[i], b[i]));
DEFINE_BLAS_BINARY_FUNC(Min, y[i] = std::min(a[i], b[i]));
#undef DEFINE_BLAS_BINARY_FUNC

#define DEFINE_BLAS_BINARY_SCALAR_FUNC(name, operation)               \
  template <>                                                         \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa, \
                                     float alpha, float* y, int offy, \
                                     Context* context) {              \
    a += offa, y += offy;                                             \
    for (int i = 0; i < n; ++i) {                                     \
      operation;                                                      \
    }                                                                 \
  }

DEFINE_BLAS_BINARY_SCALAR_FUNC(Add, y[i] = a[i] + alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Sub, y[i] = a[i] - alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Mul, y[i] = a[i] * alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Div, y[i] = a[i] / alpha);
DEFINE_BLAS_BINARY_SCALAR_FUNC(Pow, y[i] = std::pow(a[i], alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Max, y[i] = std::max(a[i], alpha));
DEFINE_BLAS_BINARY_SCALAR_FUNC(Min, y[i] = std::min(a[i], alpha));
#undef DEFINE_BLAS_BINARY_SCALAR_FUNC

#define DEFINE_BLAS_UNARY_FUNC(name, operation)                              \
  template <>                                                                \
  void name<DeviceType::kCPU, float>(int n, const float* a, int offa,        \
                                     float* y, int offy, Context* context) { \
    a += offa, y += offy;                                                    \
    for (int i = 0; i < n; ++i) {                                            \
      operation;                                                             \
    }                                                                        \
  }

DEFINE_BLAS_UNARY_FUNC(Abs, y[i] = std::abs(a[i]));
DEFINE_BLAS_UNARY_FUNC(Square, y[i] = a[i] * a[i]);
DEFINE_BLAS_UNARY_FUNC(Sqrt, y[i] = std::sqrt(a[i]));
DEFINE_BLAS_UNARY_FUNC(Log, y[i] = std::log(a[i]));
DEFINE_BLAS_UNARY_FUNC(Exp, y[i] = std::exp(a[i]));
DEFINE_BLAS_UNARY_FUNC(Sin, y[i] = std::sin(a[i]));
DEFINE_BLAS_UNARY_FUNC(Cos, y[i] = std::cos(a[i]));
DEFINE_BLAS_UNARY_FUNC(Tan, y[i] = std::tan(a[i]));
DEFINE_BLAS_UNARY_FUNC(Asin, y[i] = std::asin(a[i]));
DEFINE_BLAS_UNARY_FUNC(Acos, y[i] = std::acos(a[i]));
DEFINE_BLAS_UNARY_FUNC(Atan, y[i] = std::atan(a[i]));
DEFINE_BLAS_UNARY_FUNC(Floor, y[i] = std::floor(a[i]));
DEFINE_BLAS_UNARY_FUNC(Ceil, y[i] = std::ceil(a[i]));
DEFINE_BLAS_UNARY_FUNC(Neg, y[i] = -a[i]);
DEFINE_BLAS_UNARY_FUNC(Reciprocal, y[i] = 1 / a[i]);
#undef DEFINE_BLAS_UNARY_FUNC
#endif

// Level 1
template <>
void BlasSscal<DeviceType::kCPU, float>(int n, float alpha, float* x, int offx,
                                        Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  cblas_sscal(n, alpha, x + offx, 1);
#elif defined(USE_Eigen)
  auto x_eigen = MapVector<float>(x + offx, n);
  x_eigen = alpha * x_eigen;
#else
  for (int i = 0; i < n; ++i) {
    x[offx + i] *= alpha;
  }
#endif
}

template <>
void BlasScopy<DeviceType::kCPU, float>(int n, const float* x, int offx,
                                        float* y, int offy, Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  cblas_scopy(n, x + offx, 1, y + offy, 1);
#elif defined(USE_Eigen)
  const auto& x_eigen = MapVector<float>(const_cast<float*>(x + offx), n);
  auto y_eigen = MapVector<float>(y + offy, n);
  y_eigen = x_eigen;
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] = x[offx + i];
  }
#endif
}

template <>
void BlasSaxpy<DeviceType::kCPU, float>(int n, float alpha, const float* x,
                                        int offx, float* y, int offy,
                                        Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  cblas_saxpy(n, alpha, x + offx, 1, y + offy, 1);
#elif defined(USE_Eigen)
  const auto& x_eigen = MapVector<float>(const_cast<float*>(x + offx), n);
  auto y_eigen = MapVector<float>(y + offy, n);
  y_eigen = alpha * x_eigen + y_eigen;
#else
  for (int i = 0; i < n; ++i) {
    y[offy + i] += alpha * x[offx + i];
  }
#endif
}

template <>
void BlasSasum<DeviceType::kCPU, float>(int n, const float* x, int offx,
                                        float* y, Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  *y = cblas_sasum(n, x + offx, 1);
#elif defined(USE_Eigen)
  const auto& x_eigen = MapVector<float>(const_cast<float*>(x + offx), n);
  *y = static_cast<float>(x_eigen.cwiseAbs().sum());
#else
  double asum = 0;
  for (int i = 0; i < n; ++i) {
    asum += std::abs(x[offx + i]);
  }
  *y = static_cast<float>(asum);
#endif
}

// Level 2
inline void SgemvN(int M, int N, float alpha, const float* A, const float* x,
                   float* y) {
  for (int i = 0; i < M; ++i) {
    double sum = 0;
    for (int j = 0; j < N; ++j) {
      sum += alpha * A[i * N + j] * x[j];
    }
    y[i] += static_cast<float>(sum);
  }
}

inline void SgemvT(int M, int N, float alpha, const float* A, const float* x,
                   float* y) {
  for (int i = 0; i < N; ++i) {
    double sum = 0;
    for (int j = 0; j < M; ++j) {
      sum += alpha * A[j * N + i] * x[j];
    }
    y[i] += static_cast<float>(sum);
  }
}

template <>
void BlasSgemv<DeviceType::kCPU, float>(int TA, int M, int N, float alpha,
                                        const float* A, int offA,
                                        const float* x, int offx, float beta,
                                        float* y, int offy, Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  auto transA = TA ? CblasTrans : CblasNoTrans;
  cblas_sgemv(CblasRowMajor, transA, M, N, alpha, A + offA, N, x + offx, 1,
              beta, y + offy, 1);
#elif defined(USE_Eigen)
  const auto& A_eigen = MapMatrix<float>(const_cast<float*>(A + offA), N, M);
  if (!TA) {
    const auto& x_eigen = MapVector<float>(const_cast<float*>(x + offx), N);
    auto y_eigen = MapVector<float>(y + offy, M);
    y_eigen = alpha * A_eigen.transpose() * x_eigen + beta * y_eigen;
  } else {
    const auto& x_eigen = MapVector<float>(const_cast<float*>(x + offx), M);
    auto y_eigen = MapVector<float>(y + offy, N);
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
inline void SgemmNN(int M, int N, int K, float alpha, const float* A,
                    const float* B, float* C) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_part = alpha * A[i * K + k];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A_part * B[k * N + j];
      }
    }
  }
}

inline void SgemmTN(int M, int N, int K, float alpha, const float* A,
                    const float* B, float* C) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_part = alpha * A[k * M + i];
      for (int j = 0; j < N; ++j) {
        C[i * N + j] += A_part * B[k * N + j];
      }
    }
  }
}

inline void SgemmNT(int M, int N, int K, float alpha, const float* A,
                    const float* B, float* C) {
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

inline void SgemmTT(int M, int N, int K, float alpha, const float* A,
                    const float* B, float* C) {
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

template <>
void BlasSgemm<DeviceType::kCPU, float>(int TA, int TB, int M, int N, int K,
                                        float alpha, const float* A, int offA,
                                        const float* B, int offB, float beta,
                                        float* C, int offC, Context* context) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  int lda = TA ? M : K, ldb = TB ? K : N;
  auto transA = TA ? CblasTrans : CblasNoTrans;
  auto transB = TB ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, A + offA, lda,
              B + offB, ldb, beta, C + offC, N);
#elif defined(USE_Eigen)
  auto C_eigen = MapMatrix<float>(C + offC, N, M);
  if (!TA && !TB) {
    const auto& A_eigen = MapMatrix<float>(const_cast<float*>(A + offA), K, M);
    const auto& B_eigen = MapMatrix<float>(const_cast<float*>(B + offB), N, K);
    C_eigen = alpha * B_eigen * A_eigen + beta * C_eigen;
  } else if (TA && !TB) {
    const auto& A_eigen = MapMatrix<float>(const_cast<float*>(A + offA), M, K);
    const auto& B_eigen = MapMatrix<float>(const_cast<float*>(B + offB), N, K);
    C_eigen = alpha * B_eigen * A_eigen.transpose() + beta * C_eigen;
  } else if (!TA && TB) {
    const auto& A_eigen = MapMatrix<float>(const_cast<float*>(A + offA), K, M);
    const auto& B_eigen = MapMatrix<float>(const_cast<float*>(B + offB), K, N);
    C_eigen = alpha * B_eigen.transpose() * A_eigen + beta * C_eigen;
  } else {
    const auto& A_eigen = MapMatrix<float>(const_cast<float*>(A + offA), M, K);
    const auto& B_eigen = MapMatrix<float>(const_cast<float*>(B + offB), K, N);
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

}  // namespace Blas

}  // namespace Shadow
