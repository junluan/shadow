#include "blas.hpp"

#include "common.hpp"

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

#if !defined(USE_CUDA)
template <typename T>
void Set(int n, float val, T *y, int offy) {
#if defined(USE_Eigen)
  auto y_eigen = MapVector<T>(y + offy, n);
  y_eigen.setConstant(static_cast<T>(val));
#else
  std::fill(y + offy, y + offy + n, static_cast<T>(val));
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
  template void name(int, const float *, int, const float *, int, float *, int);

#define BLAS_BINARY_SCALAR_FUNC_EIGEN(name, operation)                  \
  template <typename T>                                                 \
  void name(int n, const T *a, int offa, float alpha, T *y, int offy) { \
    const auto &a_eigen = MapVector<T>(const_cast<T *>(a + offa), n);   \
    auto y_eigen = MapVector<T>(y + offy, n);                           \
    operation;                                                          \
  }                                                                     \
  template void name(int, const float *, int, float, float *, int);

#define BLAS_UNARY_FUNC_EIGEN(name, operation)                        \
  template <typename T>                                               \
  void name(int n, const T *a, int offa, T *y, int offy) {            \
    const auto &a_eigen = MapVector<T>(const_cast<T *>(a + offa), n); \
    auto y_eigen = MapVector<T>(y + offy, n);                         \
    operation;                                                        \
  }                                                                   \
  template void name(int, const float *, int, float *, int);

BLAS_BINARY_FUNC_EIGEN(Add, y_eigen = a_eigen.array() + b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Sub, y_eigen = a_eigen.array() - b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Mul, y_eigen = a_eigen.array() * b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Div, y_eigen = a_eigen.array() / b_eigen.array());
BLAS_BINARY_FUNC_EIGEN(Pow, y_eigen = a_eigen.array().pow(b_eigen.array()));
BLAS_BINARY_FUNC_EIGEN(Max, y_eigen = a_eigen.cwiseMax(b_eigen));
BLAS_BINARY_FUNC_EIGEN(Min, y_eigen = a_eigen.cwiseMin(b_eigen));

BLAS_BINARY_SCALAR_FUNC_EIGEN(Add, y_eigen = a_eigen.array() + alpha);
BLAS_BINARY_SCALAR_FUNC_EIGEN(Sub, y_eigen = a_eigen.array() - alpha);
BLAS_BINARY_SCALAR_FUNC_EIGEN(Mul, y_eigen = a_eigen.array() * alpha);
BLAS_BINARY_SCALAR_FUNC_EIGEN(Div, y_eigen = a_eigen.array() / alpha);
BLAS_BINARY_SCALAR_FUNC_EIGEN(Pow, y_eigen = a_eigen.array().pow(alpha));
BLAS_BINARY_SCALAR_FUNC_EIGEN(Max, y_eigen = a_eigen.cwiseMax(alpha));
BLAS_BINARY_SCALAR_FUNC_EIGEN(Min, y_eigen = a_eigen.cwiseMin(alpha));

BLAS_UNARY_FUNC_EIGEN(Abs, y_eigen = a_eigen.array().abs());
BLAS_UNARY_FUNC_EIGEN(Square, y_eigen = a_eigen.array().square());
BLAS_UNARY_FUNC_EIGEN(Sqrt, y_eigen = a_eigen.array().sqrt());
BLAS_UNARY_FUNC_EIGEN(Log, y_eigen = a_eigen.array().log());
BLAS_UNARY_FUNC_EIGEN(Exp, y_eigen = a_eigen.array().exp());
BLAS_UNARY_FUNC_EIGEN(Sin, y_eigen = a_eigen.array().sin());
BLAS_UNARY_FUNC_EIGEN(Cos, y_eigen = a_eigen.array().cos());
BLAS_UNARY_FUNC_EIGEN(Tan, y_eigen = a_eigen.array().tan());
BLAS_UNARY_FUNC_EIGEN(Asin, y_eigen = a_eigen.array().asin());
BLAS_UNARY_FUNC_EIGEN(Acos, y_eigen = a_eigen.array().acos());
BLAS_UNARY_FUNC_EIGEN(Atan, y_eigen = a_eigen.array().atan());
BLAS_UNARY_FUNC_EIGEN(Floor, y_eigen = a_eigen.array().floor());
BLAS_UNARY_FUNC_EIGEN(Ceil, y_eigen = a_eigen.array().ceil());
BLAS_UNARY_FUNC_EIGEN(Neg, y_eigen = -a_eigen.array());
BLAS_UNARY_FUNC_EIGEN(Reciprocal, y_eigen = a_eigen.array().inverse());

#else
#define BLAS_BINARY_FUNC(name, operation)                            \
  template <typename T>                                              \
  void name(int n, const T *a, int offa, const T *b, int offb, T *y, \
            int offy) {                                              \
    a += offa, b += offb, y += offy;                                 \
    for (int i = 0; i < n; ++i) {                                    \
      operation;                                                     \
    }                                                                \
  }                                                                  \
  template void name(int, const float *, int, const float *, int, float *, int);

#define BLAS_BINARY_SCALAR_FUNC(name, operation)                        \
  template <typename T>                                                 \
  void name(int n, const T *a, int offa, float alpha, T *y, int offy) { \
    a += offa, y += offy;                                               \
    for (int i = 0; i < n; ++i) {                                       \
      operation;                                                        \
    }                                                                   \
  }                                                                     \
  template void name(int, const float *, int, float, float *, int);

#define BLAS_UNARY_FUNC(name, operation)                   \
  template <typename T>                                    \
  void name(int n, const T *a, int offa, T *y, int offy) { \
    a += offa, y += offy;                                  \
    for (int i = 0; i < n; ++i) {                          \
      operation;                                           \
    }                                                      \
  }                                                        \
  template void name(int, const float *, int, float *, int);

BLAS_BINARY_FUNC(Add, y[i] = a[i] + b[i]);
BLAS_BINARY_FUNC(Sub, y[i] = a[i] - b[i]);
BLAS_BINARY_FUNC(Mul, y[i] = a[i] * b[i]);
BLAS_BINARY_FUNC(Div, y[i] = a[i] / b[i]);
BLAS_BINARY_FUNC(Pow, y[i] = std::pow(a[i], b[i]));
BLAS_BINARY_FUNC(Max, y[i] = std::max(a[i], b[i]));
BLAS_BINARY_FUNC(Min, y[i] = std::min(a[i], b[i]));

BLAS_BINARY_SCALAR_FUNC(Add, y[i] = a[i] + alpha);
BLAS_BINARY_SCALAR_FUNC(Sub, y[i] = a[i] - alpha);
BLAS_BINARY_SCALAR_FUNC(Mul, y[i] = a[i] * alpha);
BLAS_BINARY_SCALAR_FUNC(Div, y[i] = a[i] / alpha);
BLAS_BINARY_SCALAR_FUNC(Pow, y[i] = std::pow(a[i], alpha));
BLAS_BINARY_SCALAR_FUNC(Max, y[i] = std::max(a[i], alpha));
BLAS_BINARY_SCALAR_FUNC(Min, y[i] = std::min(a[i], alpha));

BLAS_UNARY_FUNC(Abs, y[i] = std::abs(a[i]));
BLAS_UNARY_FUNC(Square, y[i] = a[i] * a[i]);
BLAS_UNARY_FUNC(Sqrt, y[i] = std::sqrt(a[i]));
BLAS_UNARY_FUNC(Log, y[i] = std::log(a[i]));
BLAS_UNARY_FUNC(Exp, y[i] = std::exp(a[i]));
BLAS_UNARY_FUNC(Sin, y[i] = std::sin(a[i]));
BLAS_UNARY_FUNC(Cos, y[i] = std::cos(a[i]));
BLAS_UNARY_FUNC(Tan, y[i] = std::tan(a[i]));
BLAS_UNARY_FUNC(Asin, y[i] = std::asin(a[i]));
BLAS_UNARY_FUNC(Acos, y[i] = std::acos(a[i]));
BLAS_UNARY_FUNC(Atan, y[i] = std::atan(a[i]));
BLAS_UNARY_FUNC(Floor, y[i] = std::floor(a[i]));
BLAS_UNARY_FUNC(Ceil, y[i] = std::ceil(a[i]));
BLAS_UNARY_FUNC(Neg, y[i] = -a[i]);
BLAS_UNARY_FUNC(Reciprocal, y[i] = 1 / a[i]);
#endif

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx, void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
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
void BlasScopy(int n, const T *x, int offx, T *y, int offy, void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
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
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy,
               void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
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
void BlasSasum(int n, const T *x, int offx, float *y, void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
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
               const T *x, int offx, float beta, T *y, int offy, void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  auto transA = TA ? CblasTrans : CblasNoTrans;
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
               int offA, const T *B, int offB, float beta, T *C, int offC,
               void *ctx) {
#if defined(USE_OpenBLAS) | defined(USE_MKL)
  int lda = TA ? M : K, ldb = TB ? K : N;
  auto transA = TA ? CblasTrans : CblasNoTrans;
  auto transB = TB ? CblasTrans : CblasNoTrans;
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
template void Set(int, float, float *, int);

// Level 1
template void BlasSscal(int, float, float *, int, void *);
template void BlasScopy(int, const float *, int, float *, int, void *);
template void BlasSaxpy(int, float, const float *, int, float *, int, void *);
template void BlasSasum(int, const float *, int, float *, void *);

// Level 2
template void BlasSgemv(int, int, int, float, const float *, int, const float *,
                        int, float, float *, int, void *);

// Level 3
template void BlasSgemm(int, int, int, int, int, float, const float *, int,
                        const float *, int, float, float *, int, void *);
#endif

}  // namespace Blas

}  // namespace Shadow
