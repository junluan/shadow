#include "shadow/util/blas.hpp"

#include <algorithm>

namespace Blas {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void Set(int n, float val, T *y, int offy) {
  std::fill(y + offy, y + offy + n, val);
}

#define BINARY_FUNC(name, operation)                                           \
  template <typename T>                                                        \
  inline void v##name(int n, const T *a, int offa, const T *b, int offb, T *y, \
                      int offy) {                                              \
    for (int i = 0; i < n; ++i) {                                              \
      y[offy + i] = a[offa + i] operation b[offb + i];                         \
    }                                                                          \
  }

BINARY_FUNC(Add, +);
BINARY_FUNC(Sub, -);
BINARY_FUNC(Mul, *);
BINARY_FUNC(Div, /);

template <typename T>
void Add(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vAdd<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Sub(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vSub<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Mul(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vMul<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Div(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vDiv<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Square(int n, const T *a, int offa, T *y, int offy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i] = a[offa + i] * a[offa + i];
  }
}

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
  for (int i = 0; i < n; ++i) {
    x[offx + i] *= alpha;
  }
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i] = x[offx + i];
  }
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i] += alpha * x[offx + i];
  }
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  double asum = 0;
  for (int i = 0; i < n; ++i) {
    asum += std::abs(x[offx + i]);
  }
  *y = static_cast<T>(asum);
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
  for (int i = 0; i < (TA ? N : M); ++i) {
    y[offy + i] *= beta;
  }
  if (!TA) {
    SgemvN(M, N, alpha, A + offA, x + offx, y + offy);
  } else {
    SgemvT(M, N, alpha, A + offA, x + offx, y + offy);
  }
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
template void Square(int n, const float *a, int offa, float *y, int offy);
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

#elif defined(USE_CL)
#include <clBLAS.h>

static CLKernel *cl_set_kernel_ = nullptr;
static CLKernel *cl_add_kernel_ = nullptr;
static CLKernel *cl_sub_kernel_ = nullptr;
static CLKernel *cl_mul_kernel_ = nullptr;
static CLKernel *cl_div_kernel_ = nullptr;
static CLKernel *cl_square_kernel_ = nullptr;
static CLKernel *cl_pow_kernel_ = nullptr;

void Setup() {
  std::string cl_file = "./src/shadow/util/blas.cl";
  cl_set_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Set");
  cl_add_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Add");
  cl_sub_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Sub");
  cl_mul_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Mul");
  cl_div_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Div");
  cl_square_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Square");
  cl_pow_kernel_ = Kernel::easyCL->buildKernel(cl_file, "Pow");
  clblasSetup();
}
void Release() {
  cl_set_kernel_->~CLKernel();
  cl_add_kernel_->~CLKernel();
  cl_sub_kernel_->~CLKernel();
  cl_mul_kernel_->~CLKernel();
  cl_div_kernel_->~CLKernel();
  cl_square_kernel_->~CLKernel();
  cl_pow_kernel_->~CLKernel();
  clblasTeardown();
}

template <typename T>
void Set(int n, float val, T *y, int offy) {
  cl_kernel kernel = cl_set_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(float), &val);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 3, sizeof(int), &offy);
  size_t global = n;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

#define BINARY_FUNC(name, kname)                                               \
  template <typename T>                                                        \
  inline void v##name(int n, const T *a, int offa, const T *b, int offb, T *y, \
                      int offy) {                                              \
    cl_kernel kernel = cl_##kname->GetKernel();                                \
    clSetKernelArg(kernel, 0, sizeof(int), &n);                                \
    clSetKernelArg(kernel, 1, sizeof(cl_mem), a);                              \
    clSetKernelArg(kernel, 2, sizeof(int), &offa);                             \
    clSetKernelArg(kernel, 3, sizeof(cl_mem), b);                              \
    clSetKernelArg(kernel, 4, sizeof(int), &offb);                             \
    clSetKernelArg(kernel, 5, sizeof(cl_mem), y);                              \
    clSetKernelArg(kernel, 6, sizeof(int), &offy);                             \
    size_t global = n;                                                         \
    clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr,         \
                           &global, nullptr, 0, nullptr, nullptr);             \
    clFinish(*Kernel::easyCL->queue);                                          \
  }

BINARY_FUNC(Add, add_kernel_);
BINARY_FUNC(Sub, sub_kernel_);
BINARY_FUNC(Mul, mul_kernel_);
BINARY_FUNC(Div, div_kernel_);

template <typename T>
void Add(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vAdd<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Sub(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vSub<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Mul(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vMul<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Div(int n, const T *a, int offa, const T *b, int offb, T *y, int offy) {
  vDiv<T>(n, a, offa, b, offb, y, offy);
}

template <typename T>
void Square(int n, const T *a, int offa, T *y, int offy) {
  cl_kernel kernel = cl_square_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), a);
  clSetKernelArg(kernel, 2, sizeof(int), &offa);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 4, sizeof(int), &offy);
  size_t global = n;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void Pow(int n, const T *a, int offa, float alpha, T *y, int offy) {
  cl_kernel kernel = cl_pow_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &n);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), a);
  clSetKernelArg(kernel, 2, sizeof(int), &offa);
  clSetKernelArg(kernel, 3, sizeof(float), &alpha);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 5, sizeof(int), &offy);
  size_t global = n;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void Scale(int n, float alpha, const T *x, int offx, T *y, int offy) {
  BlasScopy(n, x, offx, y, offy);
  BlasSscal(n, alpha, y, offy);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x, int offx) {
  clblasSscal(n, alpha, *x, offx, 1, 1, Kernel::easyCL->queue, 0, nullptr,
              nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasScopy(int n, const T *x, int offx, T *y, int offy) {
  clblasScopy(n, *x, offx, 1, *y, offy, 1, 1, Kernel::easyCL->queue, 0, nullptr,
              nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasSaxpy(int n, float alpha, const T *x, int offx, T *y, int offy) {
  clblasSaxpy(n, alpha, *x, offx, 1, *y, offy, 1, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasSasum(int n, const T *x, int offx, float *y) {
  cl_mem *y_ = Kernel::MakeBuffer<cl_mem>(1, static_cast<float *>(nullptr));
  cl_mem *temp_ = Kernel::MakeBuffer<cl_mem>(n, static_cast<float *>(nullptr));
  clblasSasum(n, *y_, 0, *x, offx, 1, *temp_, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
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
              beta, *y, offy, 1, 1, Kernel::easyCL->queue, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               int offA, const T *B, int offB, float beta, T *C, int offC) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, alpha, *A, offA, lda, *B,
              offB, ldb, beta, *C, offC, N, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

// Explicit instantiation
template void Set<cl_mem>(int n, float val, cl_mem *y, int offy);
template void Add<cl_mem>(int n, const cl_mem *a, int offa, const cl_mem *b,
                          int offb, cl_mem *y, int offy);
template void Sub<cl_mem>(int n, const cl_mem *a, int offa, const cl_mem *b,
                          int offb, cl_mem *y, int offy);
template void Mul<cl_mem>(int n, const cl_mem *a, int offa, const cl_mem *b,
                          int offb, cl_mem *y, int offy);
template void Div<cl_mem>(int n, const cl_mem *a, int offa, const cl_mem *b,
                          int offb, cl_mem *y, int offy);
template void Square<cl_mem>(int n, const cl_mem *a, int offa, cl_mem *y,
                             int offy);
template void Pow<cl_mem>(int n, const cl_mem *a, int offa, float alpha,
                          cl_mem *y, int offy);
template void Scale<cl_mem>(int n, float alpha, const cl_mem *x, int offx,
                            cl_mem *y, int offy);

// Level 1
template void BlasSscal<cl_mem>(int n, float alpha, cl_mem *x, int offx);
template void BlasScopy<cl_mem>(int n, const cl_mem *x, int offx, cl_mem *y,
                                int offy);
template void BlasSaxpy<cl_mem>(int n, float alpha, const cl_mem *x, int offx,
                                cl_mem *y, int offy);
template void BlasSasum<cl_mem>(int n, const cl_mem *x, int offx, float *y);

// Level 2
template void BlasSgemv<cl_mem>(int TA, int M, int N, float alpha,
                                const cl_mem *A, int offA, const cl_mem *x,
                                int offx, float beta, cl_mem *y, int offy);

// Level 3
template void BlasSgemm<cl_mem>(int TA, int TB, int M, int N, int K,
                                float alpha, const cl_mem *A, int offA,
                                const cl_mem *B, int offB, float beta,
                                cl_mem *C, int offC);
#endif

}  // namespace Blas
