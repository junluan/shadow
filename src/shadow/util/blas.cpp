#include "shadow/util/blas.hpp"

#include <algorithm>

namespace Blas {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void SetArray(T *y, int n, float value) {
  std::fill(y, y + n, value);
}

template <typename T>
void SetArrayRepeat(T *y, int offy, int n, int value_size, const T *value) {
  for (int i = 0; i < value_size; ++i) {
    T *out_data_offset = y + offy + i * n;
    std::fill(out_data_offset, out_data_offset + n, value[i]);
  }
}

template <typename T>
void PowArray(const T *x, int n, float alpha, T *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = std::pow(x[i], alpha);
  }
}

template <typename T>
void ScaleArray(const T *x, int n, float alpha, T *y) {
  for (int i = 0; i < n; ++i) {
    y[i] = alpha * x[i];
  }
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x) {
  for (int i = 0; i < n; ++i) {
    x[i] *= alpha;
  }
}

template <typename T>
void BlasScopy(int n, const T *x, int incx, T *y, int offy, int incy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i * incy] = x[i * incx];
  }
}

template <typename T>
void BlasSasum(int n, const T *x, float *y) {
  T asum = T(0);
  for (int i = 0; i < n; ++i) {
    asum += std::abs(x[i]);
  }
  *y = asum;
}

// Level 2
inline void SgemvN(int M, int N, float alpha, const float *A, const float *x,
                   float *y) {
  for (int i = 0; i < M; ++i) {
    float sum = 0.f;
    for (int j = 0; j < N; ++j) {
      sum += alpha * A[i * N + j] * x[j];
    }
    y[i] += sum;
  }
}

inline void SgemvT(int M, int N, float alpha, const float *A, const float *x,
                   float *y) {
  for (int i = 0; i < M; ++i) {
    float sum = 0.f;
    for (int j = 0; j < N; ++j) {
      sum += alpha * A[j * M + i] * x[j];
    }
    y[i] += sum;
  }
}

template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, const T *x,
               float beta, T *y) {
  for (int i = 0; i < M; ++i) {
    y[i] *= beta;
  }
  if (!TA) {
    SgemvN(M, N, alpha, A, x, y);
  } else {
    SgemvT(M, N, alpha, A, x, y);
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
               const T *B, float beta, T *C, int offc) {
  float *C_off = C + offc;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_off[i * N + j] *= beta;
    }
  }
  if (!TA && !TB) {
    SgemmNN(M, N, K, alpha, A, B, C_off);
  } else if (TA && !TB) {
    SgemmTN(M, N, K, alpha, A, B, C_off);
  } else if (!TA && TB) {
    SgemmNT(M, N, K, alpha, A, B, C_off);
  } else {
    SgemmTT(M, N, K, alpha, A, B, C_off);
  }
}

// Explicit instantiation
template void SetArray<float>(float *y, int n, float value);
template void SetArrayRepeat<float>(float *y, int offy, int n, int value_size,
                                    const float *value);
template void PowArray<float>(const float *x, int n, float alpha, float *y);
template void ScaleArray<float>(const float *x, int n, float alpha, float *y);

// Level 1
template void BlasSscal<float>(int n, float alpha, float *x);
template void BlasScopy<float>(int n, const float *x, int incx, float *y,
                               int offy, int incy);
template void BlasSasum(int n, const float *x, float *y);

// Level 2
template void BlasSgemv<float>(int TA, int M, int N, float alpha,
                               const float *A, const float *x, float beta,
                               float *y);

// Level 3
template void BlasSgemm<float>(int TA, int TB, int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C, int offc);

#elif defined(USE_CL)
#include <clBLAS.h>

static CLKernel *cl_setarray_kernel_ = nullptr;
static CLKernel *cl_setarrayrepeat_kernel_ = nullptr;
static CLKernel *cl_powarray_kernel_ = nullptr;

void Setup() {
  std::string cl_file = "./src/shadow/util/blas.cl";
  cl_setarray_kernel_ = Kernel::easyCL->buildKernel(cl_file, "SetArray");
  cl_setarrayrepeat_kernel_ =
      Kernel::easyCL->buildKernel(cl_file, "SetArrayRepeat");
  cl_powarray_kernel_ = Kernel::easyCL->buildKernel(cl_file, "PowArray");
  clblasSetup();
}
void Release() {
  cl_setarray_kernel_->~CLKernel();
  cl_setarrayrepeat_kernel_->~CLKernel();
  cl_powarray_kernel_->~CLKernel();
  clblasTeardown();
}

template <typename T>
void SetArray(T *y, int n, float value) {
  cl_kernel kernel = cl_setarray_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 1, sizeof(int), &n);
  clSetKernelArg(kernel, 2, sizeof(float), &value);
  size_t global = n;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void SetArrayRepeat(T *y, int offy, int n, int value_size, const T *value) {
  cl_kernel kernel = cl_setarrayrepeat_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), y);
  clSetKernelArg(kernel, 1, sizeof(int), &offy);
  clSetKernelArg(kernel, 2, sizeof(int), &n);
  clSetKernelArg(kernel, 3, sizeof(int), &value_size);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), value);
  size_t global = n * value_size;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void PowArray(const T *x, int n, float alpha, T *y) {
  cl_kernel kernel = cl_powarray_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), x);
  clSetKernelArg(kernel, 1, sizeof(int), &n);
  clSetKernelArg(kernel, 2, sizeof(float), &alpha);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), y);
  size_t global = n;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void ScaleArray(const T *x, int n, float alpha, T *y) {
  BlasScopy(n, x, 1, y, 0, 1);
  BlasSscal(n, alpha, y);
}

// Level 1
template <typename T>
void BlasSscal(int n, float alpha, T *x) {
  clblasSscal(n, alpha, *x, 0, 1, 1, Kernel::easyCL->queue, 0, nullptr,
              nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasScopy(int n, const T *x, int incx, T *y, int offy, int incy) {
  clblasScopy(n, *x, 0, incx, *y, offy, incy, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasSasum(int n, const T *x, float *y) {
  cl_mem *y_ = Kernel::MakeBuffer<cl_mem>(1, static_cast<float *>(nullptr));
  cl_mem *temp_ = Kernel::MakeBuffer<cl_mem>(n, static_cast<float *>(nullptr));
  clblasSasum(n, *y_, 0, *x, 0, 1, *temp_, 1, Kernel::easyCL->queue, 0, nullptr,
              nullptr);
  clFinish(*Kernel::easyCL->queue);
  Kernel::ReadBuffer(1, y_, y);
  Kernel::ReleaseBuffer(y_);
  Kernel::ReleaseBuffer(temp_);
}

// Level 2
template <typename T>
void BlasSgemv(int TA, int M, int N, float alpha, const T *A, const T *x,
               float beta, T *y) {
  int lda = TA ? M : N;
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasSgemv(clblasRowMajor, transA, M, N, alpha, *A, 0, lda, *x, 0, 1, beta,
              *y, 0, 1, 1, Kernel::easyCL->queue, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

// Level 3
template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               const T *B, float beta, T *C, int offc) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, alpha, *A, 0, lda, *B, 0,
              ldb, beta, *C, offc, N, 1, Kernel::easyCL->queue, 0, nullptr,
              nullptr);
  clFinish(*Kernel::easyCL->queue);
}

// Explicit instantiation
template void SetArray<cl_mem>(cl_mem *data, int count, float value);
template void SetArrayRepeat<cl_mem>(cl_mem *data, int offset, int N,
                                     int value_size, const cl_mem *value);
template void PowArray<cl_mem>(const cl_mem *x, int n, float alpha, cl_mem *y);
template void ScaleArray<cl_mem>(const cl_mem *x, int count, float alpha,
                                 cl_mem *y);

// Level 1
template void BlasSscal<cl_mem>(int n, float alpha, cl_mem *x);
template void BlasScopy<cl_mem>(int n, const cl_mem *x, int incx, cl_mem *y,
                                int offy, int incy);
template void BlasSasum<cl_mem>(int n, const cl_mem *x, float *y);

// Level 2
template void BlasSgemv<cl_mem>(int TA, int M, int N, float alpha,
                                const cl_mem *A, const cl_mem *x, float beta,
                                cl_mem *y);

// Level 3
template void BlasSgemm<cl_mem>(int TA, int TB, int M, int N, int K,
                                float alpha, const cl_mem *A, const cl_mem *B,
                                float beta, cl_mem *C, int offc);
#endif

}  // namespace Blas
