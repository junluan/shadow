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

// Level 1
template <typename T>
void BlasScopy(int n, const T *x, int incx, T *y, int offy, int incy) {
  for (int i = 0; i < n; ++i) {
    y[offy + i * incy] = x[i * incx];
  }
}

// Level 3
inline void SgemmNN(int M, int N, int K, float alpha, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = alpha * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

inline void SgemmTN(int M, int N, int K, float alpha, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = alpha * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

inline void SgemmNT(int M, int N, int K, float alpha, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

inline void SgemmTT(int M, int N, int K, float alpha, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

template <typename T>
void BlasSgemm(int TA, int TB, int M, int N, int K, float alpha, const T *A,
               const T *B, float beta, T *C, int offc) {
  int lda = TA ? M : K, ldb = TB ? K : N;
  float *C_off = C + offc;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_off[i * N + j] *= beta;
    }
  }
  if (!TA && !TB) {
    SgemmNN(M, N, K, alpha, A, lda, B, ldb, C_off, N);
  } else if (TA && !TB) {
    SgemmTN(M, N, K, alpha, A, lda, B, ldb, C_off, N);
  } else if (!TA && TB) {
    SgemmNT(M, N, K, alpha, A, lda, B, ldb, C_off, N);
  } else {
    SgemmTT(M, N, K, alpha, A, lda, B, ldb, C_off, N);
  }
}

// Explicit instantiation
template void SetArray<float>(float *y, int n, float value);
template void SetArrayRepeat<float>(float *y, int offy, int n, int value_size,
                                    const float *value);

// Level 1
template void BlasScopy<float>(int n, const float *x, int incx, float *y,
                               int offy, int incy);

// Level 3
template void BlasSgemm<float>(int TA, int TB, int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta,
                               float *C, int offc);

#elif defined(USE_CL)
#include <clBLAS.h>

static CLKernel *cl_setarray_kernel_ = nullptr;
static CLKernel *cl_setarrayrepeat_kernel_ = nullptr;

void Setup() {
  std::string cl_file = "./src/shadow/util/blas.cl";
  cl_setarray_kernel_ = Kernel::easyCL->buildKernel(cl_file, "SetArray");
  cl_setarrayrepeat_kernel_ =
      Kernel::easyCL->buildKernel(cl_file, "SetArrayRepeat");
  clblasSetup();
}
void Release() {
  cl_setarray_kernel_->~CLKernel();
  cl_setarrayrepeat_kernel_->~CLKernel();
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
  // TODO(jun) finish cl BlasSasum
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
template void ScaleArray<cl_mem>(const cl_mem *x, int count, float alpha,
                                 cl_mem *y);

// Level 1
template void BlasSscal<cl_mem>(int n, float alpha, cl_mem *x);
template void BlasScopy<cl_mem>(int n, const cl_mem *x, int incx, cl_mem *y,
                                int offy, int incy);
template void BlasSasum<cl_mem>(int n, const cl_mem *x, float *y);

// Level 3
template void BlasSgemm<cl_mem>(int TA, int TB, int M, int N, int K,
                                float alpha, const cl_mem *A, const cl_mem *B,
                                float beta, cl_mem *C, int offc);
#endif

}  // namespace Blas
