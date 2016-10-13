#include "shadow/util/blas.hpp"

#include <algorithm>

namespace Blas {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void SetArray(T *data, int count, float value) {
  std::fill(data, data + count, value);
}

template <typename T>
void SetArrayRepeat(T *data, int offset, int N, int value_size,
                    const T *value) {
  for (int i = 0; i < value_size; ++i) {
    T *out_data_offset = data + offset + i * N;
    std::fill(out_data_offset, out_data_offset + N, value[i]);
  }
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  for (int i = 0; i < N; ++i) {
    Y[offset + i * incy] = X[i * incx];
  }
}

inline void SGemmNN(int M, int N, int K, float ALPHA, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

inline void SGemmNT(int M, int N, int K, float ALPHA, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

inline void SGemmTN(int M, int N, int K, float ALPHA, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

inline void SGemmTT(int M, int N, int K, float ALPHA, const float *A, int lda,
                    const float *B, int ldb, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *A,
               int lda, const T *B, int ldb, float BETA, T *C, int offset,
               int ldc) {
  float *C_off = C + offset;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C_off[i * ldc + j] *= BETA;
    }
  }
  if (!TA && !TB)
    SGemmNN(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else if (TA && !TB)
    SGemmTN(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else if (!TA && TB)
    SGemmNT(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
  else
    SGemmTT(M, N, K, ALPHA, A, lda, B, ldb, C_off, ldc);
}

// Explicit instantiation
template void SetArray<float>(float *data, int count, float value);
template void SetArrayRepeat<float>(float *data, int offset, int N,
                                    int value_size, const float *value);
template void BlasCopy<float>(int N, const float *X, int incx, float *Y,
                              int offset, int incy);
template void BlasSGemm<float>(int TA, int TB, int M, int N, int K, float ALPHA,
                               const float *A, int lda, const float *B, int ldb,
                               float BETA, float *C, int offset, int ldc);

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
void SetArray(T *data, int count, float value) {
  cl_kernel kernel = cl_setarray_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(float), &value);
  size_t global = count;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void SetArrayRepeat(T *data, int offset, int N, int value_size,
                    const T *value) {
  cl_kernel kernel = cl_setarrayrepeat_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &N);
  clSetKernelArg(kernel, 3, sizeof(int), &value_size);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), value);
  size_t global = N * value_size;
  clEnqueueNDRangeKernel(*Kernel::easyCL->queue, kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish(*Kernel::easyCL->queue);
}

template <typename T>
void BlasCopy(int N, const T *X, int incx, T *Y, int offset, int incy) {
  clblasScopy(N, *X, 0, incx, *Y, offset, incy, 1, Kernel::easyCL->queue, 0,
              nullptr, nullptr);
}

template <typename T>
void BlasSGemm(int TA, int TB, int M, int N, int K, float ALPHA, const T *bufA,
               int lda, const T *bufB, int ldb, float BETA, T *bufC, int offset,
               int ldc) {
  clblasTranspose transA = TA ? clblasTrans : clblasNoTrans;
  clblasTranspose transB = TB ? clblasTrans : clblasNoTrans;
  clblasSgemm(clblasRowMajor, transA, transB, M, N, K, ALPHA, *bufA, 0, lda,
              *bufB, 0, ldb, BETA, *bufC, offset, ldc, 1, Kernel::easyCL->queue,
              0, nullptr, nullptr);
}

// Explicit instantiation
template void SetArray<cl_mem>(cl_mem *data, int count, float value);
template void SetArrayRepeat<cl_mem>(cl_mem *data, int offset, int N,
                                     int value_size, const cl_mem *value);
template void BlasCopy<cl_mem>(int N, const cl_mem *X, int incx, cl_mem *Y,
                               int offset, int incy);
template void BlasSGemm<cl_mem>(int TA, int TB, int M, int N, int K,
                                float ALPHA, const cl_mem *A, int lda,
                                const cl_mem *B, int ldb, float BETA, cl_mem *C,
                                int offset, int ldc);
#endif

}  // namespace Blas
