#include "kernel.hpp"
#include "util/log.hpp"

#include <cmath>

namespace Shadow {

namespace Kernel {

#if defined(USE_CUDA)
cublasHandle_t cublas_handle_ = nullptr;
#if defined(USE_CUDNN)
cudnnHandle_t cudnn_handle_ = nullptr;
#endif

void Setup(int device_id) {
  if (cublas_handle_ == nullptr) {
    CUDA_CHECK(cudaSetDevice(device_id));
    cublasCreate(&cublas_handle_);
    CHECK_NOTNULL(cublas_handle_);
  }

#if defined(USE_CUDNN)
  if (cudnn_handle_ == nullptr) {
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CHECK_NOTNULL(cudnn_handle_);
  }
#endif
}

void Release() {
  if (cublas_handle_ != nullptr) {
    cublasDestroy(cublas_handle_);
    cublas_handle_ = nullptr;
  }

#if defined(USE_CUDNN)
  if (cudnn_handle_ != nullptr) {
    CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
    cudnn_handle_ = nullptr;
  }
#endif
}

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr) {
  T *buffer;
  CUDA_CHECK(cudaMalloc(&buffer, size * sizeof(Dtype)));
  if (host_ptr != nullptr) {
    WriteBuffer(size, host_ptr, buffer);
  }
  return buffer;
}

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToHost));
}

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyHostToDevice));
}

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToDevice));
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  CUDA_CHECK(cudaFree(buffer));
}

// Explicit instantiation
template int *MakeBuffer<int, int>(int size, int *host_ptr);
template float *MakeBuffer<float, float>(int size, float *host_ptr);
template unsigned char *MakeBuffer<unsigned char, unsigned char>(
    int size, unsigned char *host_ptr);

template void ReadBuffer<int, int>(int size, const int *src, int *des);
template void ReadBuffer<float, float>(int size, const float *src, float *des);
template void ReadBuffer<unsigned char, unsigned char>(int size,
                                                       const unsigned char *src,
                                                       unsigned char *des);

template void WriteBuffer<int, int>(int size, const int *src, int *des);
template void WriteBuffer<float, float>(int size, const float *src, float *des);
template void WriteBuffer<unsigned char, unsigned char>(
    int size, const unsigned char *src, unsigned char *des);

template void CopyBuffer<int, int>(int size, const int *src, int *des);
template void CopyBuffer<float, float>(int size, const float *src, float *des);
template void CopyBuffer<unsigned char, unsigned char>(int size,
                                                       const unsigned char *src,
                                                       unsigned char *des);

template void ReleaseBuffer<int>(int *buffer);
template void ReleaseBuffer<float>(float *buffer);
template void ReleaseBuffer<unsigned char>(unsigned char *buffer);
#endif

}  // namespace Kernel

}  // namespace Shadow
