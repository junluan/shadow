#include "shadow/kernel.hpp"
#include "shadow/util/log.hpp"

#include <cmath>

namespace Kernel {

#if defined(USE_CUDA)
cublasHandle_t cublas_handle_ = nullptr;

void Setup(int device_id) {
  CheckError(cudaSetDevice(device_id));
  cublasCreate(&cublas_handle_);
}

void Release() {
  if (cublas_handle_ != nullptr) cublasDestroy(cublas_handle_);
}

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr) {
  T *buffer;
  CheckError(cudaMalloc(&buffer, size * sizeof(Dtype)));
  if (host_ptr != nullptr) {
    WriteBuffer(size, host_ptr, buffer);
  }
  return buffer;
}

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToHost));
}

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyHostToDevice));
}

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToDevice));
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  CheckError(cudaFree(buffer));
}

// Explicit instantiation
template int *MakeBuffer<int, int>(int size, int *host_ptr);
template float *MakeBuffer<float, float>(int size, float *host_ptr);

template void ReadBuffer<int, int>(int size, const int *src, int *des);
template void ReadBuffer<float, float>(int size, const float *src, float *des);

template void WriteBuffer<int, int>(int size, const int *src, int *des);
template void WriteBuffer<float, float>(int size, const float *src, float *des);

template void CopyBuffer<int, int>(int size, const int *src, int *des);
template void CopyBuffer<float, float>(int size, const float *src, float *des);

template void ReleaseBuffer<int>(int *buffer);
template void ReleaseBuffer<float>(float *buffer);

dim3 GridDim(int size) {
  unsigned int k = (unsigned int)(size - 1) / BLOCK + 1;
  unsigned int x = k;
  unsigned int y = 1;
  if (x > 65535) {
    x = (unsigned int)std::ceil(std::sqrt(k));
    y = (size - 1) / (x * BLOCK) + 1;
  }
  return dim3(x, y, 1);
}
#endif

}  // namespace Kernel
