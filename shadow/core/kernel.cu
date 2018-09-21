#include "kernel.hpp"
#include "util/log.hpp"

#include <cmath>

namespace Shadow {

namespace Kernel {

#if defined(USE_CUDA)
void Synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

template <typename T, typename Dtype>
T *MakeBuffer(size_t size, Dtype *host_ptr) {
  T *buffer;
  CUDA_CHECK(cudaMalloc(&buffer, size * sizeof(Dtype)));
  if (host_ptr != nullptr) {
    WriteBuffer(size, host_ptr, buffer);
  }
  return buffer;
}

template <typename T, typename Dtype>
void ReadBuffer(size_t size, const T *src, Dtype *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToHost));
}

template <typename T, typename Dtype>
void WriteBuffer(size_t size, const Dtype *src, T *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyHostToDevice));
}

template <typename T, typename Dtype>
void CopyBuffer(size_t size, const T *src, T *des) {
  CUDA_CHECK(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToDevice));
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  CUDA_CHECK(cudaFree(buffer));
}

// Explicit instantiation
template int *MakeBuffer<int, int>(size_t size, int *host_ptr);
template float *MakeBuffer<float, float>(size_t size, float *host_ptr);
template unsigned char *MakeBuffer<unsigned char, unsigned char>(
    size_t size, unsigned char *host_ptr);

template void ReadBuffer<int, int>(size_t size, const int *src, int *des);
template void ReadBuffer<float, float>(size_t size, const float *src,
                                       float *des);
template void ReadBuffer<unsigned char, unsigned char>(size_t size,
                                                       const unsigned char *src,
                                                       unsigned char *des);

template void WriteBuffer<int, int>(size_t size, const int *src, int *des);
template void WriteBuffer<float, float>(size_t size, const float *src,
                                        float *des);
template void WriteBuffer<unsigned char, unsigned char>(
    size_t size, const unsigned char *src, unsigned char *des);

template void CopyBuffer<int, int>(size_t size, const int *src, int *des);
template void CopyBuffer<float, float>(size_t size, const float *src,
                                       float *des);
template void CopyBuffer<unsigned char, unsigned char>(size_t size,
                                                       const unsigned char *src,
                                                       unsigned char *des);

template void ReleaseBuffer<int>(int *buffer);
template void ReleaseBuffer<float>(float *buffer);
template void ReleaseBuffer<unsigned char>(unsigned char *buffer);
#endif

}  // namespace Kernel

}  // namespace Shadow
