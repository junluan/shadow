#include "allocator.hpp"

#include "common.hpp"

namespace Shadow {

#if defined(USE_CUDA)

template <typename T>
void *Allocator::MakeBuffer(size_t size, const void *host_ptr, int align) {
  T *device_ptr;
  CUDA_CHECK(cudaMalloc(&device_ptr, size * sizeof(T)));
  if (host_ptr != nullptr) {
    WriteBuffer<T>(size, host_ptr, device_ptr);
  }
  return device_ptr;
}

template <typename T>
void Allocator::ReadBuffer(size_t size, const void *src, void *des) {
  CUDA_CHECK(cudaMemcpy(des, src, size * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void Allocator::WriteBuffer(size_t size, const void *src, void *des) {
  CUDA_CHECK(cudaMemcpy(des, src, size * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void Allocator::CopyBuffer(size_t size, const void *src, void *des) {
  CUDA_CHECK(cudaMemcpy(des, src, size * sizeof(T), cudaMemcpyDeviceToDevice));
}

template <typename T>
void Allocator::ReleaseBuffer(void *device_ptr) {
  CUDA_CHECK(cudaFree(device_ptr));
}

#define INSTANTIATE_ALLOCATOR(T)                                            \
  template void *Allocator::MakeBuffer<T>(size_t, const void *, int align); \
  template void Allocator::ReadBuffer<T>(size_t, const void *, void *);     \
  template void Allocator::WriteBuffer<T>(size_t, const void *, void *);    \
  template void Allocator::CopyBuffer<T>(size_t, const void *, void *);     \
  template void Allocator::ReleaseBuffer<T>(void *);

INSTANTIATE_ALLOCATOR(int)
INSTANTIATE_ALLOCATOR(float)
INSTANTIATE_ALLOCATOR(unsigned char)
#undef INSTANTIATE_BUFFER_OPERATION

#endif

}  // namespace Shadow
