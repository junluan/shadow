#include "allocator.hpp"

#include "common.hpp"

namespace Shadow {

#if defined(USE_CUDA)

class GPUAllocator : public Allocator {
 public:
  Device GetDevice() const override { return Device::kGPU; }

  void *MakeBuffer(size_t size, const void *host_ptr) const override {
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    if (host_ptr != nullptr) {
      WriteBuffer(size, host_ptr, ptr);
    }
    return ptr;
  }

  void ReadBuffer(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }

  void WriteBuffer(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }

  void CopyBuffer(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  }

  void ReleaseBuffer(void *ptr) const override { CUDA_CHECK(cudaFree(ptr)); }
};

template <>
Allocator *GetAllocator<Device::kGPU>() {
  static GPUAllocator allocator;
  return &allocator;
}

#endif

}  // namespace Shadow
