#include "allocator.hpp"

#include "common.hpp"

namespace Shadow {

class GPUAllocator : public Allocator {
 public:
  DeviceType device_type() const override { return DeviceType::kGPU; }

  void *malloc(size_t size, const void *host_ptr) const override {
    void *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, size));
    if (host_ptr != nullptr) {
      write(size, host_ptr, ptr);
    }
    return ptr;
  }

  void read(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
  }

  void write(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
  }

  void copy(size_t size, const void *src, void *dst) const override {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
  }

  void free(void *ptr) const override { CUDA_CHECK(cudaFree(ptr)); }
};

template <>
Allocator *GetAllocator<DeviceType::kGPU>() {
  static GPUAllocator allocator;
  return &allocator;
}

}  // namespace Shadow
