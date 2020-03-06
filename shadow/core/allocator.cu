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
    if (cuda_stream_ == nullptr) {
      CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost,
                                 cuda_stream_));
    }
  }

  void write(size_t size, const void *src, void *dst) const override {
    if (cuda_stream_ == nullptr) {
      CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice,
                                 cuda_stream_));
    }
  }

  void copy(size_t size, const void *src, void *dst) const override {
    if (cuda_stream_ == nullptr) {
      CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToDevice,
                                 cuda_stream_));
    }
  }

  void free(void *ptr) const override { CUDA_CHECK(cudaFree(ptr)); }

  void set_stream(void *stream) override {
    cuda_stream_ = cudaStream_t(stream);
  }

 private:
  cudaStream_t cuda_stream_ = nullptr;
};

template <>
std::shared_ptr<Allocator> GetAllocator<DeviceType::kGPU>() {
  return std::make_shared<GPUAllocator>();
}

}  // namespace Shadow
