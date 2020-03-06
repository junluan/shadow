#include "context.hpp"

#include "common.hpp"

#include "util/log.hpp"

namespace Shadow {

class GPUContext : public Context {
 public:
  explicit GPUContext(const ArgumentHelper &arguments) {
    device_id_ = arguments.GetSingleArgument<int>("device_id", 0);

    check_device(device_id_);
    CUDA_CHECK(cudaSetDevice(device_id_));

    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CHECK_NOTNULL(cublas_handle_);

#if defined(USE_CUDNN)
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CHECK_NOTNULL(cudnn_handle_);
#endif
  }
  ~GPUContext() override {
    if (cublas_handle_ != nullptr) {
      CUBLAS_CHECK(cublasDestroy(cublas_handle_));
      cublas_handle_ = nullptr;
    }

#if defined(USE_CUDNN)
    if (cudnn_handle_ != nullptr) {
      CUDNN_CHECK(cudnnDestroy(cudnn_handle_));
      cudnn_handle_ = nullptr;
    }
#endif
  }

  Allocator *allocator() const override {
    return GetAllocator<DeviceType::kGPU>();
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  int device_id() const override { return device_id_; }

  void switch_device() override { CUDA_CHECK(cudaSetDevice(device_id_)); }

  void synchronize() override { CUDA_CHECK(cudaDeviceSynchronize()); }

  void *blas_handle() const override {
    CHECK_NOTNULL(cublas_handle_);
    return cublas_handle_;
  }

#if defined(USE_CUDNN)
  void *cudnn_handle() const override {
    CHECK_NOTNULL(cudnn_handle_);
    return cudnn_handle_;
  }
#endif

 private:
  static void check_device(int device_id) {
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    CHECK_GE(device_id, 0);
    CHECK_LT(device_id, num_devices);
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device_id));
    DLOG(INFO) << "GPU ID: " << device_id << ", Type: " << prop.name
               << ", Capability: " << prop.major << "." << prop.minor;
  }

  int device_id_ = 0;

  cublasHandle_t cublas_handle_ = nullptr;
#if defined(USE_CUDNN)
  cudnnHandle_t cudnn_handle_ = nullptr;
#endif
};

template <>
std::shared_ptr<Context> GetContext<DeviceType::kGPU>(
    const ArgumentHelper &arguments) {
  return std::make_shared<GPUContext>(arguments);
}

}  // namespace Shadow
