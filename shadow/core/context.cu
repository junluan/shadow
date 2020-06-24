#include "context.hpp"

#include "external.hpp"

namespace Shadow {

class GPUContext : public Context {
 public:
  explicit GPUContext(const ArgumentHelper& arguments) {
    device_id_ = arguments.GetSingleArgument<int>("device_id", 0);
    default_cuda_stream_ =
        arguments.GetSingleArgument<bool>("default_cuda_stream", false);

    check_device(device_id_);
    CUDA_CHECK(cudaSetDevice(device_id_));

    if (default_cuda_stream_) {
      cuda_stream_ = cudaStreamPerThread;
    } else {
      CUDA_CHECK(cudaStreamCreate(&cuda_stream_));
    }

    CHECK_NOTNULL(cuda_stream_);

    CUBLAS_CHECK(cublasCreate(&cublas_handle_));
    CHECK_NOTNULL(cublas_handle_);
    CUBLAS_CHECK(cublasSetStream(cublas_handle_, cuda_stream_));

#if defined(USE_CUDNN)
    CUDNN_CHECK(cudnnCreate(&cudnn_handle_));
    CHECK_NOTNULL(cudnn_handle_);
    CUDNN_CHECK(cudnnSetStream(cudnn_handle_, cuda_stream_));
#endif

    allocator_ = GetAllocator<DeviceType::kGPU>();
    allocator_->set_stream(cuda_stream_);
  }
  ~GPUContext() override {
    if (cuda_stream_ != nullptr && !default_cuda_stream_) {
      CUDA_CHECK(cudaStreamDestroy(cuda_stream_));
      cuda_stream_ = nullptr;
    }

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

  Allocator* allocator() const override { return allocator_.get(); }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  int device_id() const override { return device_id_; }

  void switch_device() override { CUDA_CHECK(cudaSetDevice(device_id_)); }

  void synchronize() override {
    CUDA_CHECK(cudaStreamSynchronize(cuda_stream_));
  }

  void* cuda_stream() const override {
    CHECK_NOTNULL(cuda_stream_);
    return cuda_stream_;
  }

  void* cublas_handle() const override {
    CHECK_NOTNULL(cublas_handle_);
    return cublas_handle_;
  }

#if defined(USE_CUDNN)
  void* cudnn_handle() const override {
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
  bool default_cuda_stream_ = false;

  std::shared_ptr<Allocator> allocator_ = nullptr;

  cudaStream_t cuda_stream_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;
#if defined(USE_CUDNN)
  cudnnHandle_t cudnn_handle_ = nullptr;
#endif
};

template <>
std::shared_ptr<Context> GetContext<DeviceType::kGPU>(
    const ArgumentHelper& arguments) {
  return std::make_shared<GPUContext>(arguments);
}

}  // namespace Shadow
