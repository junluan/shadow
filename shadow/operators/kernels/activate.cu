#include "activate.hpp"

namespace Shadow {

namespace Vision {

__device__ float ActivateValue(float x, int type, float slope) {
  switch (type) {
    case kRelu:
      return x > 0 ? x : 0;
    case kLeaky:
      return x > 0 ? x : slope * x;
    case kSigmoid:
      return 1 / (1 + expf(-x));
    case kSoftPlus:
      return logf(1 + expf(x));
    case kTanh: {
      auto exp_2x = expf(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    case kRelu6: {
      x = x > 0 ? x : 0;
      return x < 6 ? x : 6;
    }
    default:
      return x;
  }
}

__global__ void KernelActivate(const float* in_data, float* out_data, int count,
                               int type, float slope) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] = ActivateValue(in_data[globalid], type, slope);
  }
}

template <>
void Activate<DeviceType::kGPU, float>(const float* in_data, float* out_data,
                                       int count, int type, float slope,
                                       Context* context) {
  KernelActivate<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->cuda_stream())>>>(in_data, out_data,
                                                           count, type, slope);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelPRelu(const float* in_data, float* out_data, int count,
                            int channels, int dim, int div_factor,
                            const float* slope_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c = (globalid / dim) % channels / div_factor;
    auto value = in_data[globalid];
    out_data[globalid] = value > 0 ? value : value * slope_data[c];
  }
}

template <>
void PRelu<DeviceType::kGPU, float>(const float* in_data, float* out_data,
                                    const VecInt& in_shape, bool channel_shared,
                                    const float* slope_data, Context* context) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  KernelPRelu<<<GetBlocks(count), NumThreads, 0,
                cudaStream_t(context->cuda_stream())>>>(
      in_data, out_data, count, channels, dim, div_factor, slope_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ActivateGPU,
                           ActivateKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class ActivateKernelCUDNN : public ActivateKernel {
 public:
  ActivateKernelCUDNN() {
    cudnn::createActivationDesc<float>(&activate_desc_);
    cudnn::createTensorDesc<float>(&in_out_desc_);
    default_kernel_ =
        std::make_shared<ActivateKernelDefault<DeviceType::kGPU>>();
  }
  ~ActivateKernelCUDNN() override {
    if (activate_desc_ != nullptr) {
      cudnnDestroyActivationDescriptor(activate_desc_);
      activate_desc_ = nullptr;
    }
    if (in_out_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(in_out_desc_);
      in_out_desc_ = nullptr;
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& slope, std::shared_ptr<Blob>& output,
           Workspace* ws, int activate_type, float slope_val) override {
    if (activate_type == kRelu || activate_type == kSigmoid ||
        activate_type == kTanh) {
      int batch = input->shape(0), num = input->num();

      cudnn::setActivationDesc<float>(&activate_desc_, activate_type,
                                      static_cast<double>(slope_val));
      cudnn::setTensor4dDesc<float>(&in_out_desc_, batch, num, 1, 1);

      CUDNN_CHECK(cudnnActivationForward(
          cudnnHandle_t(ws->Ctx()->cudnn_handle()), activate_desc_,
          cudnn::dataType<float>::one, in_out_desc_, input->data<float>(),
          cudnn::dataType<float>::zero, in_out_desc_,
          output->mutable_data<float>()));
    } else {
      default_kernel_->Run(input, slope, output, ws, activate_type, slope_val);
    }
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnActivationDescriptor_t activate_desc_ = nullptr;
  cudnnTensorDescriptor_t in_out_desc_ = nullptr;

  std::shared_ptr<ActivateKernelDefault<DeviceType::kGPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_CUDNN(ActivateGPU, ActivateKernelCUDNN);

#endif

}  // namespace Shadow
