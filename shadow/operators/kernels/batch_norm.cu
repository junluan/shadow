#include "batch_norm.hpp"

#include "core/blas.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelBatchNorm(const float* in_data, int count,
                                const float* mean_data,
                                const float* variance_data, int channel,
                                int inner_num, float scale_factor, float eps,
                                float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_num) % channel;
    out_data[globalid] =
        (in_data[globalid] - mean_data[index] * scale_factor) /
        sqrtf(fabsf(variance_data[index] * scale_factor) + eps);
  }
}

template <>
void BatchNorm<DeviceType::kGPU, float>(const float* in_data, int count,
                                        const float* mean_data,
                                        const float* variance_data, int channel,
                                        int inner_num, float scale_factor,
                                        float eps, float* out_data,
                                        Context* context) {
  KernelBatchNorm<<<GetBlocks(count), NumThreads, 0,
                    cudaStream_t(context->cuda_stream())>>>(
      in_data, count, mean_data, variance_data, channel, inner_num,
      scale_factor, eps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(BatchNormGPU,
                           BatchNormKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class BatchNormKernelCUDNN : public BatchNormKernel {
 public:
  BatchNormKernelCUDNN() {
    cudnn::createTensorDesc<float>(&in_out_desc_);
    cudnn::createTensorDesc<float>(&param_desc_);
  }
  ~BatchNormKernelCUDNN() override {
    if (in_out_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(in_out_desc_);
      in_out_desc_ = nullptr;
    }
    if (param_desc_ != nullptr) {
      cudnnDestroyTensorDescriptor(param_desc_);
      param_desc_ = nullptr;
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& mean,
           const std::shared_ptr<Blob>& variance, std::shared_ptr<Blob>& output,
           Workspace* ws, float scale_factor, float eps) override {
    int batch = input->shape(0), channel = input->shape(1),
        inner_num = input->count(2);

    CHECK_EQ(channel, mean->count());
    CHECK_EQ(channel, variance->count());

    cudnn::setTensor4dDesc<float>(&in_out_desc_, batch, channel, inner_num, 1);
    cudnn::setTensor4dDesc<float>(&param_desc_, 1, channel, 1, 1);

    ws->GrowTempBuffer(4 * channel * sizeof(float));

    auto mean_cudnn = ws->CreateTempBlob({channel}, DataType::kF32);
    auto variance_cudnn = ws->CreateTempBlob({channel}, DataType::kF32);
    auto scale_cudnn = ws->CreateTempBlob({channel}, DataType::kF32);
    auto bias_cudnn = ws->CreateTempBlob({channel}, DataType::kF32);

    Blas::Mul<DeviceType::kGPU, float>(
        channel, mean->data<float>(), 0, scale_factor,
        mean_cudnn->mutable_data<float>(), 0, ws->Ctx().get());
    Blas::Mul<DeviceType::kGPU, float>(
        channel, variance->data<float>(), 0, scale_factor,
        variance_cudnn->mutable_data<float>(), 0, ws->Ctx().get());

    Blas::Set<DeviceType::kGPU, float>(
        channel, 1, scale_cudnn->mutable_data<float>(), 0, ws->Ctx().get());
    Blas::Set<DeviceType::kGPU, float>(
        channel, 0, bias_cudnn->mutable_data<float>(), 0, ws->Ctx().get());

    double eps_d = eps > CUDNN_BN_MIN_EPSILON ? eps : CUDNN_BN_MIN_EPSILON;

    CUDNN_CHECK(cudnnBatchNormalizationForwardInference(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<float>::one, cudnn::dataType<float>::zero, in_out_desc_,
        input->data<float>(), in_out_desc_, output->mutable_data<float>(),
        param_desc_, scale_cudnn->data<float>(), bias_cudnn->data<float>(),
        mean_cudnn->data<float>(), variance_cudnn->data<float>(), eps_d));
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnTensorDescriptor_t in_out_desc_ = nullptr, param_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(BatchNormGPU, BatchNormKernelCUDNN);

#endif

}  // namespace Shadow
