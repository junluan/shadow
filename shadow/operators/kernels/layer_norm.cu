#include "layer_norm.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(LayerNormGPU,
                           LayerNormKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class LayerNormKernelCUDNN : public LayerNormKernel {
 public:
  LayerNormKernelCUDNN() {
    cudnn::createTensorDesc<float>(&in_out_desc_);
    cudnn::createTensorDesc<float>(&param_desc_);
  }
  ~LayerNormKernelCUDNN() override {
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
           const std::shared_ptr<Blob>& scale,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& normalized_shape, float eps) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int inner_num = input->count(input->num_axes() - normalized_shape.size());
    int count = input->count(), outer_num = count / inner_num;

    cudnn::setTensor4dDesc<float>(&in_out_desc_, 1, outer_num, inner_num, 1);
    cudnn::setTensor4dDesc<float>(&param_desc_, 1, outer_num, 1, 1);

    ws->GrowTempBuffer(2 * outer_num * sizeof(float));

    auto scale_cudnn = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto bias_cudnn = ws->CreateTempBlob({outer_num}, DataType::kF32);

    Blas::Set<DeviceType::kGPU, float>(
        outer_num, 1, scale_cudnn->mutable_data<float>(), 0, ws->Ctx().get());
    Blas::Set<DeviceType::kGPU, float>(
        outer_num, 0, bias_cudnn->mutable_data<float>(), 0, ws->Ctx().get());

    double eps_d = eps > CUDNN_BN_MIN_EPSILON ? eps : CUDNN_BN_MIN_EPSILON;

    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<float>::one, cudnn::dataType<float>::zero, in_out_desc_,
        in_data, in_out_desc_, out_data, param_desc_,
        scale_cudnn->data<float>(), bias_cudnn->data<float>(), 1., nullptr,
        nullptr, eps_d, nullptr, nullptr));

    if (scale != nullptr && bias != nullptr) {
      CHECK(scale->shape() == normalized_shape);
      CHECK(bias->shape() == normalized_shape);
      Vision::ScaleBias<DeviceType::kGPU, float>(
          out_data, count, scale->data<float>(), bias->data<float>(), inner_num,
          1, out_data, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnTensorDescriptor_t in_out_desc_ = nullptr, param_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(LayerNormGPU, LayerNormKernelCUDNN);

#endif

}  // namespace Shadow
