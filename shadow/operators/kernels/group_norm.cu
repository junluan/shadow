#include "group_norm.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelSubtractMeanAndSquare(const float* in_data,
                                            const float* mean_data, int count,
                                            int inner_num, float* out_data,
                                            float* square_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    auto val = in_data[globalid] - mean_data[globalid / inner_num];
    out_data[globalid] = val;
    square_data[globalid] = val * val;
  }
}

template <>
void SubtractMeanAndSquare<DeviceType::kGPU, float>(
    const float* in_data, const float* mean_data, int count, int inner_num,
    float* out_data, float* square_data, Context* context) {
  KernelSubtractMeanAndSquare<<<GetBlocks(count), NumThreads, 0,
                                cudaStream_t(context->cuda_stream())>>>(
      in_data, mean_data, count, inner_num, out_data, square_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelDivideVariance(const float* in_data,
                                     const float* variance_data, int count,
                                     int inner_num, float eps,
                                     float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        in_data[globalid] / sqrtf(variance_data[globalid / inner_num] + eps);
  }
}

template <>
void DivideVariance<DeviceType::kGPU, float>(const float* in_data,
                                             const float* variance_data,
                                             int count, int inner_num,
                                             float eps, float* out_data,
                                             Context* context) {
  KernelDivideVariance<<<GetBlocks(count), NumThreads, 0,
                         cudaStream_t(context->cuda_stream())>>>(
      in_data, variance_data, count, inner_num, eps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GroupNormGPU,
                           GroupNormKernelDefault<DeviceType::kGPU>);

#if defined(USE_CUDNN)

class GroupNormKernelCUDNN : public GroupNormKernel {
 public:
  GroupNormKernelCUDNN() {
    cudnn::createTensorDesc<float>(&in_out_desc_);
    cudnn::createTensorDesc<float>(&param_desc_);
  }
  ~GroupNormKernelCUDNN() override {
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
           Workspace* ws, int group, float eps) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    int count = input->count();
    int outer_num = input->shape(0) * group, inner_num = count / outer_num;

    cudnn::setTensor4dDesc<float>(&in_out_desc_, 1, outer_num, inner_num, 1);
    cudnn::setTensor4dDesc<float>(&param_desc_, 1, outer_num, 1, 1);

    ws->GrowTempBuffer(2 * outer_num * sizeof(float));

    auto scale_cudnn = ws->CreateTempBlob({outer_num}, DataType::kF32);
    auto bias_cudnn = ws->CreateTempBlob({outer_num}, DataType::kF32);

    Blas::Set<DeviceType::kGPU, float>(
        outer_num, 1, scale_cudnn->mutable_data<float>(), 0, ws->Ctx());
    Blas::Set<DeviceType::kGPU, float>(
        outer_num, 0, bias_cudnn->mutable_data<float>(), 0, ws->Ctx());

    double eps_d = eps > CUDNN_BN_MIN_EPSILON ? eps : CUDNN_BN_MIN_EPSILON;

    CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(
        cudnnHandle_t(ws->Ctx()->cudnn_handle()), CUDNN_BATCHNORM_SPATIAL,
        cudnn::dataType<float>::one, cudnn::dataType<float>::zero, in_out_desc_,
        in_data, in_out_desc_, out_data, param_desc_,
        scale_cudnn->data<float>(), bias_cudnn->data<float>(), 1., nullptr,
        nullptr, eps_d, nullptr, nullptr));

    if (scale != nullptr && bias != nullptr) {
      int channel = input->shape(1), spatial_dim = input->count(2);
      CHECK_EQ(scale->count(), channel);
      CHECK_EQ(bias->count(), channel);
      Vision::ScaleBias<DeviceType::kGPU, float>(
          out_data, count, scale->data<float>(), bias->data<float>(), channel,
          spatial_dim, out_data, ws->Ctx());
    }
  }

  DeviceType device_type() const override { return DeviceType::kGPU; }

  std::string kernel_type() const override { return "CUDNN"; }

 private:
  cudnnTensorDescriptor_t in_out_desc_ = nullptr, param_desc_ = nullptr;
};

REGISTER_OP_KERNEL_CUDNN(GroupNormGPU, GroupNormKernelCUDNN);

#endif

}  // namespace Shadow
