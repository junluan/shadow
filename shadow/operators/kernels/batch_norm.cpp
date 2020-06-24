#include "batch_norm.hpp"

#include "core/blas.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <>
void BatchNorm<DeviceType::kCPU, float>(const float* in_data, int count,
                                        const float* mean_data,
                                        const float* variance_data, int channel,
                                        int inner_num, float scale_factor,
                                        float eps, float* out_data,
                                        Context* context) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_num) % channel;
    out_data[i] =
        (in_data[i] - mean_data[index] * scale_factor) /
        std::sqrt(std::abs(variance_data[index] * scale_factor) + eps);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(BatchNormCPU,
                           BatchNormKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class BatchNormKernelDNNL : public BatchNormKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& mean,
           const std::shared_ptr<Blob>& variance, std::shared_ptr<Blob>& output,
           Workspace* ws, float scale_factor, float eps) override {
    int channel = input->shape(1);

    ws->GrowTempBuffer(2 * channel * sizeof(float));

    auto mean_dnnl = ws->CreateTempBlob({channel}, DataType::kF32);
    auto variance_dnnl = ws->CreateTempBlob({channel}, DataType::kF32);

    Blas::Mul<DeviceType::kCPU, float>(
        channel, mean->data<float>(), 0, scale_factor,
        mean_dnnl->mutable_data<float>(), 0, ws->Ctx());
    Blas::Mul<DeviceType::kCPU, float>(
        channel, variance->data<float>(), 0, scale_factor,
        variance_dnnl->mutable_data<float>(), 0, ws->Ctx());

    const auto& in_out_desc = idnnl::create_memory_desc<float>(input->shape());

    const auto& batch_norm_desc =
        idnnl::create_batch_normalization_desc(in_out_desc, eps);

    idnnl::batch_normalization_forward(
        ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(), batch_norm_desc,
        input->data<float>(), mean_dnnl->data<float>(),
        variance_dnnl->data<float>(), output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(BatchNormCPU, BatchNormKernelDNNL);

#endif

}  // namespace Shadow
