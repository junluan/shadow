#include "layer_norm.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(LayerNormCPU,
                           LayerNormKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class LayerNormKernelDNNL : public LayerNormKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scale,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& normalized_shape, float eps) override {
    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));

    int inner_num = input->count(input->num_axes() - normalized_shape.size());
    int count = input->count(), outer_num = count / inner_num;

    auto* out_data = output->mutable_data<float>();

    idnnl::common_forward<dnnl::layer_normalization_forward>(
        ws->Ctx()->dnnl_handle(),
        dnnl::layer_normalization_forward::desc(
            dnnl::prop_kind::forward_inference,
            in_out_desc.reshape({outer_num, inner_num}), eps,
            dnnl::normalization_flags::none),
        input->data<float>(), out_data);

    if (scale != nullptr && bias != nullptr) {
      CHECK(scale->shape() == normalized_shape);
      CHECK(bias->shape() == normalized_shape);
      Vision::ScaleBias<DeviceType::kCPU, float>(
          out_data, count, scale->data<float>(), bias->data<float>(), inner_num,
          1, out_data, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(LayerNormCPU, LayerNormKernelDNNL);

#endif

}  // namespace Shadow
