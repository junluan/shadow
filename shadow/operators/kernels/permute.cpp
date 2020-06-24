#include "permute.hpp"

namespace Shadow {

namespace Vision {

template <>
void Permute<DeviceType::kCPU, float>(const float* in_data, int count,
                                      int num_axes, const int* order,
                                      const int* old_steps,
                                      const int* new_steps, float* out_data,
                                      Context* context) {
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      old_idx += (idx / new_steps[j]) * old_steps[order[j]];
      idx %= new_steps[j];
    }
    out_data[i] = in_data[old_idx];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PermuteCPU, PermuteKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class PermuteKernelDNNL : public PermuteKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& order_value) override {
    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));
    const auto& dst_desc =
        idnnl::create_memory_desc<float>(
            output->shape(), idnnl::get_memory_format(output->num_axes()))
            .permute_axes(order_value);

    const auto& reorder_desc = idnnl::create_reorder_desc(
        ws->Ctx()->dnnl_engine(), src_desc, dst_desc);

    idnnl::reorder_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                           reorder_desc, input->data<float>(),
                           output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(PermuteCPU, PermuteKernelDNNL);

#endif

}  // namespace Shadow
