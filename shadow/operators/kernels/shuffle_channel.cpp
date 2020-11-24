#include "shuffle_channel.hpp"

namespace Shadow {

namespace Vision {

template <>
void ShuffleChannel<DeviceType::kCPU, float>(const float* in_data, int batch,
                                             int channel, int spatial_dim,
                                             int group, float* out_data,
                                             Context* context) {
  int num = channel * spatial_dim;
  int group_column = channel / group;
  for (int b = 0; b < batch; ++b, in_data += num, out_data += num) {
    for (int c = 0; c < channel; ++c) {
      int c_out = (c % group_column) * group + c / group_column;
      memcpy(out_data + c_out * spatial_dim, in_data + c * spatial_dim,
             spatial_dim * sizeof(float));
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ShuffleChannelCPU,
                           ShuffleChannelKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class ShuffleChannelKernelDNNL : public ShuffleChannelKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int group) override {
    int channel = input->shape(1);

    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));

    idnnl::common_forward<dnnl::shuffle_forward>(
        ws->Ctx()->dnnl_handle(),
        dnnl::shuffle_forward::desc(dnnl::prop_kind::forward_inference,
                                    in_out_desc, 1, channel / group),
        input->data<float>(), output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(ShuffleChannelCPU, ShuffleChannelKernelDNNL);

#endif

}  // namespace Shadow
