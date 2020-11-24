#include "concat.hpp"

namespace Shadow {

namespace Vision {

template <>
void Concat<DeviceType::kCPU, float>(const float* in_data, int count,
                                     int num_concats, int concat_size,
                                     int out_concat_axis, int in_concat_axis,
                                     int offset_concat_axis, float* out_data,
                                     Context* context) {
  for (int n = 0; n < num_concats; ++n) {
    memcpy(out_data + (n * out_concat_axis + offset_concat_axis) * concat_size,
           in_data + n * in_concat_axis * concat_size,
           in_concat_axis * concat_size * sizeof(float));
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConcatCPU, ConcatKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class ConcatKernelDNNL : public ConcatKernel {
 public:
  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::shared_ptr<Blob>& output, Workspace* ws, int axis) override {
    std::vector<dnnl::memory::desc> srcs_desc;
    std::vector<const void*> srcs_data;
    for (const auto& input : inputs) {
      srcs_desc.push_back(idnnl::create_memory_desc<float>(
          input->shape(), idnnl::get_memory_format(input->num_axes())));
      srcs_data.push_back(input->data<float>());
    }

    idnnl::concat_forward(
        ws->Ctx()->dnnl_handle(),
        dnnl::concat::primitive_desc(
            axis, srcs_desc,
            static_cast<dnnl::stream*>(ws->Ctx()->dnnl_handle())->get_engine()),
        srcs_data, output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(ConcatCPU, ConcatKernelDNNL);

#endif

}  // namespace Shadow
