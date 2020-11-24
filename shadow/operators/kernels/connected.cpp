#include "connected.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConnectedCPU,
                           ConnectedKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class ConnectedKernelDNNL : public ConnectedKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& weight,
           const std::shared_ptr<Blob>& bias, std::shared_ptr<Blob>& output,
           Workspace* ws, int num_output, bool bias_term,
           bool transpose) override {
    int batch = input->shape(0), inner_num = input->count(1);

    const auto& src_desc = idnnl::create_memory_desc<float>(
        {batch, inner_num}, dnnl::memory::format_tag::nc);
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        {batch, num_output}, dnnl::memory::format_tag::nc);
    const auto& weight_desc = idnnl::create_memory_desc<float>(
        {num_output, inner_num}, transpose ? dnnl::memory::format_tag::oi
                                           : dnnl::memory::format_tag::io);
    const auto& bias_desc = idnnl::create_memory_desc<float>(
        {num_output}, bias_term ? dnnl::memory::format_tag::x
                                : dnnl::memory::format_tag::undef);

    idnnl::inner_product_forward(
        ws->Ctx()->dnnl_handle(),
        dnnl::inner_product_forward::desc(dnnl::prop_kind::forward_inference,
                                          src_desc, weight_desc, bias_desc,
                                          dst_desc),
        input->data<float>(), weight->data<float>(),
        bias_term ? bias->data<float>() : nullptr,
        output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(ConnectedCPU, ConnectedKernelDNNL);

#endif

}  // namespace Shadow
