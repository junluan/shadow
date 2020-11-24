#include "softmax.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <>
void Softmax<DeviceType::kCPU, float>(const float* in_data, int outer_num,
                                      int dim, int inner_num, float* val_data,
                                      float* out_data, Context* context) {
  int val_count = outer_num * inner_num, count = val_count * dim;

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const auto* in_data_offset = in_data + n * dim * inner_num + s;
    auto max_val = std::numeric_limits<float>::lowest();
    for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
      max_val = std::max(*in_data_offset, max_val);
    }
    val_data[i] = max_val;
  }

  for (int i = 0; i < count; ++i) {
    int n = i / dim / inner_num, s = i % inner_num;
    out_data[i] = std::exp(in_data[i] - val_data[n * inner_num + s]);
  }

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const auto* out_data_offset = out_data + n * dim * inner_num + s;
    auto sum = 0.f;
    for (int c = 0; c < dim; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[i] = sum;
  }

  for (int i = 0; i < count; ++i) {
    int n = i / dim / inner_num, s = i % inner_num;
    out_data[i] /= val_data[n * inner_num + s];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SoftmaxCPU, SoftmaxKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class SoftmaxKernelDNNL : public SoftmaxKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis) override {
    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));

    idnnl::common_forward<dnnl::softmax_forward>(
        ws->Ctx()->dnnl_handle(),
        dnnl::softmax_forward::desc(dnnl::prop_kind::forward_inference,
                                    in_out_desc, axis),
        input->data<float>(), output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(SoftmaxCPU, SoftmaxKernelDNNL);

#endif

}  // namespace Shadow
