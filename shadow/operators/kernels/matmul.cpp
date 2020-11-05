#include "matmul.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(MatMulCPU, MatMulKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class MatMulKernelDNNL : public MatMulKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input_a,
           const std::shared_ptr<Blob>& input_b, std::shared_ptr<Blob>& output,
           Workspace* ws, bool transpose_a, bool transpose_b) override {
    int num_axes_a = input_a->num_axes(), num_axes_b = input_b->num_axes();

    auto src_a_desc = idnnl::create_memory_desc<float>(
        input_a->shape(), idnnl::get_memory_format(num_axes_a));
    auto src_b_desc = idnnl::create_memory_desc<float>(
        input_b->shape(), idnnl::get_memory_format(num_axes_b));
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), idnnl::get_memory_format(output->num_axes()));

    if (transpose_a) {
      VecInt order(num_axes_a - 2);
      std::iota(order.begin(), order.end(), 0);
      order.insert(order.end(), {num_axes_a - 1, num_axes_a - 2});
      src_a_desc = src_a_desc.permute_axes(order);
    }

    if (transpose_b) {
      VecInt order(num_axes_b - 2);
      std::iota(order.begin(), order.end(), 0);
      order.insert(order.end(), {num_axes_b - 1, num_axes_b - 2});
      src_b_desc = src_b_desc.permute_axes(order);
    }

    if (num_axes_a < num_axes_b) {
      auto dims_a = src_a_desc.dims();
      dims_a.insert(dims_a.begin(), num_axes_b - num_axes_a, 1);
      src_a_desc = src_a_desc.reshape(dims_a);
    } else if (num_axes_a > num_axes_b) {
      auto dims_b = src_b_desc.dims();
      dims_b.insert(dims_b.begin(), num_axes_a - num_axes_b, 1);
      src_b_desc = src_b_desc.reshape(dims_b);
    }

    idnnl::matmul_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                          dnnl::matmul::desc(src_a_desc, src_b_desc, dst_desc),
                          input_a->data<float>(), input_b->data<float>(),
                          output->mutable_data<float>());
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(MatMulCPU, MatMulKernelDNNL);

#endif

}  // namespace Shadow
