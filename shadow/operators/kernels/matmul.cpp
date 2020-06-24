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

    if (num_axes_a == num_axes_b) {
      auto src_a_desc = idnnl::create_memory_desc<float>(
          input_a->shape(), idnnl::get_memory_format(num_axes_a));
      auto src_b_desc = idnnl::create_memory_desc<float>(
          input_b->shape(), idnnl::get_memory_format(num_axes_b));
      const auto& dst_desc = idnnl::create_memory_desc<float>(
          output->shape(), dnnl::memory::format_tag::any);

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

      const auto& matmul_desc =
          idnnl::create_matmul_desc(src_a_desc, src_b_desc, dst_desc);

      idnnl::matmul_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                            matmul_desc, input_a->data<float>(),
                            input_b->data<float>(),
                            output->mutable_data<float>());
    } else {
      int rows_a = input_a->shape(-2), cols_a = input_a->shape(-1);
      int rows_b = input_b->shape(-2), cols_b = input_b->shape(-1);

      int outer_num = output->count(0, output->num_axes() - 2),
          inner_num = output->count(output->num_axes() - 2);
      int inner_num_a = num_axes_a >= num_axes_b ? (rows_a * cols_a) : 0;
      int inner_num_b = num_axes_a <= num_axes_b ? (rows_b * cols_b) : 0;

      auto src_a_desc = idnnl::create_memory_desc<float>(
          {rows_a, cols_a}, dnnl::memory::format_tag::ab);
      auto src_b_desc = idnnl::create_memory_desc<float>(
          {rows_b, cols_b}, dnnl::memory::format_tag::ab);
      const auto& dst_desc = idnnl::create_memory_desc<float>(
          {output->shape(-2), output->shape(-1)},
          dnnl::memory::format_tag::any);

      if (transpose_a) {
        src_a_desc = src_a_desc.permute_axes({1, 0});
      }

      if (transpose_b) {
        src_b_desc = src_b_desc.permute_axes({1, 0});
      }

      const auto& matmul_desc =
          idnnl::create_matmul_desc(src_a_desc, src_b_desc, dst_desc);

      for (int n = 0; n < outer_num; ++n) {
        idnnl::matmul_forward(ws->Ctx()->dnnl_engine(),
                              ws->Ctx()->dnnl_stream(), matmul_desc,
                              input_a->data<float>() + n * inner_num_a,
                              input_b->data<float>() + n * inner_num_b,
                              output->mutable_data<float>() + n * inner_num);
      }
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }
};

REGISTER_OP_KERNEL_DNNL(MatMulCPU, MatMulKernelDNNL);

#endif

}  // namespace Shadow
