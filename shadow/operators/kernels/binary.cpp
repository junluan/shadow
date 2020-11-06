#include "binary.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float Binary(float a, float b, int operation) {
  switch (operation) {
    case kAdd:
      return a + b;
    case kSub:
      return a - b;
    case kMul:
      return a * b;
    case kDiv:
      return a / b;
    case kPow:
      return std::pow(a, b);
    case kMax:
      return std::max(a, b);
    case kMin:
      return std::min(a, b);
    default:
      return 0;
  }
}

template <>
void BroadcastBinary<DeviceType::kCPU, float>(
    const float* in_data, const int* in_shape, const float* scalar_data,
    const int* scalar_shape, int operation, int num_axes, int count,
    const int* out_shape, float* out_data, Context* context) {
  VecInt in_shape_acc(1, 1), scalar_shape_acc(1, 1);
  for (int n = num_axes - 1; n > 0; --n) {
    in_shape_acc.insert(in_shape_acc.begin(), in_shape[n] * in_shape_acc[0]);
    scalar_shape_acc.insert(scalar_shape_acc.begin(),
                            scalar_shape[n] * scalar_shape_acc[0]);
  }
  for (int i = 0; i < count; ++i) {
    int in_index = 0, scalar_index = 0, cc = i;
    for (int n = num_axes - 1; n >= 0; --n) {
      int dim = cc % out_shape[n];
      in_index += (dim % in_shape[n]) * in_shape_acc[n];
      scalar_index += (dim % scalar_shape[n]) * scalar_shape_acc[n];
      cc /= out_shape[n];
    }
    out_data[i] =
        Binary(in_data[in_index], scalar_data[scalar_index], operation);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(BinaryCPU, BinaryKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class BinaryKernelDNNL : public BinaryKernel {
 public:
  BinaryKernelDNNL() {
    default_kernel_ = std::make_shared<BinaryKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, float scalar_value) override {
    int num_axes = input->num_axes();

    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(num_axes));
    const auto& scalar_desc = idnnl::create_memory_desc<float>(
        std::vector<int>(num_axes, 1), idnnl::get_memory_format(num_axes));
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), idnnl::get_memory_format(num_axes));

    try {
      idnnl::binary_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                            dnnl::binary::desc(get_algorithm(operation),
                                               src_desc, scalar_desc, dst_desc),
                            input->data<float>(), &scalar_value,
                            output->mutable_data<float>());
    } catch (std::exception& e) {
      default_kernel_->Run(input, output, ws, operation, scalar_value);
    }
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& scalar, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, bool need_broadcast) override {
    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));
    const auto& scalar_desc = idnnl::create_memory_desc<float>(
        scalar->shape(), idnnl::get_memory_format(scalar->num_axes()));
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), idnnl::get_memory_format(output->num_axes()));

    try {
      idnnl::binary_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                            dnnl::binary::desc(get_algorithm(operation),
                                               src_desc, scalar_desc, dst_desc),
                            input->data<float>(), scalar->data<float>(),
                            output->mutable_data<float>());
    } catch (std::exception& e) {
      default_kernel_->Run(input, scalar, output, ws, operation,
                           need_broadcast);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  static dnnl::algorithm get_algorithm(int operation) {
    switch (operation) {
      case kAdd:
        return dnnl::algorithm::binary_add;
      case kSub:
        return dnnl::algorithm::binary_sub;
      case kMul:
        return dnnl::algorithm::binary_mul;
      case kDiv:
        return dnnl::algorithm::binary_div;
      case kMax:
        return dnnl::algorithm::binary_max;
      case kMin:
        return dnnl::algorithm::binary_min;
      default:
        return dnnl::algorithm::undef;
    }
  }

  std::shared_ptr<BinaryKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(BinaryCPU, BinaryKernelDNNL);

#endif

}  // namespace Shadow
