#include "eltwise.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(EltwiseCPU, EltwiseKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class EltwiseKernelDNNL : public EltwiseKernel {
 public:
  EltwiseKernelDNNL() {
    default_kernel_ =
        std::make_shared<EltwiseKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::shared_ptr<Blob>& output, Workspace* ws, int operation,
           const VecFloat& coeff) override {
    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        inputs[0]->shape(), idnnl::get_memory_format(inputs[0]->num_axes()));

    auto algorithm = dnnl::algorithm::undef;
    switch (operation) {
      case kProd:
        algorithm = dnnl::algorithm::binary_mul;
        break;
      case kSum:
        if (std::all_of(coeff.begin(), coeff.end(),
                        [](float c) { return std::abs(c - 1.f) < EPS; })) {
          algorithm = dnnl::algorithm::binary_add;
        }
        break;
      case kMax:
        algorithm = dnnl::algorithm::binary_max;
        break;
      case kMin:
        algorithm = dnnl::algorithm::binary_min;
        break;
      default:
        algorithm = dnnl::algorithm::undef;
    }

    if (algorithm != dnnl::algorithm::undef) {
      const auto& binary_desc =
          dnnl::binary::desc(algorithm, in_out_desc, in_out_desc, in_out_desc);

      auto* out_data = output->mutable_data<float>();

      idnnl::binary_forward(ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
                            binary_desc, inputs[0]->data<float>(),
                            inputs[1]->data<float>(), out_data);
      for (int n = 2; n < inputs.size(); ++n) {
        idnnl::binary_forward(ws->Ctx()->dnnl_engine(),
                              ws->Ctx()->dnnl_stream(), binary_desc, out_data,
                              inputs[n]->data<float>(), out_data);
      }
    } else {
      default_kernel_->Run(inputs, output, ws, operation, coeff);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  std::shared_ptr<EltwiseKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(EltwiseCPU, EltwiseKernelDNNL);

#endif

}  // namespace Shadow
