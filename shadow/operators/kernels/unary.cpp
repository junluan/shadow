#include "unary.hpp"

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(UnaryCPU, UnaryKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class UnaryKernelDNNL : public UnaryKernel {
 public:
  UnaryKernelDNNL() {
    default_kernel_ = std::make_shared<UnaryKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation) override {
    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));

    dnnl::algorithm algorithm;
    float alpha = 0.f, beta = 0.f;
    switch (operation) {
      case kAbs:
        algorithm = dnnl::algorithm::eltwise_abs;
        break;
      case kSquare:
        algorithm = dnnl::algorithm::eltwise_square;
        break;
      case kSqrt:
        algorithm = dnnl::algorithm::eltwise_sqrt;
        break;
      case kLog:
        algorithm = dnnl::algorithm::eltwise_log;
        break;
      case kExp:
        algorithm = dnnl::algorithm::eltwise_exp;
        break;
      case kNeg:
        algorithm = dnnl::algorithm::eltwise_linear;
        alpha = -1.f;
        break;
      case kReciprocal:
        algorithm = dnnl::algorithm::eltwise_pow;
        alpha = 1.f, beta = -1.f;
        break;
      default:
        algorithm = dnnl::algorithm::undef;
    }

    if (algorithm != dnnl::algorithm::undef) {
      idnnl::common_forward<dnnl::eltwise_forward>(
          ws->Ctx()->dnnl_handle(),
          dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                      algorithm, in_out_desc, alpha, beta),
          input->data<float>(), output->mutable_data<float>());
    } else {
      default_kernel_->Run(input, output, ws, operation);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  std::shared_ptr<UnaryKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(UnaryCPU, UnaryKernelDNNL);

#endif

}  // namespace Shadow
