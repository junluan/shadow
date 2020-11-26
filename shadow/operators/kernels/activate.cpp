#include "activate.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float Activate(float x, int type, float slope) {
  switch (type) {
    case kRelu:
      return x > 0 ? x : 0;
    case kLeaky:
      return x > 0 ? x : slope * x;
    case kSigmoid:
      return 1 / (1 + std::exp(-x));
    case kSoftPlus:
      return std::log1p(std::exp(x));
    case kTanh:
      return std::tanh(x);
    case kRelu6:
      return x < 0 ? 0 : (x > 6 ? 6 : x);
    case kHardSwish:
      return x < -3 ? 0 : (x > 3 ? x : (x * (x + 3) / 6.f));
    case kGelu:
      return 0.5f * x *
             (1 + std::tanh(0.797885f * (x + 0.044715f * std::pow(x, 3.f))));
    default:
      return x;
  }
}

template <>
void Activate<DeviceType::kCPU, float>(const float* in_data, float* out_data,
                                       int count, int type, float slope,
                                       Context* context) {
  for (int i = 0; i < count; ++i) {
    out_data[i] = Activate(in_data[i], type, slope);
  }
}

template <>
void PRelu<DeviceType::kCPU, float>(const float* in_data, float* out_data,
                                    const VecInt& in_shape, bool channel_shared,
                                    const float* slope_data, Context* context) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  for (int i = 0; i < count; ++i) {
    int c = (i / dim) % channels / div_factor;
    out_data[i] = in_data[i] > 0 ? in_data[i] : in_data[i] * slope_data[c];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ActivateCPU,
                           ActivateKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class ActivateKernelDNNL : public ActivateKernel {
 public:
  ActivateKernelDNNL() {
    default_kernel_ =
        std::make_shared<ActivateKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& slope, std::shared_ptr<Blob>& output,
           Workspace* ws, int activate_type, float slope_val) override {
    const auto& in_out_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));

    auto algorithm = dnnl::algorithm::undef;
    float alpha = 0.f;
    switch (activate_type) {
      case kPRelu:
        if (slope->count() == 1) {
          algorithm = dnnl::algorithm::eltwise_relu;
          alpha = slope->data<float>()[0];
        }
        break;
      case kRelu:
        algorithm = dnnl::algorithm::eltwise_relu;
        break;
      case kLeaky:
        algorithm = dnnl::algorithm::eltwise_relu;
        alpha = slope_val;
        break;
      case kSigmoid:
        algorithm = dnnl::algorithm::eltwise_logistic;
        break;
      case kSoftPlus:
        algorithm = dnnl::algorithm::eltwise_soft_relu;
        break;
      case kTanh:
        algorithm = dnnl::algorithm::eltwise_tanh;
        break;
      case kRelu6:
        algorithm = dnnl::algorithm::eltwise_bounded_relu;
        alpha = 6.f;
        break;
      default:
        algorithm = dnnl::algorithm::undef;
    }

    if (algorithm != dnnl::algorithm::undef) {
      idnnl::common_forward<dnnl::eltwise_forward>(
          ws->Ctx()->dnnl_handle(),
          dnnl::eltwise_forward::desc(dnnl::prop_kind::forward_inference,
                                      algorithm, in_out_desc, alpha),
          input->data<float>(), output->mutable_data<float>());
    } else {
      default_kernel_->Run(input, slope, output, ws, activate_type, slope_val);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  std::shared_ptr<ActivateKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(ActivateCPU, ActivateKernelDNNL);

#endif

}  // namespace Shadow
