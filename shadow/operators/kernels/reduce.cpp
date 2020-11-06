#include "reduce.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float Reduce(const float* data, const int* list, int num_list,
                    int offset, int operation) {
  switch (operation) {
    case kProd: {
      double val = 1;
      for (int i = 0; i < num_list; ++i) {
        val *= data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kSum: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val);
    }
    case kMax: {
      float val = std::numeric_limits<float>::lowest();
      for (int i = 0; i < num_list; ++i) {
        val = std::max(val, data[list[i] + offset]);
      }
      return val;
    }
    case kMin: {
      float val = std::numeric_limits<float>::max();
      for (int i = 0; i < num_list; ++i) {
        val = std::min(val, data[list[i] + offset]);
      }
      return val;
    }
    case kAvg: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += data[list[i] + offset];
      }
      return static_cast<float>(val / num_list);
    }
    case kLpNorm1: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        val += std::abs(data[list[i] + offset]);
      }
      return static_cast<float>(val);
    }
    case kLpNorm2: {
      double val = 0;
      for (int i = 0; i < num_list; ++i) {
        auto abs_data = std::abs(data[list[i] + offset]);
        val += abs_data * abs_data;
      }
      return std::sqrt(static_cast<float>(val));
    }
    default:
      return 0;
  }
}

template <>
void Reduce<DeviceType::kCPU, float>(const float* in_data, const int* list_data,
                                     const int* offset_data, int num_list,
                                     int operation, int count, float* out_data,
                                     Context* context) {
  for (int i = 0; i < count; ++i) {
    out_data[i] =
        Reduce(in_data, list_data, num_list, offset_data[i], operation);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReduceCPU, ReduceKernelDefault<DeviceType::kCPU>);

#if defined(USE_DNNL)

class ReduceKernelDNNL : public ReduceKernel {
 public:
  ReduceKernelDNNL() {
    default_kernel_ = std::make_shared<ReduceKernelDefault<DeviceType::kCPU>>();
  }

  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int operation, const VecInt& axes) override {
    const auto& src_desc = idnnl::create_memory_desc<float>(
        input->shape(), idnnl::get_memory_format(input->num_axes()));
    const auto& dst_desc = idnnl::create_memory_desc<float>(
        output->shape(), idnnl::get_memory_format(output->num_axes()));

    dnnl::algorithm algorithm;
    float p = 0.f, eps = 0.f;
    switch (operation) {
      case kProd:
        algorithm = dnnl::algorithm::reduction_mul;
        break;
      case kSum:
        algorithm = dnnl::algorithm::reduction_sum;
        break;
      case kMax:
        algorithm = dnnl::algorithm::reduction_max;
        break;
      case kMin:
        algorithm = dnnl::algorithm::reduction_min;
        break;
      case kAvg:
        algorithm = dnnl::algorithm::reduction_mean;
        break;
      case kLpNorm1:
        algorithm = dnnl::algorithm::reduction_norm_lp_sum;
        p = 1.f;
        break;
      case kLpNorm2:
        algorithm = dnnl::algorithm::reduction_norm_lp_sum;
        p = 2.f;
        break;
      default:
        algorithm = dnnl::algorithm::undef;
    }

    if (algorithm != dnnl::algorithm::undef) {
      idnnl::common_forward<dnnl::reduction>(
          ws->Ctx()->dnnl_engine(), ws->Ctx()->dnnl_stream(),
          dnnl::reduction::desc(algorithm, src_desc, dst_desc, p, eps),
          input->data<float>(), output->mutable_data<float>());
    } else {
      default_kernel_->Run(input, output, ws, operation, axes);
    }
  }

  DeviceType device_type() const override { return DeviceType::kCPU; }

  std::string kernel_type() const override { return "DNNL"; }

 private:
  std::shared_ptr<ReduceKernelDefault<DeviceType::kCPU>> default_kernel_ =
      nullptr;
};

REGISTER_OP_KERNEL_DNNL(ReduceCPU, ReduceKernelDNNL);

#endif

}  // namespace Shadow
