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
      return std::log(1 + std::exp(x));
    case kTanh: {
      auto exp_2x = std::exp(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    case kRelu6: {
      x = x > 0 ? x : 0;
      return x < 6 ? x : 6;
    }
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

}  // namespace Shadow
