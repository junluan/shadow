#include "group_norm.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <>
void SubtractMeanAndSquare<DeviceType::kCPU, float>(
    const float* in_data, const float* mean_data, int count, int inner_num,
    float* out_data, float* square_data, Context* context) {
  for (int i = 0; i < count; ++i) {
    auto val = *in_data++ - mean_data[i / inner_num];
    *out_data++ = val;
    *square_data++ = val * val;
  }
}

template <>
void DivideVariance<DeviceType::kCPU, float>(const float* in_data,
                                             const float* variance_data,
                                             int count, int inner_num,
                                             float eps, float* out_data,
                                             Context* context) {
  for (int i = 0; i < count; ++i) {
    *out_data++ = *in_data++ / std::sqrt(variance_data[i / inner_num] + eps);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GroupNormCPU,
                           GroupNormKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
