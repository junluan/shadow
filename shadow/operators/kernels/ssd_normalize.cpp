#include "ssd_normalize.hpp"

namespace Shadow {

namespace Vision {

template <>
void SSDNormalize<DeviceType::kCPU, float>(const float* in_data, int outer_num,
                                           int channels, int inner_num,
                                           float eps, float* val_data,
                                           float* out_data, Context* context) {
  int val_count = outer_num * inner_num, count = val_count * channels;

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const auto* out_data_offset = out_data + n * channels * inner_num + s;
    double sum = 0;
    for (int c = 0; c < channels; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[i] = static_cast<float>(std::sqrt(sum + eps));
  }

  for (int i = 0; i < count; ++i) {
    int n = i / channels / inner_num, s = i % inner_num;
    out_data[i] = in_data[i] / val_data[n * inner_num + s];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SSDNormalizeCPU,
                           SSDNormalizeKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
