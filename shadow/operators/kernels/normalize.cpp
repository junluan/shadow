#include "normalize.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

template <>
void Normalize<DeviceType::kCPU, float>(const float* in_data, int outer_num,
                                        int dim, int inner_num, float* val_data,
                                        float p, float eps, float* out_data,
                                        Context* context) {
  int val_count = outer_num * inner_num, count = val_count * dim;

  for (int i = 0; i < val_count; ++i) {
    int n = i / inner_num, s = i % inner_num;
    const auto* in_data_offset = in_data + n * dim * inner_num + s;
    double val = 0;
    if (p == 1) {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        val += std::abs(*in_data_offset);
      }
      val_data[i] = static_cast<float>(val);
    } else if (p == 2) {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        auto abs_data = std::abs(*in_data_offset);
        val += abs_data * abs_data;
      }
      val_data[i] = std::sqrt(static_cast<float>(val));
    } else {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        val += std::pow(std::abs(*in_data_offset), p);
      }
      val_data[i] = std::pow(static_cast<float>(val), 1.f / p);
    }
  }

  for (int i = 0; i < count; ++i) {
    int n = i / dim / inner_num, s = i % inner_num;
    out_data[i] = in_data[i] / std::max(val_data[n * inner_num + s], eps);
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(NormalizeCPU,
                           NormalizeKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
