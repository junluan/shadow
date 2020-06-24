#include "scale.hpp"

namespace Shadow {

namespace Vision {

template <>
void ScaleBias<DeviceType::kCPU, float>(const float* in_data, int count,
                                        const float* scale_data,
                                        const float* bias_data, int scale_num,
                                        int inner_num, float* out_data,
                                        Context* context) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_num) % scale_num;
    out_data[i] = in_data[i] * scale_data[index] + bias_data[index];
  }
}

template <>
void Scale<DeviceType::kCPU, float>(const float* in_data, int count,
                                    const float* scale_data, int scale_num,
                                    int inner_num, float* out_data,
                                    Context* context) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_num) % scale_num;
    out_data[i] = in_data[i] * scale_data[index];
  }
}

template <>
void Bias<DeviceType::kCPU, float>(const float* in_data, int count,
                                   const float* bias_data, int scale_num,
                                   int inner_num, float* out_data,
                                   Context* context) {
  for (int i = 0; i < count; ++i) {
    int index = (i / inner_num) % scale_num;
    out_data[i] = in_data[i] + bias_data[index];
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ScaleCPU, ScaleKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
