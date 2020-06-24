#include "pad.hpp"

namespace Shadow {

namespace Vision {

template <>
void Pad<DeviceType::kCPU, float>(const float* in_data, const VecInt& in_shape,
                                  const VecInt& paddings,
                                  const VecInt& out_shape, float* out_data,
                                  Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < channel; ++c) {
      for (int h = 0; h < in_h; ++h) {
        if (h + paddings[0] < 0 || h >= in_h + paddings[1]) continue;
        int copy_w = in_w + std::min(paddings[2], 0) + std::min(paddings[3], 0);
        int in_offset = ((b * channel + c) * in_h + h) * in_w;
        int out_offset = ((b * channel + c) * out_h + h + paddings[0]) * out_w;
        if (paddings[2] < 0) {
          in_offset -= paddings[2];
        } else {
          out_offset += paddings[2];
        }
        memcpy(out_data + out_offset, in_data + in_offset,
               copy_w * sizeof(float));
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PadCPU, PadKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
