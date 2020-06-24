#include "reorg.hpp"

namespace Shadow {

namespace Vision {

template <>
void Reorg<DeviceType::kCPU, float>(const float* in_data,
                                    const VecInt& in_shape, int stride,
                                    float* out_data, Context* context) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c / (stride * stride);
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < in_h; ++h) {
        for (int w = 0; w < in_w; ++w) {
          int c2 = c % out_c;
          int offset = c / out_c;
          int h2 = h * stride + offset / stride;
          int w2 = w * stride + offset % stride;
          int in_index = ((b * in_c + c) * in_h + h) * in_w + w;
          int out_index =
              ((b * out_c + c2) * in_h * stride + h2) * in_w * stride + w2;
          out_data[in_index] = in_data[out_index];
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReorgCPU, ReorgKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
