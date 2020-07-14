#include "reorg.hpp"

namespace Shadow {

namespace Vision {

inline void ReorgDarknet(const float* in_data, int batch, int in_c, int in_h,
                         int in_w, int stride, float* out_data,
                         Context* context) {
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

inline void ReorgNatural(const float* in_data, int batch, int in_c, int in_h,
                         int in_w, int stride, float* out_data,
                         Context* context) {
  int out_c = in_c * stride * stride;
  int out_h = in_h / stride, out_w = in_w / stride;
  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < out_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int offset = c / stride;
          int c2 = offset / stride;
          int h2 = h * stride + offset % stride;
          int w2 = w * stride + c % stride;
          int in_index = ((b * in_c + c2) * in_h + h2) * in_w + w2;
          int out_index = ((b * out_c + c) * out_h + h) * out_w + w;
          out_data[out_index] = in_data[in_index];
        }
      }
    }
  }
}

template <>
void Reorg<DeviceType::kCPU, float>(const float* in_data,
                                    const VecInt& in_shape, int type,
                                    int stride, float* out_data,
                                    Context* context) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  if (type == kDarknet) {
    ReorgDarknet(in_data, batch, in_c, in_h, in_w, stride, out_data, context);
  } else if (type == kNatural) {
    ReorgNatural(in_data, batch, in_c, in_h, in_w, stride, out_data, context);
  } else {
    LOG(FATAL) << "Unsupported reorg type: " << type;
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReorgCPU, ReorgKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
