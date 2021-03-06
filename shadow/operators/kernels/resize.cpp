#include "resize.hpp"

namespace Shadow {

namespace Vision {

template <>
void ResizeNearest2D<DeviceType::kCPU, float>(const float* in_data,
                                              const VecInt& in_shape,
                                              const VecInt& out_shape,
                                              float* out_data,
                                              Context* context) {
  int outer_num = in_shape[0] * in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  float fh = static_cast<float>(in_h) / out_h;
  float fw = static_cast<float>(in_w) / out_w;

  for (int n = 0; n < outer_num; ++n, in_data += in_h * in_w) {
    for (int h = 0; h < out_h; ++h) {
      int src_h = static_cast<int>(h * fh);
      for (int w = 0; w < out_w; ++w) {
        int src_w = static_cast<int>(w * fw);
        int src_index = src_h * in_w + src_w;
        *out_data++ = in_data[src_index];
      }
    }
  }
}

template <>
void ResizeBilinear2D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, bool align_corners,
    const VecInt& out_shape, float* out_data, Context* context) {
  int outer_num = in_shape[0] * in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  float fh = align_corners ? static_cast<float>(in_h - 1) / (out_h - 1)
                           : static_cast<float>(in_h) / out_h;
  float fw = align_corners ? static_cast<float>(in_w - 1) / (out_w - 1)
                           : static_cast<float>(in_w) / out_w;

  for (int n = 0; n < outer_num; ++n, in_data += in_h * in_w) {
    for (int h = 0; h < out_h; ++h) {
      float src_h_f;
      if (align_corners) {
        src_h_f = h * fh;
      } else {
        src_h_f = (h + 0.5f) * fh - 0.5f;
        src_h_f = src_h_f < 0 ? 0 : src_h_f;
      }
      int src_h_l = static_cast<int>(src_h_f), src_h_h = src_h_l + 1;
      if (src_h_l >= in_h - 1) {
        src_h_h = src_h_l = in_h - 1;
      }
      float sh = src_h_f - src_h_l;
      for (int w = 0; w < out_w; ++w) {
        float src_w_f;
        if (align_corners) {
          src_w_f = w * fw;
        } else {
          src_w_f = (w + 0.5f) * fw - 0.5f;
          src_w_f = src_w_f < 0 ? 0 : src_w_f;
        }
        int src_w_l = static_cast<int>(src_w_f), src_w_h = src_w_l + 1;
        if (src_w_l >= in_w - 1) {
          src_w_h = src_w_l = in_w - 1;
        }
        float sw = src_w_f - src_w_l;
        int src_index_0 = src_h_l * in_w + src_w_l;
        int src_index_1 = src_h_h * in_w + src_w_l;
        int src_index_2 = src_h_l * in_w + src_w_h;
        int src_index_3 = src_h_h * in_w + src_w_h;
        *out_data++ = (1 - sh) * (1 - sw) * in_data[src_index_0] +
                      sh * (1 - sw) * in_data[src_index_1] +
                      (1 - sh) * sw * in_data[src_index_2] +
                      sh * sw * in_data[src_index_3];
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ResizeCPU, ResizeKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
