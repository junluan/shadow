#include "deform_conv.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float deform_im2col_bilinear(const float* data, float x, float y,
                                    int width, int height) {
  auto h_low = static_cast<int>(std::floor(y));
  auto w_low = static_cast<int>(std::floor(x));
  int h_high = h_low + 1, w_high = w_low + 1;
  float lh = y - h_low, lw = x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = (h_low >= 0 && w_low >= 0) ? data[h_low * width + w_low] : 0;
  float v2 =
      (h_low >= 0 && w_high <= width - 1) ? data[h_low * width + w_high] : 0;
  float v3 =
      (h_high <= height - 1 && w_low >= 0) ? data[h_high * width + w_low] : 0;
  float v4 = (h_high <= height - 1 && w_high <= width - 1)
                 ? data[h_high * width + w_high]
                 : 0;
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

template <>
void DeformIm2Col2D<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* offset_data,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int deform_group,
    const VecInt& out_shape, float* col_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int channel_per_deform_group = in_c / deform_group;
  for (int c = 0; c < in_c; ++c, in_data += in_h * in_w) {
    int deform_group_idx = c / channel_per_deform_group;
    const auto* offset_data_ptr =
        offset_data +
        2 * deform_group_idx * kernel_size_h * kernel_size_w * out_h * out_w;
    for (int k_s = 0; k_s < kernel_size_h * kernel_size_w;
         ++k_s, offset_data_ptr += out_h * out_w) {
      int kh = k_s / kernel_size_w, kw = k_s % kernel_size_w;
      int h_offset = kh * dilation_h - pad_h;
      for (int h = 0; h < out_h; ++h, h_offset += stride_h) {
        int w_offset = kw * dilation_w - pad_w;
        for (int w = 0; w < out_w;
             ++w, w_offset += stride_w, ++offset_data_ptr) {
          auto h_in = h_offset + offset_data_ptr[0];
          auto w_in = w_offset + offset_data_ptr[out_h * out_w];
          if (h_in > -1 && w_in > -1 && h_in < in_h && w_in < in_w) {
            *col_data++ =
                deform_im2col_bilinear(in_data, w_in, h_in, in_w, in_h);
          } else {
            *col_data++ = 0.f;
          }
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeformConvCPU,
                           DeformConvKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
