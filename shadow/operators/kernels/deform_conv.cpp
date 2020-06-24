#include "deform_conv.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float deform_im2col_bilinear(const float* bottom_data, int data_width,
                                    int height, int width, float h, float w) {
  auto h_low = static_cast<int>(std::floor(h));
  auto w_low = static_cast<int>(std::floor(w));
  int h_high = h_low + 1, w_high = w_low + 1;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = static_cast<float>(h_low);
  }
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = static_cast<float>(w_low);
  }
  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = bottom_data[h_low * data_width + w_low];
  float v2 = bottom_data[h_low * data_width + w_high];
  float v3 = bottom_data[h_high * data_width + w_low];
  float v4 = bottom_data[h_high * data_width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <>
void DeformIm2Col<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* offset_data,
    int offset, int deform_group, int kernel_size, int stride, int pad,
    int dilation, int zero_point, const VecInt& out_shape, float* out_data,
    Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int channel_per_deform_group = in_c / deform_group;
  for (int c_im = 0; c_im < in_c; ++c_im) {
    for (int h_col = 0; h_col < out_h; ++h_col) {
      for (int w_col = 0; w_col < out_w; ++w_col) {
        int c_col = c_im * kernel_size * kernel_size;
        int deform_group_index = c_im / channel_per_deform_group;
        int h_in = h_col * stride - pad;
        int w_in = w_col * stride - pad;
        auto* data_col_ptr = out_data + (c_col * out_h + h_col) * out_w + w_col;
        const auto* data_im_ptr =
            in_data + offset + (c_im * in_h + h_in) * in_w + w_in;
        const auto* data_offset_ptr =
            offset_data +
            deform_group_index * 2 * kernel_size * kernel_size * out_h * out_w;
        for (int i = 0; i < kernel_size; ++i) {
          for (int j = 0; j < kernel_size; ++j) {
            int data_offset_h_ptr =
                ((2 * (i * kernel_size + j)) * out_h + h_col) * out_w + w_col;
            int data_offset_w_ptr =
                ((2 * (i * kernel_size + j) + 1) * out_h + h_col) * out_w +
                w_col;
            auto offset_h = data_offset_ptr[data_offset_h_ptr];
            auto offset_w = data_offset_ptr[data_offset_w_ptr];
            auto val = static_cast<float>(zero_point);
            auto h_im = h_in + i * dilation + offset_h;
            auto w_im = w_in + j * dilation + offset_w;
            if (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w) {
              auto map_h = i * dilation + offset_h;
              auto map_w = j * dilation + offset_w;
              int cur_height = in_h - h_in;
              int cur_width = in_w - w_in;
              val = deform_im2col_bilinear(data_im_ptr, in_w, cur_height,
                                           cur_width, map_h, map_w);
            }
            *data_col_ptr = val;
            data_col_ptr += out_h * out_w;
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
