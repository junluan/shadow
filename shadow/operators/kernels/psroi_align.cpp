#include "psroi_align.hpp"

#include <cmath>

namespace Shadow {

namespace Vision {

inline float psroi_align_bilinear(const float* data, float x, float y,
                                  int width, int height) {
  if (x < -1.0 || x > width || y < -1.0 || y > height) {
    return 0;
  }

  x = std::max(x, 0.f), y = std::max(y, 0.f);

  auto h_low = static_cast<int>(std::floor(y));
  auto w_low = static_cast<int>(std::floor(x));
  int h_high = h_low + 1, w_high = w_low + 1;

  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    y = static_cast<float>(h_low);
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    x = static_cast<float>(w_low);
  }

  float lh = y - h_low, lw = x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = data[h_low * width + w_low];
  float v2 = data[h_low * width + w_high];
  float v3 = data[h_high * width + w_low];
  float v4 = data[h_high * width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <>
void PSROIAlign<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* roi_data,
    int num_rois, int out_c, int pooled_h, int pooled_w, float spatial_scale,
    int sampling_ratio, float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int in_num = in_c * in_h * in_w;
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = static_cast<int>(roi_data[roi_offset]);
    CHECK_GE(roi_batch_id, 0);
    CHECK_LT(roi_batch_id, batch);
    float roi_start_w = roi_data[roi_offset + 1] * spatial_scale - 0.5f;
    float roi_start_h = roi_data[roi_offset + 2] * spatial_scale - 0.5f;
    float roi_end_w = roi_data[roi_offset + 3] * spatial_scale - 0.5f;
    float roi_end_h = roi_data[roi_offset + 4] * spatial_scale - 0.5f;
    float roi_height = roi_end_h - roi_start_h;
    float roi_width = roi_end_w - roi_start_w;
    float bin_size_h = roi_height / pooled_h;
    float bin_size_w = roi_width / pooled_w;
    int roi_bin_grid_h = sampling_ratio > 0
                             ? sampling_ratio
                             : static_cast<int>(std::ceil(bin_size_h));
    int roi_bin_grid_w = sampling_ratio > 0
                             ? sampling_ratio
                             : static_cast<int>(std::ceil(bin_size_w));
    float grid_size = std::max(roi_bin_grid_h * roi_bin_grid_w, 1);
    const auto* in_data_ptr = in_data + roi_batch_id * in_num;
    for (int c = 0; c < out_c; ++c) {
      for (int ph = 0; ph < pooled_h; ++ph) {
        for (int pw = 0; pw < pooled_w; ++pw, in_data_ptr += in_h * in_w) {
          double sum_val = 0;
          for (int h = 0; h < roi_bin_grid_h; ++h) {
            float y = roi_start_h + ph * bin_size_h +
                      (h + 0.5f) * bin_size_h / roi_bin_grid_h;
            for (int w = 0; w < roi_bin_grid_w; ++w) {
              float x = roi_start_w + pw * bin_size_w +
                        (w + 0.5f) * bin_size_w / roi_bin_grid_w;
              sum_val += psroi_align_bilinear(in_data_ptr, x, y, in_w, in_h);
            }
          }
          *out_data++ = static_cast<float>(sum_val / grid_size);
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PSROIAlignCPU,
                           PSROIAlignKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
