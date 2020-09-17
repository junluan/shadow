#include "psroi_pooling.hpp"

#include "util/util.hpp"

namespace Shadow {

namespace Vision {

template <>
void PSROIPooling<DeviceType::kCPU, float>(
    const float* in_data, const VecInt& in_shape, const float* roi_data,
    int num_rois, int output_dim, int group_size, int pooled_h, int pooled_w,
    float spatial_scale, float* out_data, Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int in_num = in_c * in_h * in_w, out_num = output_dim * pooled_h * pooled_w;
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = static_cast<int>(roi_data[roi_offset]);
    CHECK_GE(roi_batch_id, 0);
    CHECK_LT(roi_batch_id, batch);
    int roi_start_w = Util::round(roi_data[roi_offset + 1] * spatial_scale);
    int roi_start_h = Util::round(roi_data[roi_offset + 2] * spatial_scale);
    int roi_end_w = Util::round(roi_data[roi_offset + 3] * spatial_scale);
    int roi_end_h = Util::round(roi_data[roi_offset + 4] * spatial_scale);
    float roi_height = std::max(roi_end_h - roi_start_h, 1);
    float roi_width = std::max(roi_end_w - roi_start_w, 1);
    float bin_size_h = roi_height / static_cast<float>(pooled_h);
    float bin_size_w = roi_width / static_cast<float>(pooled_w);
    const auto* batch_in_data = in_data + roi_batch_id * in_num;
    auto* batch_out_data = out_data + n * out_num;
    for (int c = 0; c < output_dim; ++c) {
      for (int ph = 0; ph < pooled_h; ++ph) {
        for (int pw = 0; pw < pooled_w; ++pw) {
          auto hstart = static_cast<int>(std::floor(ph * bin_size_h));
          auto wstart = static_cast<int>(std::floor(pw * bin_size_w));
          auto hend = static_cast<int>(std::ceil((ph + 1) * bin_size_h));
          auto wend = static_cast<int>(std::ceil((pw + 1) * bin_size_w));
          hstart = std::min(std::max(hstart + roi_start_h, 0), in_h - 1);
          hend = std::min(std::max(hend + roi_start_h, 0), in_h - 1);
          wstart = std::min(std::max(wstart + roi_start_w, 0), in_w - 1);
          wend = std::min(std::max(wend + roi_start_w, 0), in_w - 1);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int gh = ph * group_size / pooled_h;
          int gw = pw * group_size / pooled_w;
          gh = std::min(std::max(gh, 0), group_size - 1);
          gw = std::min(std::max(gw, 0), group_size - 1);
          int c_in = (c * group_size + gh) * group_size + gw;
          double sum_val = 0;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              sum_val += batch_in_data[(c_in * in_h + h) * in_w + w];
            }
          }
          int bin_area = (hend - hstart) * (wend - wstart);
          int pool_index = (c * pooled_h + ph) * pooled_w + pw;
          batch_out_data[pool_index] =
              is_empty ? 0.f : static_cast<float>(sum_val / bin_area);
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PSROIPoolingCPU,
                           PSROIPoolingKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
