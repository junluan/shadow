#include "roi_align.hpp"

namespace Shadow {

namespace Vision {

template <>
void ROIAlign<DeviceType::kCPU, float>(const float* in_data,
                                       const VecInt& in_shape,
                                       const float* roi_data, int num_rois,
                                       int pooled_h, int pooled_w,
                                       float spatial_scale, float* out_data,
                                       Context* context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = static_cast<int>(roi_data[roi_offset]);
    CHECK_GE(roi_batch_id, 0);
    CHECK_LT(roi_batch_id, batch);
    float roi_start_w = roi_data[roi_offset + 1] * spatial_scale;
    float roi_start_h = roi_data[roi_offset + 2] * spatial_scale;
    float roi_end_w = roi_data[roi_offset + 3] * spatial_scale;
    float roi_end_h = roi_data[roi_offset + 4] * spatial_scale;
    float roi_height = roi_end_h - roi_start_h;
    float roi_width = roi_end_w - roi_start_w;
    float bin_size_h = roi_height / static_cast<float>(pooled_h - 1);
    float bin_size_w = roi_width / static_cast<float>(pooled_w - 1);
    for (int c = 0; c < in_c; ++c) {
      for (int ph = 0; ph < pooled_h; ++ph) {
        float src_h_f = roi_start_h + ph * bin_size_h;
        int src_h = static_cast<int>(src_h_f);
        float sh = src_h_f - src_h;
        int src_h_off = (roi_batch_id * in_c + c) * in_h + src_h;
        for (int pw = 0; pw < pooled_w; ++pw) {
          float src_w_f = roi_start_w + pw * bin_size_w;
          int src_w = static_cast<int>(src_w_f);
          float sw = src_w_f - src_w;
          int src_index_0 = src_h_off * in_w + src_w;
          int src_index_1 = (src_h_off + 1) * in_w + src_w;
          int src_index_2 = src_h_off * in_w + src_w + 1;
          int src_index_3 = (src_h_off + 1) * in_w + src_w + 1;
          *out_data++ = (1 - sh) * (1 - sw) * in_data[src_index_0] +
                        sh * (1 - sw) * in_data[src_index_1] +
                        (1 - sh) * sw * in_data[src_index_2] +
                        sh * sw * in_data[src_index_3];
        }
      }
    }
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ROIAlignCPU,
                           ROIAlignKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
