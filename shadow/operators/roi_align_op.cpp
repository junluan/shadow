#include "roi_align_op.hpp"

namespace Shadow {

void ROIAlignOp::Forward() {
  CHECK_EQ(bottoms_size(), 2);

  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom_fea, top);

  int in_c = bottom_fea->shape(1), num_rois = bottom_roi->shape(0);
  top->reshape({num_rois, in_c, pooled_h_, pooled_w_});

  Vision::ROIAlign(bottom_fea->data(), bottom_fea->shape(), bottom_roi->data(),
                   num_rois, pooled_h_, pooled_w_, spatial_scale_,
                   top->mutable_data(), op_ws_->Ctx());
}

REGISTER_OPERATOR(ROIAlign, ROIAlignOp);

namespace Vision {

#if !defined(USE_CUDA)
template <typename T>
void ROIAlign(const T *in_data, const VecInt &in_shape, const T *roi_data,
              int num_rois, int pooled_h, int pooled_w, float spatial_scale,
              T *out_data, Context *context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = roi_data[roi_offset];
    float roi_start_w = roi_data[roi_offset + 1] * spatial_scale;
    float roi_start_h = roi_data[roi_offset + 2] * spatial_scale;
    float roi_end_w = roi_data[roi_offset + 3] * spatial_scale;
    float roi_end_h = roi_data[roi_offset + 4] * spatial_scale;
    assert(roi_batch_id >= 0);
    assert(roi_batch_id < batch);
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
          *out_data++ =
              static_cast<T>((1 - sh) * (1 - sw) * in_data[src_index_0] +
                             sh * (1 - sw) * in_data[src_index_1] +
                             (1 - sh) * sw * in_data[src_index_2] +
                             sh * sw * in_data[src_index_3]);
        }
      }
    }
  }
}

template void ROIAlign(const float *, const VecInt &, const float *, int, int,
                       int, float, float *, Context *);
#endif

}  // namespace Vision

}  // namespace Shadow
