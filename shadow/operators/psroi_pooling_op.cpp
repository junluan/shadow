#include "psroi_pooling_op.hpp"

namespace Shadow {

void PSROIPoolingOp::Forward() {
  CHECK_EQ(bottoms_size(), 2);

  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom_fea, top);

  int num_rois = bottom_roi->shape(0);

  top->reshape({num_rois, output_dim_, pooled_h_, pooled_w_});

  Vision::PSROIPooling(bottom_fea->data(), bottom_fea->shape(),
                       bottom_roi->data(), num_rois, output_dim_, group_size_,
                       pooled_h_, pooled_w_, spatial_scale_,
                       top->mutable_data());

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(PSROIPooling, PSROIPoolingOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void PSROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                  int num_rois, int output_dim, int group_size, int pooled_h,
                  int pooled_w, float spatial_scale, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int in_num = in_c * in_h * in_w, out_num = output_dim * pooled_h * pooled_w;
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = roi_data[roi_offset];
    float roi_start_w = Util::round(roi_data[roi_offset + 1]) * spatial_scale;
    float roi_start_h = Util::round(roi_data[roi_offset + 2]) * spatial_scale;
    float roi_end_w =
        (Util::round(roi_data[roi_offset + 3]) + 1) * spatial_scale;
    float roi_end_h =
        (Util::round(roi_data[roi_offset + 4]) + 1) * spatial_scale;
    CHECK_GE(roi_batch_id, 0);
    CHECK_LT(roi_batch_id, batch);
    float roi_height = std::max(roi_end_h - roi_start_h, 0.1f);
    float roi_width = std::max(roi_end_w - roi_start_w, 0.1f);
    float bin_size_h = roi_height / static_cast<float>(pooled_h);
    float bin_size_w = roi_width / static_cast<float>(pooled_w);
    const T *batch_in_data = in_data + roi_batch_id * in_num;
    T *batch_out_data = out_data + n * out_num;
    for (int c = 0; c < output_dim; ++c) {
      for (int ph = 0; ph < pooled_h; ++ph) {
        for (int pw = 0; pw < pooled_w; ++pw) {
          auto hstart =
              static_cast<int>(std::floor(ph * bin_size_h + roi_start_h));
          auto wstart =
              static_cast<int>(std::floor(pw * bin_size_w + roi_start_w));
          auto hend =
              static_cast<int>(std::ceil((ph + 1) * bin_size_h + roi_start_h));
          auto wend =
              static_cast<int>(std::ceil((pw + 1) * bin_size_w) + roi_start_w);
          hstart = std::min(std::max(hstart, 0), in_h);
          hend = std::min(std::max(hend, 0), in_h);
          wstart = std::min(std::max(wstart, 0), in_w);
          wend = std::min(std::max(wend, 0), in_w);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          int gh = ph * group_size / pooled_h;
          int gw = pw * group_size / pooled_w;
          gh = std::min(std::max(gh, 0), group_size - 1);
          gw = std::min(std::max(gw, 0), group_size - 1);
          int c_in = (c * group_size + gh) * group_size + gw;
          auto sum_val = static_cast<T>(0);
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              sum_val += batch_in_data[(c_in * in_h + h) * in_w + w];
            }
          }
          float bin_area = (hend - hstart) * (wend - wstart);
          int pool_index = (c * pooled_h + ph) * pooled_w + pw;
          batch_out_data[pool_index] = is_empty ? T(0) : sum_val / bin_area;
        }
      }
    }
  }
}

template void PSROIPooling(const float *in_data, const VecInt &in_shape,
                           const float *roi_data, int num_rois, int output_dim,
                           int group_size, int pooled_h, int pooled_w,
                           float spatial_scale, float *out_data);

#elif defined(USE_CL)
#endif

}  // namespace Vision

}  // namespace Shadow
