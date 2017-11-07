#include "deformable_psroi_pooling_op.hpp"

namespace Shadow {

void DeformablePSROIPoolingOp::Reshape() {
  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom_fea, top);

  int num_rois = bottom_roi->shape(0);

  top->reshape({num_rois, output_dim_, pooled_size_, pooled_size_});

  VecString str;
  for (int i = 0; i < bottoms_size(); ++i) {
    const auto *bottom = bottoms<float>(i);
    str.push_back(
        Util::format_vector(bottom->shape(), ",", bottom->name() + "(", ")"));
  }
  DLOG(INFO) << op_name_ << "(" << op_type_
             << "): " << Util::format_vector(str, " + ") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void DeformablePSROIPoolingOp::Forward() {
  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  int num_rois = bottom_roi->shape(0);

  if (no_trans_) {
    Vision::DeformablePSROIPooling(
        bottom_fea->data(), bottom_fea->shape(), bottom_roi->data(),
        static_cast<decltype(bottom_roi->data())>(nullptr), VecInt{}, num_rois,
        output_dim_, group_size_, pooled_size_, part_size_, sample_per_part_,
        spatial_scale_, trans_std_, no_trans_, top->mutable_data());
  } else {
    const auto *bottom_trans = bottoms<float>(2);
    Vision::DeformablePSROIPooling(
        bottom_fea->data(), bottom_fea->shape(), bottom_roi->data(),
        bottom_trans->data(), bottom_trans->shape(), num_rois, output_dim_,
        group_size_, pooled_size_, part_size_, sample_per_part_, spatial_scale_,
        trans_std_, no_trans_, top->mutable_data());
  }
}

REGISTER_OPERATOR(DeformablePSROIPooling, DeformablePSROIPoolingOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
inline float bilinear_interp(const float *data, float x, float y, int width,
                             int height) {
  auto x1 = static_cast<int>(std::floor(x));
  auto x2 = static_cast<int>(std::ceil(x));
  auto y1 = static_cast<int>(std::floor(y));
  auto y2 = static_cast<int>(std::ceil(y));
  float dist_x = x - x1;
  float dist_y = y - y1;
  float value11 = data[y1 * width + x1];
  float value12 = data[y2 * width + x1];
  float value21 = data[y1 * width + x2];
  float value22 = data[y2 * width + x2];
  float value = (1 - dist_x) * (1 - dist_y) * value11 +
                (1 - dist_x) * dist_y * value12 +
                dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
  return value;
}

template <typename T>
void DeformablePSROIPooling(const T *in_data, const VecInt &in_shape,
                            const T *roi_data, const T *trans_data,
                            const VecInt &trans_shape, int num_rois,
                            int output_dim, int group_size, int pooled_size,
                            int part_size, int sample_per_part,
                            float spatial_scale, float trans_std, bool no_trans,
                            T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int in_num = in_c * in_h * in_w,
      out_num = output_dim * pooled_size * pooled_size;
  int num_classes = no_trans ? 1 : trans_shape[1] / 2;
  int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = roi_data[roi_offset];
    float roi_start_w =
        Util::round(roi_data[roi_offset + 1]) * spatial_scale - 0.5f;
    float roi_start_h =
        Util::round(roi_data[roi_offset + 2]) * spatial_scale - 0.5f;
    float roi_end_w =
        (Util::round(roi_data[roi_offset + 3]) + 1) * spatial_scale - 0.5f;
    float roi_end_h =
        (Util::round(roi_data[roi_offset + 4]) + 1) * spatial_scale - 0.5f;
    CHECK_GE(roi_batch_id, 0);
    CHECK_LT(roi_batch_id, batch);
    float roi_height = std::max(roi_end_h - roi_start_h, 0.1f);
    float roi_width = std::max(roi_end_w - roi_start_w, 0.1f);
    float bin_size_h = roi_height / static_cast<float>(pooled_size);
    float bin_size_w = roi_width / static_cast<float>(pooled_size);
    float sub_bin_size_h = bin_size_h / static_cast<float>(sample_per_part);
    float sub_bin_size_w = bin_size_w / static_cast<float>(sample_per_part);
    const T *batch_in_data = in_data + roi_batch_id * in_num;
    T *batch_out_data = out_data + n * out_num;
    for (int c = 0; c < output_dim; ++c) {
      for (int ph = 0; ph < pooled_size; ++ph) {
        for (int pw = 0; pw < pooled_size; ++pw) {
          auto part_h = static_cast<int>(
              std::floor(static_cast<float>(ph) / pooled_size * part_size));
          auto part_w = static_cast<int>(
              std::floor(static_cast<float>(pw) / pooled_size * part_size));
          int class_id = c / channels_each_class;
          T trans_x =
              no_trans
                  ? static_cast<T>(0)
                  : trans_data[(((n * num_classes + class_id) * 2) * part_size +
                                part_h) *
                                   part_size +
                               part_w] *
                        trans_std;
          T trans_y = no_trans
                          ? static_cast<T>(0)
                          : trans_data[(((n * num_classes + class_id) * 2 + 1) *
                                            part_size +
                                        part_h) *
                                           part_size +
                                       part_w] *
                                trans_std;
          float hstart = ph * bin_size_h + roi_start_h + trans_y * roi_height;
          float wstart = pw * bin_size_w + roi_start_w + trans_x * roi_width;
          int gh = ph * group_size / pooled_size;
          int gw = pw * group_size / pooled_size;
          gh = std::min(std::max(gh, 0), group_size - 1);
          gw = std::min(std::max(gw, 0), group_size - 1);
          int count = 0;
          int c_in = (c * group_size + gh) * group_size + gw;
          auto sum_val = static_cast<T>(0);
          for (int ih = 0; ih < sample_per_part; ++ih) {
            for (int iw = 0; iw < sample_per_part; ++iw) {
              float w = wstart + iw * sub_bin_size_w;
              float h = hstart + ih * sub_bin_size_h;
              if (w < -0.5f || w > in_w - 0.5f || h < -0.5f ||
                  h > in_h - 0.5f) {
                continue;
              }
              w = std::min(std::max(w, 0.f), in_w - 1.f);
              h = std::min(std::max(h, 0.f), in_h - 1.f);
              sum_val += bilinear_interp(batch_in_data + c_in * in_h * in_w, w,
                                         h, in_w, in_h);
              count++;
            }
          }
          int pool_index = (c * pooled_size + ph) * pooled_size + pw;
          batch_out_data[pool_index] = count == 0 ? T(0) : sum_val / count;
        }
      }
    }
  }
}

template void DeformablePSROIPooling(
    const float *in_data, const VecInt &in_shape, const float *roi_data,
    const float *trans_data, const VecInt &trans_shape, int num_rois,
    int output_dim, int group_size, int pooled_size, int part_size,
    int sample_per_part, float spatial_scale, float trans_std, bool no_trans,
    float *out_data);

#elif defined(USE_CL)
#endif

}  // namespace Vision

}  // namespace Shadow
