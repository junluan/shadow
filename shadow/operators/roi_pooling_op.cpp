#include "roi_pooling_op.hpp"

namespace Shadow {

void ROIPoolingOp::Reshape() {
  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom_fea, top);

  int in_c = bottom_fea->shape(1), num_rois = bottom_roi->shape(0);

  top->reshape({num_rois, in_c, pooled_h_, pooled_w_});

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

void ROIPoolingOp::Forward() {
  const auto *bottom_fea = bottoms<float>(0);
  const auto *bottom_roi = bottoms<float>(1);
  auto *top = mutable_tops<float>(0);

  int num_rois = bottom_roi->shape(0);

  Vision::ROIPooling(bottom_fea->data(), bottom_fea->shape(),
                     bottom_roi->data(), num_rois, pooled_h_, pooled_w_,
                     spatial_scale_, top->mutable_data());
}

REGISTER_OPERATOR(ROIPooling, ROIPoolingOp);

namespace Vision {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void ROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                int num_rois, int pooled_h, int pooled_w, float spatial_scale,
                T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int in_num = in_c * in_h * in_w, out_num = in_c * pooled_h * pooled_w;
  for (int n = 0; n < num_rois; ++n) {
    int roi_offset = 5 * n;
    int roi_batch_id = roi_data[roi_offset];
    int roi_start_w = Util::round(roi_data[roi_offset + 1] * spatial_scale);
    int roi_start_h = Util::round(roi_data[roi_offset + 2] * spatial_scale);
    int roi_end_w = Util::round(roi_data[roi_offset + 3] * spatial_scale);
    int roi_end_h = Util::round(roi_data[roi_offset + 4] * spatial_scale);
    assert(roi_batch_id >= 0);
    assert(roi_batch_id < batch);
    int roi_height = std::max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = std::max(roi_end_w - roi_start_w + 1, 1);
    float bin_size_h = roi_height / static_cast<float>(pooled_h);
    float bin_size_w = roi_width / static_cast<float>(pooled_w);
    const T *batch_in_data = in_data + roi_batch_id * in_num;
    T *batch_out_data = out_data + n * out_num;
    for (int c = 0; c < in_c; ++c) {
      for (int ph = 0; ph < pooled_h; ++ph) {
        for (int pw = 0; pw < pooled_w; ++pw) {
          auto hstart = static_cast<int>(std::floor(ph * bin_size_h));
          auto wstart = static_cast<int>(std::floor(pw * bin_size_w));
          auto hend = static_cast<int>(std::ceil((ph + 1) * bin_size_h));
          auto wend = static_cast<int>(std::ceil((pw + 1) * bin_size_w));
          hstart = std::min(std::max(hstart + roi_start_h, 0), in_h);
          hend = std::min(std::max(hend + roi_start_h, 0), in_h);
          wstart = std::min(std::max(wstart + roi_start_w, 0), in_w);
          wend = std::min(std::max(wend + roi_start_w, 0), in_w);
          bool is_empty = (hend <= hstart) || (wend <= wstart);
          T max_val = is_empty
                          ? T(0)
                          : batch_in_data[(c * in_h + hstart) * in_w + wstart];
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              max_val =
                  std::max(max_val, batch_in_data[(c * in_h + h) * in_w + w]);
            }
          }
          int pool_index = (c * pooled_h + ph) * pooled_w + pw;
          batch_out_data[pool_index] = max_val;
        }
      }
    }
  }
}

template void ROIPooling(const float *in_data, const VecInt &in_shape,
                         const float *roi_data, int num_rois, int pooled_h,
                         int pooled_w, float spatial_scale, float *out_data);
#elif defined(USE_CL)
template <typename T>
void ROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                int num_rois, int pooled_h, int pooled_w, float spatial_scale,
                T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * in_c * pooled_h * pooled_w;

  size_t global = count;
  auto *kernel = Kernel::cl_kernels_["ROIPooling"];
  kernel->SetArguments(*in_data, count, *roi_data, in_c, in_h, in_w, pooled_h,
                       pooled_w, spatial_scale, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template void ROIPooling(const BufferF *in_data, const VecInt &in_shape,
                         const BufferF *roi_data, int num_rois, int pooled_h,
                         int pooled_w, float spatial_scale, BufferF *out_data);
#endif
}

}  // namespace Shadow
