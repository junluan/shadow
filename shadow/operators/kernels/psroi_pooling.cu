#include "psroi_pooling.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPSROIPooling(const float* in_data, int count,
                                   const float* roi_data, int in_c, int in_h,
                                   int in_w, int output_dim, int group_size,
                                   int pooled_h, int pooled_w,
                                   float spatial_scale, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c_out = (globalid / pooled_w / pooled_h) % output_dim;
    int n = globalid / pooled_w / pooled_h / output_dim;

    const auto* offset_bottom_rois = roi_data + n * 5;
    int roi_batch_id = static_cast<int>(offset_bottom_rois[0]);
    auto roi_start_w = roundf(offset_bottom_rois[1]) * spatial_scale;
    auto roi_start_h = roundf(offset_bottom_rois[2]) * spatial_scale;
    auto roi_end_w = (roundf(offset_bottom_rois[3]) + 1) * spatial_scale;
    auto roi_end_h = (roundf(offset_bottom_rois[4]) + 1) * spatial_scale;

    auto roi_height = fmaxf(roi_end_h - roi_start_h, 0.1f);
    auto roi_width = fmaxf(roi_end_w - roi_start_w, 0.1f);
    auto bin_size_h = roi_height / pooled_h;
    auto bin_size_w = roi_width / pooled_w;

    int hstart = static_cast<int>(floorf(ph * bin_size_h + roi_start_h));
    int wstart = static_cast<int>(floorf(pw * bin_size_w + roi_start_w));
    int hend = static_cast<int>(ceilf((ph + 1) * bin_size_h + roi_start_h));
    int wend = static_cast<int>(ceilf((pw + 1) * bin_size_w + roi_start_w));

    hstart = min(max(hstart, 0), in_h);
    hend = min(max(hend, 0), in_h);
    wstart = min(max(wstart, 0), in_w);
    wend = min(max(wend, 0), in_w);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gh = ph * group_size / in_h;
    int gw = pw * group_size / in_w;
    gh = min(max(gh, 0), group_size - 1);
    gw = min(max(gw, 0), group_size - 1);
    int c_in = (c_out * group_size + gh) * group_size + gw;
    const auto* offset_bottom_data =
        in_data + (roi_batch_id * in_c + c_in) * in_h * in_w;

    double sum_val = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        sum_val += offset_bottom_data[h * in_w + w];
      }
    }
    int bin_area = (hend - hstart) * (wend - wstart);
    out_data[globalid] =
        is_empty ? 0.f : static_cast<float>(sum_val / bin_area);
  }
}

template <>
void PSROIPooling<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* roi_data,
    int num_rois, int output_dim, int group_size, int pooled_h, int pooled_w,
    float spatial_scale, float* out_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * output_dim * pooled_h * pooled_w;
  KernelPSROIPooling<<<GetBlocks(count), NumThreads, 0,
                       cudaStream_t(context->cuda_stream())>>>(
      in_data, count, roi_data, in_c, in_h, in_w, output_dim, group_size,
      pooled_h, pooled_w, spatial_scale, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PSROIPoolingGPU,
                           PSROIPoolingKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
