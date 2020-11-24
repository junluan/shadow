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

    roi_data += n * 5;
    int roi_batch_id = static_cast<int>(roi_data[0]);
    int roi_start_w = static_cast<int>(roundf(roi_data[1] * spatial_scale));
    int roi_start_h = static_cast<int>(roundf(roi_data[2] * spatial_scale));
    int roi_end_w = static_cast<int>(roundf(roi_data[3] * spatial_scale));
    int roi_end_h = static_cast<int>(roundf(roi_data[4] * spatial_scale));

    float roi_height = max(roi_end_h - roi_start_h, 1);
    float roi_width = max(roi_end_w - roi_start_w, 1);
    float bin_size_h = roi_height / static_cast<float>(pooled_h);
    float bin_size_w = roi_width / static_cast<float>(pooled_w);

    auto hstart = static_cast<int>(floorf(ph * bin_size_h));
    auto wstart = static_cast<int>(floorf(pw * bin_size_w));
    auto hend = static_cast<int>(ceilf((ph + 1) * bin_size_h));
    auto wend = static_cast<int>(ceilf((pw + 1) * bin_size_w));

    hstart = min(max(hstart + roi_start_h, 0), in_h - 1);
    hend = min(max(hend + roi_start_h, 0), in_h - 1);
    wstart = min(max(wstart + roi_start_w, 0), in_w - 1);
    wend = min(max(wend + roi_start_w, 0), in_w - 1);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gh = ph * group_size / pooled_h;
    int gw = pw * group_size / pooled_w;
    gh = min(max(gh, 0), group_size - 1);
    gw = min(max(gw, 0), group_size - 1);
    int c_in = (c_out * group_size + gh) * group_size + gw;
    const auto* offset_in_data =
        in_data + (roi_batch_id * in_c + c_in) * in_h * in_w;

    double sum_val = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        sum_val += offset_in_data[h * in_w + w];
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
                       cudaStream_t(context->stream())>>>(
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
