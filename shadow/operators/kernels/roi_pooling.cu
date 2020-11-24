#include "roi_pooling.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPOIPooling(const float* in_data, int count,
                                 const float* roi_data, int in_c, int in_h,
                                 int in_w, int pooled_h, int pooled_w,
                                 float spatial_scale, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c = (globalid / pooled_w / pooled_h) % in_c;
    int n = globalid / pooled_w / pooled_h / in_c;

    roi_data += n * 5;
    int roi_batch_id = static_cast<int>(roi_data[0]);
    int roi_start_w = static_cast<int>(roundf(roi_data[1] * spatial_scale));
    int roi_start_h = static_cast<int>(roundf(roi_data[2] * spatial_scale));
    int roi_end_w = static_cast<int>(roundf(roi_data[3] * spatial_scale));
    int roi_end_h = static_cast<int>(roundf(roi_data[4] * spatial_scale));

    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    auto bin_size_h = roi_height / static_cast<float>(pooled_h);
    auto bin_size_w = roi_width / static_cast<float>(pooled_w);

    int hstart = static_cast<int>(floorf(ph * bin_size_h));
    int wstart = static_cast<int>(floorf(pw * bin_size_w));
    int hend = static_cast<int>(ceilf((ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceilf((pw + 1) * bin_size_w));

    hstart = min(max(hstart + roi_start_h, 0), in_h);
    hend = min(max(hend + roi_start_h, 0), in_h);
    wstart = min(max(wstart + roi_start_w, 0), in_w);
    wend = min(max(wend + roi_start_w, 0), in_w);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    in_data += (roi_batch_id * in_c + c) * in_h * in_w;

    auto max_val = is_empty ? 0.f : in_data[hstart * in_w + wstart];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        max_val = fmaxf(max_val, in_data[h * in_w + w]);
      }
    }
    out_data[globalid] = max_val;
  }
}

template <>
void ROIPooling<DeviceType::kGPU, float>(const float* in_data,
                                         const VecInt& in_shape,
                                         const float* roi_data, int num_rois,
                                         int pooled_h, int pooled_w,
                                         float spatial_scale, float* out_data,
                                         Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * in_c * pooled_h * pooled_w;
  KernelPOIPooling<<<GetBlocks(count), NumThreads, 0,
                     cudaStream_t(context->stream())>>>(
      in_data, count, roi_data, in_c, in_h, in_w, pooled_h, pooled_w,
      spatial_scale, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ROIPoolingGPU,
                           ROIPoolingKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
