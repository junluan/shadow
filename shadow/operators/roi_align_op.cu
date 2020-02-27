#include "roi_align_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelPOIAlign(const T *in_data, int count, const T *roi_data,
                               int in_c, int in_h, int in_w, int pooled_h,
                               int pooled_w, float spatial_scale, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c = (globalid / pooled_w / pooled_h) % in_c;
    int n = globalid / pooled_w / pooled_h / in_c;

    roi_data += n * 5;
    int roi_batch_id = static_cast<int>(roi_data[0]);
    float roi_start_w = roi_data[1] * spatial_scale;
    float roi_start_h = roi_data[2] * spatial_scale;
    float roi_end_w = roi_data[3] * spatial_scale;
    float roi_end_h = roi_data[4] * spatial_scale;

    float roi_height = roi_end_h - roi_start_h;
    float roi_width = roi_end_w - roi_start_w;
    float bin_size_h = roi_height / static_cast<float>(pooled_h - 1);
    float bin_size_w = roi_width / static_cast<float>(pooled_w - 1);

    float src_h_f = roi_start_h + ph * bin_size_h;
    int src_h = static_cast<int>(src_h_f);
    float sh = src_h_f - src_h;

    float src_w_f = roi_start_w + pw * bin_size_w;
    int src_w = static_cast<int>(src_w_f);
    float sw = src_w_f - src_w;

    int src_h_off = (roi_batch_id * in_c + c) * in_h + src_h;

    int src_index_0 = src_h_off * in_w + src_w;
    int src_index_1 = (src_h_off + 1) * in_w + src_w;
    int src_index_2 = src_h_off * in_w + src_w + 1;
    int src_index_3 = (src_h_off + 1) * in_w + src_w + 1;

    out_data[globalid] = static_cast<T>(
        (1 - sh) * (1 - sw) * in_data[src_index_0] +
        sh * (1 - sw) * in_data[src_index_1] +
        (1 - sh) * sw * in_data[src_index_2] + sh * sw * in_data[src_index_3]);
  }
}

template <typename T>
void ROIAlign(const T *in_data, const VecInt &in_shape, const T *roi_data,
              int num_rois, int pooled_h, int pooled_w, float spatial_scale,
              T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * in_c * pooled_h * pooled_w;
  KernelPOIAlign<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, roi_data, in_c, in_h, in_w, pooled_h, pooled_w,
      spatial_scale, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void ROIAlign(const float *, const VecInt &, const float *, int, int,
                       int, float, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
