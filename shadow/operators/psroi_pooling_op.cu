#include "psroi_pooling_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelPSROIPooling(const T *in_data, int count,
                                   const T *roi_data, int in_c, int in_h,
                                   int in_w, int output_dim, int group_size,
                                   int pooled_h, int pooled_w,
                                   float spatial_scale, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c_out = (globalid / pooled_w / pooled_h) % output_dim;
    int n = globalid / pooled_w / pooled_h / output_dim;

    const T *offset_bottom_rois = roi_data + n * 5;
    int roi_batch_id = offset_bottom_rois[0];
    T roi_start_w =
        static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale;
    T roi_start_h =
        static_cast<T>(round(offset_bottom_rois[2])) * spatial_scale;
    T roi_end_w =
        static_cast<T>(round(offset_bottom_rois[3]) + 1) * spatial_scale;
    T roi_end_h =
        static_cast<T>(round(offset_bottom_rois[4]) + 1) * spatial_scale;

    T roi_height = max(roi_end_h - roi_start_h, T(0.1));
    T roi_width = max(roi_end_w - roi_start_w, T(0.1));
    T bin_size_h = roi_height / static_cast<T>(pooled_h);
    T bin_size_w = roi_width / static_cast<T>(pooled_w);

    int hstart = static_cast<int>(floor(ph * bin_size_h + roi_start_h));
    int wstart = static_cast<int>(floor(pw * bin_size_w + roi_start_w));
    int hend = static_cast<int>(ceil((ph + 1) * bin_size_h + roi_start_h));
    int wend = static_cast<int>(ceil((pw + 1) * bin_size_w + roi_start_w));

    hstart = min(max(hstart, 0), in_h);
    hend = min(max(hend, 0), in_h);
    wstart = min(max(wstart, 0), in_w);
    wend = min(max(wend, 0), in_w);

    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int gh = static_cast<int>(floor(static_cast<T>(ph) * group_size / in_h));
    int gw = static_cast<int>(floor(static_cast<T>(pw) * group_size / in_w));
    gh = min(max(gh, 0), group_size - 1);
    gw = min(max(gw, 0), group_size - 1);
    int c_in = (c_out * group_size + gh) * group_size + gw;
    const T *offset_bottom_data =
        in_data + (roi_batch_id * in_c + c_in) * in_h * in_w;

    auto sum_val = T(0);
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        sum_val += offset_bottom_data[h * in_w + w];
      }
    }
    T bin_area = (hend - hstart) * (wend - wstart);
    out_data[globalid] = is_empty ? T(0) : sum_val / bin_area;
  }
}

template <typename T>
void PSROIPooling(const T *in_data, const VecInt &in_shape, const T *roi_data,
                  int num_rois, int output_dim, int group_size, int pooled_h,
                  int pooled_w, float spatial_scale, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * output_dim * pooled_h * pooled_w;
  KernelPSROIPooling<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, roi_data, in_c, in_h, in_w, output_dim, group_size,
      pooled_h, pooled_w, spatial_scale, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void PSROIPooling(const float *in_data, const VecInt &in_shape,
                           const float *roi_data, int num_rois, int output_dim,
                           int group_size, int pooled_h, int pooled_w,
                           float spatial_scale, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
