#include "deform_psroi_pooling_op.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
__device__ T bilinear_interp(const T *data, T x, T y, int width, int height) {
  int x1 = floor(x);
  int x2 = ceil(x);
  int y1 = floor(y);
  int y2 = ceil(y);
  T dist_x = static_cast<T>(x - x1);
  T dist_y = static_cast<T>(y - y1);
  T value11 = data[y1 * width + x1];
  T value12 = data[y2 * width + x1];
  T value21 = data[y1 * width + x2];
  T value22 = data[y2 * width + x2];
  T value = (1 - dist_x) * (1 - dist_y) * value11 +
            (1 - dist_x) * dist_y * value12 + dist_x * (1 - dist_y) * value21 +
            dist_x * dist_y * value22;
  return value;
}

template <typename T>
__global__ void DeformPSROIPoolForwardKernel(
    int count, const T *bottom_data, T spatial_scale, int channels, int height,
    int width, int pooled_height, int pooled_width, const T *bottom_rois,
    const T *bottom_trans, bool no_trans, T trans_std, int sample_per_part,
    int output_dim, int group_size, int part_size, int num_classes,
    int channels_each_class, T *top_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_width;
    int ph = (globalid / pooled_width) % pooled_height;
    int ctop = (globalid / pooled_width / pooled_height) % output_dim;
    int n = globalid / pooled_width / pooled_height / output_dim;

    const T *offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = offset_bottom_rois[0];
    T roi_start_w =
        static_cast<T>(round(offset_bottom_rois[1])) * spatial_scale - 0.5;
    T roi_start_h =
        static_cast<T>(round(offset_bottom_rois[2])) * spatial_scale - 0.5;
    T roi_end_w =
        static_cast<T>(round(offset_bottom_rois[3]) + 1.) * spatial_scale - 0.5;
    T roi_end_h =
        static_cast<T>(round(offset_bottom_rois[4]) + 1.) * spatial_scale - 0.5;

    T roi_width = max(roi_end_w - roi_start_w, 0.1);
    T roi_height = max(roi_end_h - roi_start_h, 0.1);
    T bin_size_h = roi_height / static_cast<T>(pooled_height);
    T bin_size_w = roi_width / static_cast<T>(pooled_width);

    T sub_bin_size_h = bin_size_h / static_cast<T>(sample_per_part);
    T sub_bin_size_w = bin_size_w / static_cast<T>(sample_per_part);

    int part_h = floor(static_cast<T>(ph) / pooled_height * part_size);
    int part_w = floor(static_cast<T>(pw) / pooled_width * part_size);
    int class_id = ctop / channels_each_class;
    T trans_x =
        no_trans
            ? static_cast<T>(0)
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  trans_std;
    T trans_y =
        no_trans
            ? static_cast<T>(0)
            : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  trans_std;

    T wstart = static_cast<T>(pw) * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    T hstart = static_cast<T>(ph) * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    T sum = 0;
    int count = 0;
    int gw = floor(static_cast<T>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<T>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    const T *offset_bottom_data =
        bottom_data + (roi_batch_ind * channels) * height * width;
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        T w = wstart + iw * sub_bin_size_w;
        T h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = min(max(w, 0.), width - 1.);
        h = min(max(h, 0.), height - 1.);
        int c = (ctop * group_size + gh) * group_size + gw;
        T val = bilinear_interp(offset_bottom_data + c * height * width, w, h,
                                width, height);
        sum += val;
        count++;
      }
    }
    top_data[globalid] = count == 0 ? static_cast<T>(0) : sum / count;
  }
}

template <typename T>
void DeformPSROIPooling(const T *in_data, const VecInt &in_shape,
                        const T *roi_data, const T *trans_data,
                        const VecInt &trans_shape, int num_rois, int output_dim,
                        int group_size, int pooled_size, int part_size,
                        int sample_per_part, float spatial_scale,
                        float trans_std, bool no_trans, T *out_data,
                        Context *context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int num_classes = no_trans ? 1 : trans_shape[1] / 2;
  int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
  int count = num_rois * output_dim * pooled_size * pooled_size;
  DeformPSROIPoolForwardKernel<T><<<GetBlocks(count), NumThreads, 0,
                                    cudaStream_t(context->cuda_stream())>>>(
      count, in_data, spatial_scale, in_c, in_h, in_w, pooled_size, pooled_size,
      roi_data, trans_data, no_trans, trans_std, sample_per_part, output_dim,
      group_size, part_size, num_classes, channels_each_class, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void DeformPSROIPooling(const float *, const VecInt &, const float *,
                                 const float *, const VecInt &, int, int, int,
                                 int, int, int, float, float, bool, float *,
                                 Context *);

}  // namespace Vision

}  // namespace Shadow
