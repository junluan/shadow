#include "deform_psroi_pooling.hpp"

namespace Shadow {

namespace Vision {

__device__ float bilinear_interp(const float* data, float x, float y, int width,
                                 int height) {
  int x1 = static_cast<int>(floorf(x));
  int x2 = static_cast<int>(ceilf(x));
  int y1 = static_cast<int>(floorf(y));
  int y2 = static_cast<int>(ceilf(y));
  auto dist_x = x - x1;
  auto dist_y = y - y1;
  auto value11 = data[y1 * width + x1];
  auto value12 = data[y2 * width + x1];
  auto value21 = data[y1 * width + x2];
  auto value22 = data[y2 * width + x2];
  auto value = (1 - dist_x) * (1 - dist_y) * value11 +
               (1 - dist_x) * dist_y * value12 +
               dist_x * (1 - dist_y) * value21 + dist_x * dist_y * value22;
  return value;
}

__global__ void DeformPSROIPoolForwardKernel(
    int count, const float* bottom_data, float spatial_scale, int channels,
    int height, int width, int pooled_height, int pooled_width,
    const float* bottom_rois, const float* bottom_trans, bool no_trans,
    float trans_std, int sample_per_part, int output_dim, int group_size,
    int part_size, int num_classes, int channels_each_class, float* top_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_width;
    int ph = (globalid / pooled_width) % pooled_height;
    int ctop = (globalid / pooled_width / pooled_height) % output_dim;
    int n = globalid / pooled_width / pooled_height / output_dim;

    const auto* offset_bottom_rois = bottom_rois + n * 5;
    int roi_batch_ind = static_cast<int>(offset_bottom_rois[0]);
    auto roi_start_w = roundf(offset_bottom_rois[1]) * spatial_scale - 0.5f;
    auto roi_start_h = roundf(offset_bottom_rois[2]) * spatial_scale - 0.5f;
    auto roi_end_w =
        (roundf(offset_bottom_rois[3]) + 1.f) * spatial_scale - 0.5f;
    auto roi_end_h =
        (roundf(offset_bottom_rois[4]) + 1.f) * spatial_scale - 0.5f;

    auto roi_width = fmaxf(roi_end_w - roi_start_w, 0.1f);
    auto roi_height = fmaxf(roi_end_h - roi_start_h, 0.1f);
    auto bin_size_h = roi_height / pooled_height;
    auto bin_size_w = roi_width / pooled_width;

    auto sub_bin_size_h = bin_size_h / sample_per_part;
    auto sub_bin_size_w = bin_size_w / sample_per_part;

    int part_h = static_cast<int>(
        floorf(static_cast<float>(ph) / pooled_height * part_size));
    int part_w = static_cast<int>(
        floorf(static_cast<float>(pw) / pooled_width * part_size));
    int class_id = ctop / channels_each_class;
    auto trans_x =
        no_trans
            ? 0.f
            : bottom_trans[(((n * num_classes + class_id) * 2) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  trans_std;
    auto trans_y =
        no_trans
            ? 0.f
            : bottom_trans[(((n * num_classes + class_id) * 2 + 1) * part_size +
                            part_h) *
                               part_size +
                           part_w] *
                  trans_std;

    auto wstart = pw * bin_size_w + roi_start_w;
    wstart += trans_x * roi_width;
    auto hstart = ph * bin_size_h + roi_start_h;
    hstart += trans_y * roi_height;

    int gw = floor(static_cast<float>(pw) * group_size / pooled_width);
    int gh = floor(static_cast<float>(ph) * group_size / pooled_height);
    gw = min(max(gw, 0), group_size - 1);
    gh = min(max(gh, 0), group_size - 1);

    const auto* offset_bottom_data =
        bottom_data + (roi_batch_ind * channels) * height * width;

    double sum = 0;
    int cou = 0;
    for (int ih = 0; ih < sample_per_part; ih++) {
      for (int iw = 0; iw < sample_per_part; iw++) {
        auto w = wstart + iw * sub_bin_size_w;
        auto h = hstart + ih * sub_bin_size_h;
        if (w < -0.5 || w > width - 0.5 || h < -0.5 || h > height - 0.5) {
          continue;
        }
        w = fminf(fmaxf(w, 0.f), width - 1.f);
        h = fminf(fmaxf(h, 0.f), height - 1.f);
        int c = (ctop * group_size + gh) * group_size + gw;
        auto val = bilinear_interp(offset_bottom_data + c * height * width, w,
                                   h, width, height);
        sum += val;
        cou++;
      }
    }
    top_data[globalid] = cou == 0 ? 0.f : static_cast<float>(sum / cou);
  }
}

template <>
void DeformPSROIPooling<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* roi_data,
    const float* trans_data, const VecInt& trans_shape, int num_rois,
    int output_dim, int group_size, int pooled_size, int part_size,
    int sample_per_part, float spatial_scale, float trans_std, bool no_trans,
    float* out_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int num_classes = no_trans ? 1 : trans_shape[1] / 2;
  int channels_each_class = no_trans ? output_dim : output_dim / num_classes;
  int count = num_rois * output_dim * pooled_size * pooled_size;
  DeformPSROIPoolForwardKernel<<<GetBlocks(count), NumThreads, 0,
                                 cudaStream_t(context->cuda_stream())>>>(
      count, in_data, spatial_scale, in_c, in_h, in_w, pooled_size, pooled_size,
      roi_data, trans_data, no_trans, trans_std, sample_per_part, output_dim,
      group_size, part_size, num_classes, channels_each_class, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeformPSROIPoolingGPU,
                           DeformPSROIPoolingKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
