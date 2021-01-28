#include "roi_align.hpp"

namespace Shadow {

namespace Vision {

__device__ float roi_align_bilinear(const float* data, float x, float y,
                                    int width, int height) {
  if (x < -1.0 || x > width || y < -1.0 || y > height) {
    return 0;
  }

  x = fmaxf(x, 0.f), y = fmaxf(y, 0.f);

  auto h_low = static_cast<int>(floorf(y));
  auto w_low = static_cast<int>(floorf(x));
  int h_high = h_low + 1, w_high = w_low + 1;

  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    y = static_cast<float>(h_low);
  }

  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    x = static_cast<float>(w_low);
  }

  float lh = y - h_low, lw = x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = data[h_low * width + w_low];
  float v2 = data[h_low * width + w_high];
  float v3 = data[h_high * width + w_low];
  float v4 = data[h_high * width + w_high];
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

__global__ void KernelPOIAlign(const float* in_data, int count,
                               const float* roi_data, int in_c, int in_h,
                               int in_w, int pooled_h, int pooled_w,
                               float spatial_scale, int sampling_ratio,
                               bool align_corners, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int pw = globalid % pooled_w;
    int ph = (globalid / pooled_w) % pooled_h;
    int c = (globalid / pooled_w / pooled_h) % in_c;
    int n = globalid / pooled_w / pooled_h / in_c;

    roi_data += n * 5;
    int roi_batch_id = static_cast<int>(roi_data[0]);

    float offset = align_corners ? 0.5f : 0.f;
    float roi_start_w = roi_data[1] * spatial_scale - offset;
    float roi_start_h = roi_data[2] * spatial_scale - offset;
    float roi_end_w = roi_data[3] * spatial_scale - offset;
    float roi_end_h = roi_data[4] * spatial_scale - offset;

    float roi_height = roi_end_h - roi_start_h;
    float roi_width = roi_end_w - roi_start_w;

    if (!align_corners) {
      roi_height = fmaxf(roi_height, 1.f);
      roi_width = fmaxf(roi_width, 1.f);
    }

    float bin_size_h = roi_height / pooled_h;
    float bin_size_w = roi_width / pooled_w;

    int roi_bin_grid_h = sampling_ratio > 0
                             ? sampling_ratio
                             : static_cast<int>(ceilf(bin_size_h));
    int roi_bin_grid_w = sampling_ratio > 0
                             ? sampling_ratio
                             : static_cast<int>(ceilf(bin_size_w));

    float grid_size = max(roi_bin_grid_h * roi_bin_grid_w, 1);

    in_data += (roi_batch_id * in_c + c) * in_h * in_w;

    double sum_val = 0;
    for (int h = 0; h < roi_bin_grid_h; ++h) {
      float y = roi_start_h + ph * bin_size_h +
                (h + 0.5f) * bin_size_h / roi_bin_grid_h;
      for (int w = 0; w < roi_bin_grid_w; ++w) {
        float x = roi_start_w + pw * bin_size_w +
                  (w + 0.5f) * bin_size_w / roi_bin_grid_w;
        sum_val += roi_align_bilinear(in_data, x, y, in_w, in_h);
      }
    }

    out_data[globalid] = static_cast<float>(sum_val / grid_size);
  }
}

template <>
void ROIAlign<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* roi_data,
    int num_rois, int pooled_h, int pooled_w, float spatial_scale,
    int sampling_ratio, bool align_corners, float* out_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int count = num_rois * in_c * pooled_h * pooled_w;
  KernelPOIAlign<<<GetBlocks(count), NumThreads, 0,
                   cudaStream_t(context->stream())>>>(
      in_data, count, roi_data, in_c, in_h, in_w, pooled_h, pooled_w,
      spatial_scale, sampling_ratio, align_corners, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ROIAlignGPU,
                           ROIAlignKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
