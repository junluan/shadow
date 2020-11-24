#include "deform_conv.hpp"

namespace Shadow {

namespace Vision {

__device__ float deform_im2col_bilinear(const float* in_data, int in_h,
                                        int in_w, float h_im, float w_im) {
  auto h_low = static_cast<int>(floorf(h_im));
  auto w_low = static_cast<int>(floorf(w_im));
  int h_high = h_low + 1, w_high = w_low + 1;
  float lh = h_im - h_low, lw = w_im - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = (h_low >= 0 && w_low >= 0) ? in_data[h_low * in_w + w_low] : 0;
  float v2 =
      (h_low >= 0 && w_high <= in_w - 1) ? in_data[h_low * in_w + w_high] : 0;
  float v3 =
      (h_high <= in_h - 1 && w_low >= 0) ? in_data[h_high * in_w + w_low] : 0;
  float v4 = (h_high <= in_h - 1 && w_high <= in_w - 1)
                 ? in_data[h_high * in_w + w_high]
                 : 0;
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

__global__ void deform_im2col_gpu_kernel(
    int count, const float* in_data, int in_data_offset,
    const float* offset_data, int offset_data_offset, int in_h, int in_w,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation, int zero_point, int channel_per_deform_group,
    int out_h, int out_w, float* col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w_col = globalid % out_w;
    int h_col = (globalid / out_w) % out_h;
    int c_im = (globalid / out_w) / out_h;

    int c_col = c_im * kernel_size_h * kernel_size_w;
    int deform_group_idx = c_im / channel_per_deform_group;
    int h_in = h_col * stride_h - pad_h, w_in = w_col * stride_w - pad_w;

    const auto* in_data_ptr = in_data + in_data_offset + c_im * in_h * in_w;
    const auto* offset_data_ptr =
        offset_data + offset_data_offset +
        deform_group_idx * 2 * kernel_size_h * kernel_size_w * out_h * out_w;
    auto* col_data_ptr = col_data + (c_col * out_h + h_col) * out_w + w_col;

    for (int i = 0; i < kernel_size_h; ++i) {
      for (int j = 0; j < kernel_size_w; ++j) {
        int offset_data_idx = 2 * (i * kernel_size_w + j);
        auto offset_h =
            offset_data_ptr[(offset_data_idx * out_h + h_col) * out_w + w_col];
        auto offset_w =
            offset_data_ptr[((offset_data_idx + 1) * out_h + h_col) * out_w +
                            w_col];
        auto val = static_cast<float>(zero_point);
        auto h_im = h_in + i * dilation + offset_h;
        auto w_im = w_in + j * dilation + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < in_h && w_im < in_w) {
          val = deform_im2col_bilinear(in_data_ptr, in_h, in_w, h_im, w_im);
        }
        *col_data_ptr = val;
        col_data_ptr += out_h * out_w;
      }
    }
  }
}

template <>
void DeformIm2Col<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, int in_data_offset,
    const float* offset_data, int offset_data_offset, int kernel_size_h,
    int kernel_size_w, int stride_h, int stride_w, int pad_h, int pad_w,
    int dilation, int deform_group, int zero_point, const VecInt& out_shape,
    float* col_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  deform_im2col_gpu_kernel<<<GetBlocks(count), NumThreads, 0,
                             cudaStream_t(context->stream())>>>(
      count, in_data, in_data_offset, offset_data, offset_data_offset, in_h,
      in_w, kernel_size_h, kernel_size_w, stride_h, stride_w, pad_h, pad_w,
      dilation, zero_point, in_c / deform_group, out_h, out_w, col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeformConvGPU,
                           DeformConvKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
