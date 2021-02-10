#include "deform_conv.hpp"

namespace Shadow {

namespace Vision {

__device__ float deform_im2col_bilinear(const float* data, float x, float y,
                                        int width, int height) {
  auto h_low = static_cast<int>(floorf(y));
  auto w_low = static_cast<int>(floorf(x));
  int h_high = h_low + 1, w_high = w_low + 1;
  float lh = y - h_low, lw = x - w_low;
  float hh = 1 - lh, hw = 1 - lw;
  float v1 = (h_low >= 0 && w_low >= 0) ? data[h_low * width + w_low] : 0;
  float v2 =
      (h_low >= 0 && w_high <= width - 1) ? data[h_low * width + w_high] : 0;
  float v3 =
      (h_high <= height - 1 && w_low >= 0) ? data[h_high * width + w_low] : 0;
  float v4 = (h_high <= height - 1 && w_high <= width - 1)
                 ? data[h_high * width + w_high]
                 : 0;
  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;
  return (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
}

__global__ void deform_im2col_gpu_kernel(
    int count, const float* in_data, const float* offset_data, int in_h,
    int in_w, int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
    int pad_h, int pad_w, int dilation_h, int dilation_w,
    int channel_per_deform_group, int out_h, int out_w, float* col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    int c_in = temp / out_h;
    int c_out = c_in * kernel_size_h * kernel_size_w;
    int deform_group_idx = c_in / channel_per_deform_group;
    int h_offset = h_out * stride_h - pad_h;
    int w_offset = w_out * stride_w - pad_w;
    in_data += c_in * in_h * in_w;
    offset_data +=
        (2 * deform_group_idx * kernel_size_h * kernel_size_w * out_h + h_out) *
            out_w +
        w_out;
    col_data += (c_out * out_h + h_out) * out_w + w_out;
    for (int kh = 0; kh < kernel_size_h; ++kh) {
      for (int kw = 0; kw < kernel_size_w;
           ++kw, offset_data += 2 * out_h * out_w) {
        auto h_in = h_offset + kh * dilation_h + offset_data[0];
        auto w_in = w_offset + kw * dilation_w + offset_data[out_h * out_w];
        if (h_in > -1 && h_in < in_h && w_in > -1 && w_in < in_w) {
          *col_data = deform_im2col_bilinear(in_data, w_in, h_in, in_w, in_h);
        } else {
          *col_data = 0.f;
        }
        col_data += out_h * out_w;
      }
    }
  }
}

template <>
void DeformIm2Col2D<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* offset_data,
    int kernel_size_h, int kernel_size_w, int stride_h, int stride_w, int pad_h,
    int pad_w, int dilation_h, int dilation_w, int deform_group,
    const VecInt& out_shape, float* col_data, Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  deform_im2col_gpu_kernel<<<GetBlocks(count), NumThreads, 0,
                             cudaStream_t(context->stream())>>>(
      count, in_data, offset_data, in_h, in_w, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
      in_c / deform_group, out_h, out_w, col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeformConvGPU,
                           DeformConvKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
