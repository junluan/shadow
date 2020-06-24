#include "deform_conv.hpp"

namespace Shadow {

namespace Vision {

__device__ float deform_im2col_bilinear(const float* bottom_data,
                                        int data_width, int height, int width,
                                        float h, float w) {
  int h_low = static_cast<int>(floorf(h)), w_low = static_cast<int>(floorf(w));
  int h_high = h_low + 1, w_high = w_low + 1;
  if (h_low >= height - 1) {
    h_high = h_low = height - 1;
    h = h_low;
  }
  if (w_low >= width - 1) {
    w_high = w_low = width - 1;
    w = w_low;
  }

  auto lh = h - h_low;
  auto lw = w - w_low;
  auto hh = 1.f - lh, hw = 1.f - lw;

  auto v1 = bottom_data[h_low * data_width + w_low];
  auto v2 = bottom_data[h_low * data_width + w_high];
  auto v3 = bottom_data[h_high * data_width + w_low];
  auto v4 = bottom_data[h_high * data_width + w_high];
  auto w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

__global__ void deform_im2col_gpu_kernel(
    int n, const float* data_im, const float* data_offset, int im_offset,
    int height, int width, int kernel_h, int kernel_w, int pad_h, int pad_w,
    int stride_h, int stride_w, int dilation_h, int dilation_w, int zero_point,
    int channel_per_deform_group, int height_col, int width_col,
    float* data_col) {
  CUDA_KERNEL_LOOP(globalid, n) {
    int w_col = globalid % width_col;
    int h_col = (globalid / width_col) % height_col;
    int c_im = (globalid / width_col) / height_col;
    int c_col = c_im * kernel_h * kernel_w;

    int deform_group_index = c_im / channel_per_deform_group;

    int h_in = h_col * stride_h - pad_h;
    int w_in = w_col * stride_w - pad_w;
    auto* data_col_ptr =
        data_col + (c_col * height_col + h_col) * width_col + w_col;
    const auto* data_im_ptr =
        data_im + im_offset + (c_im * height + h_in) * width + w_in;
    const auto* data_offset_ptr = data_offset + deform_group_index * 2 *
                                                    kernel_h * kernel_w *
                                                    height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        auto offset_h = data_offset_ptr[data_offset_h_ptr];
        auto offset_w = data_offset_ptr[data_offset_w_ptr];
        auto val = static_cast<float>(zero_point);
        auto h_im = h_in + i * dilation_h + offset_h;
        auto w_im = w_in + j * dilation_w + offset_w;
        if (h_im >= 0 && w_im >= 0 && h_im < height && w_im < width) {
          auto map_h = i * dilation_h + offset_h;
          auto map_w = j * dilation_w + offset_w;
          int cur_height = height - h_in;
          int cur_width = width - w_in;
          val = deform_im2col_bilinear(data_im_ptr, width, cur_height,
                                       cur_width, map_h, map_w);
        }
        *data_col_ptr = val;
        data_col_ptr += height_col * width_col;
      }
    }
  }
}

template <>
void DeformIm2Col<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, const float* offset_data,
    int offset, int deform_group, int kernel_size, int stride, int pad,
    int dilation, int zero_point, const VecInt& out_shape, float* out_data,
    Context* context) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int channel_per_deform_group = in_c / deform_group;
  int count = in_c * out_h * out_w;
  deform_im2col_gpu_kernel<<<GetBlocks(count), NumThreads, 0,
                             cudaStream_t(context->cuda_stream())>>>(
      count, in_data, offset_data, offset, in_h, in_w, kernel_size, kernel_size,
      pad, pad, stride, stride, dilation, dilation, zero_point,
      channel_per_deform_group, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(DeformConvGPU,
                           DeformConvKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
