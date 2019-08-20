#include "conv_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelIm2Col(const T *in_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size_h,
                             int kernel_size_w, int stride_h, int stride_w,
                             int pad_h, int pad_w, int dilation, int zero_point,
                             int out_h, int out_w, T *col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int h_index = globalid / out_w;
    int h_col = h_index % out_h;
    int w_col = globalid % out_w;
    int c_im = h_index / out_h;
    int c_col = c_im * kernel_size_h * kernel_size_w;
    int h_offset = h_col * stride_h - pad_h;
    int w_offset = w_col * stride_w - pad_w;
    col_data += (c_col * out_h + h_col) * out_w + w_col;
    in_data += offset + (c_im * in_h + h_offset) * in_w + w_offset;
    for (int i = 0; i < kernel_size_h; ++i) {
      for (int j = 0; j < kernel_size_w; ++j) {
        int h_im = h_offset + i * dilation;
        int w_im = w_offset + j * dilation;
        *col_data = (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w)
                        ? in_data[i * dilation * in_w + j * dilation]
                        : static_cast<T>(zero_point);
        col_data += out_h * out_w;
      }
    }
  }
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  KernelIm2Col<T><<<GetBlocks(count), NumThreads>>>(
      in_data, offset, count, in_c, in_h, in_w, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, dilation, zero_point, out_h, out_w,
      col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size_h, int kernel_size_w, int stride_h,
                     int stride_w, int pad_h, int pad_w, int dilation,
                     int zero_point, const VecInt &out_shape, float *col_data);

template <typename T>
__global__ void KernelDepthwise(const T *in_data, int count,
                                const T *weight_data, const T *bias_data,
                                int in_c, int in_h, int in_w, int out_h,
                                int out_w, int kernel_size_h, int kernel_size_w,
                                int stride_h, int stride_w, int pad_h,
                                int pad_w, int bias_term, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w = globalid % out_w;
    int h = (globalid / out_w) % out_h;
    int c = (globalid / out_w / out_h) % in_c;
    int n = globalid / out_w / out_h / in_c;

    const T *in_offset_data = in_data + (n * in_c + c) * in_h * in_w;
    const T *weight_offset_data =
        weight_data + c * kernel_size_h * kernel_size_w;

    int hstart = h * stride_h - pad_h, wstart = w * stride_w - pad_w;
    int hend = min(hstart + kernel_size_h, in_h + pad_h);
    int wend = min(wstart + kernel_size_w, in_w + pad_w);
    hstart = max(hstart, 0), wstart = max(wstart, 0);
    hend = min(hend, in_h), wend = min(wend, in_w);
    int khstart = hend < kernel_size_h ? (kernel_size_h - hend) : 0;
    int kwstart = wend < kernel_size_w ? (kernel_size_w - wend) : 0;
    auto sum_val = T(0);
    for (int kh = hstart; kh < hend; ++kh) {
      for (int kw = wstart; kw < wend; ++kw) {
        sum_val += in_offset_data[kh * in_w + kw] *
                   weight_offset_data[(khstart + kh - hstart) * kernel_size_w +
                                      kwstart + kw - wstart];
      }
    }
    if (bias_term) {
      sum_val += bias_data[c];
    }
    out_data[globalid] = sum_val;
  }
}

template <typename T>
void Depthwise(const T *in_data, const VecInt &in_shape, const T *weight_data,
               const T *bias_data, int kernel_size_h, int kernel_size_w,
               int stride_h, int stride_w, int pad_h, int pad_w, int bias_term,
               const VecInt &out_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelDepthwise<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, weight_data, bias_data, in_c, in_h, in_w, out_h, out_w,
      kernel_size_h, kernel_size_w, stride_h, stride_w, pad_h, pad_w, bias_term,
      out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Depthwise(const float *in_data, const VecInt &in_shape,
                        const float *weight_data, const float *bias_data,
                        int kernel_size_h, int kernel_size_w, int stride_h,
                        int stride_w, int pad_h, int pad_w, int bias_term,
                        const VecInt &out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
