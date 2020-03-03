#include "conv_op.hpp"

namespace Shadow {

namespace Vision {

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

template void Im2Col(const float *, const VecInt &, int, int, int, int, int,
                     int, int, int, int, const VecInt &, float *);

template <typename T>
__global__ void KernelDepthwise(const T *in_data, int count,
                                const T *weight_data, const T *bias_data,
                                int in_c, int in_h, int in_w, int out_h,
                                int out_w, int kernel_size_h, int kernel_size_w,
                                int stride_h, int stride_w, int pad_h,
                                int pad_w, int dilation, int bias_term,
                                T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w = globalid % out_w;
    int h = (globalid / out_w) % out_h;
    int c = (globalid / out_w / out_h) % in_c;
    int n = globalid / out_w / out_h / in_c;

    const T *in_offset_data = in_data + (n * in_c + c) * in_h * in_w;
    const T *weight_offset_data =
        weight_data + c * kernel_size_h * kernel_size_w;

    auto sum_val = T(0);
    for (int kh = 0; kh < kernel_size_h; ++kh) {
      for (int kw = 0; kw < kernel_size_w; ++kw) {
        int h_in = h * stride_h - pad_h + kh * dilation;
        int w_in = w * stride_w - pad_w + kw * dilation;
        if (h_in >= 0 && h_in < in_h && w_in >= 0 && w_in < in_w) {
          sum_val += in_offset_data[h_in * in_w + w_in] * *weight_offset_data;
        }
        weight_offset_data++;
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
               int stride_h, int stride_w, int pad_h, int pad_w, int dilation,
               int bias_term, const VecInt &out_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelDepthwise<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, weight_data, bias_data, in_c, in_h, in_w, out_h, out_w,
      kernel_size_h, kernel_size_w, stride_h, stride_w, pad_h, pad_w, dilation,
      bias_term, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Depthwise(const float *, const VecInt &, const float *,
                        const float *, int, int, int, int, int, int, int, int,
                        const VecInt &, float *);

}  // namespace Vision

}  // namespace Shadow
