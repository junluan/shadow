#include "depthwise_conv_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelDepthwiseConv(const T *in_data, int count,
                                    const T *weight_data, const T *bias_data,
                                    int in_c, int in_h, int in_w, int out_h,
                                    int out_w, int kernel_size, int stride,
                                    int pad, int bias_term, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w = globalid % out_w;
    int h = (globalid / out_w) % out_h;
    int c = (globalid / out_w / out_h) % in_c;
    int n = globalid / out_w / out_h / in_c;

    const T *in_offset_data = in_data + (n * in_c + c) * in_h * in_w;
    const T *weight_offset_data = weight_data + c * kernel_size * kernel_size;

    int hstart = h * stride - pad, wstart = w * stride - pad;
    int hend = min(hstart + kernel_size, in_h + pad);
    int wend = min(wstart + kernel_size, in_w + pad);
    hstart = max(hstart, 0), wstart = max(wstart, 0);
    hend = min(hend, in_h), wend = min(wend, in_w);
    int khstart = hend < kernel_size ? (kernel_size - hend) : 0;
    int kwstart = wend < kernel_size ? (kernel_size - wend) : 0;
    auto sum_val = T(0);
    for (int kh = hstart; kh < hend; ++kh) {
      for (int kw = wstart; kw < wend; ++kw) {
        sum_val += in_offset_data[kh * in_w + kw] *
                   weight_offset_data[(khstart + kh - hstart) * kernel_size +
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
void DepthwiseConv(const T *in_data, const VecInt &in_shape,
                   const T *weight_data, const T *bias_data, int kernel_size,
                   int stride, int pad, int bias_term, const VecInt &out_shape,
                   T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelDepthwiseConv<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, weight_data, bias_data, in_c, in_h, in_w, out_h, out_w,
      kernel_size, stride, pad, bias_term, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void DepthwiseConv(const float *in_data, const VecInt &in_shape,
                            const float *weight_data, const float *bias_data,
                            int kernel_size, int stride, int pad, int bias_term,
                            const VecInt &out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow