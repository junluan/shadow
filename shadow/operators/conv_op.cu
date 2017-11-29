#include "conv_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelIm2Col(const T *in_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int dilation, int zero_point, int out_h,
                             int out_w, T *col_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int h_index = globalid / out_w;
    int h_col = h_index % out_h;
    int w_col = globalid % out_w;
    int c_im = h_index / out_h;
    int c_col = c_im * kernel_size * kernel_size;
    int h_offset = h_col * stride - pad;
    int w_offset = w_col * stride - pad;
    col_data += (c_col * out_h + h_col) * out_w + w_col;
    in_data += offset + (c_im * in_h + h_offset) * in_w + w_offset;
    for (int i = 0; i < kernel_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
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
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *col_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  KernelIm2Col<T><<<GetBlocks(count), NumThreads>>>(
      in_data, offset, count, in_c, in_h, in_w, kernel_size, stride, pad,
      dilation, zero_point, out_h, out_w, col_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     int zero_point, const VecInt &out_shape, float *col_data);
#endif

}  // namespace Vision

}  // namespace Shadow