#include "deconv_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelCol2Im(const T *col_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int dilation, int out_h, int out_w,
                             T *in_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w_im = globalid % in_w + pad;
    int h_im = (globalid / in_w) % in_h + pad;
    int c_im = globalid / (in_w * in_h);
    int kernel_extent = (kernel_size - 1) * dilation + 1;
    int w_col_start =
        (w_im < kernel_extent) ? 0 : (w_im - kernel_extent) / stride + 1;
    int w_col_end = min(w_im / stride + 1, out_w);
    int h_col_start =
        (h_im < kernel_extent) ? 0 : (h_im - kernel_extent) / stride + 1;
    int h_col_end = min(h_im / stride + 1, out_h);
    T val = T(0);
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride);
        int w_k = (w_im - w_col * stride);
        if (h_k % dilation == 0 && w_k % dilation == 0) {
          h_k /= dilation, w_k /= dilation;
          int col_index =
              (((c_im * kernel_size + h_k) * kernel_size + w_k) * out_h +
               h_col) *
                  out_w +
              w_col;
          val += col_data[col_index];
        }
      }
    }
    in_data[globalid + offset] = val;
  }
}

template <typename T>
void Col2Im(const T *col_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *in_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * in_h * in_w;
  KernelCol2Im<T><<<GetBlocks(count), NumThreads>>>(
      col_data, offset, count, in_c, in_h, in_w, kernel_size, stride, pad,
      dilation, out_h, out_w, in_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Col2Im(const float *col_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     const VecInt &out_shape, float *in_data);
#endif

}  // namespace Vision

}  // namespace Shadow
