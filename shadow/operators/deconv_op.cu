#include "deconv_op.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
__global__ void KernelCol2Im(const T *col_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size_h,
                             int kernel_size_w, int stride_h, int stride_w,
                             int pad_h, int pad_w, int dilation, int out_h,
                             int out_w, T *in_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int w_im = globalid % in_w + pad_w;
    int h_im = (globalid / in_w) % in_h + pad_h;
    int c_im = globalid / (in_w * in_h);
    int kernel_extent_h = (kernel_size_h - 1) * dilation + 1;
    int kernel_extent_w = (kernel_size_w - 1) * dilation + 1;
    int w_col_start =
        (w_im < kernel_extent_w) ? 0 : (w_im - kernel_extent_w) / stride_w + 1;
    int w_col_end = min(w_im / stride_w + 1, out_w);
    int h_col_start =
        (h_im < kernel_extent_h) ? 0 : (h_im - kernel_extent_h) / stride_h + 1;
    int h_col_end = min(h_im / stride_h + 1, out_h);
    T val = T(0);
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
      for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
        int h_k = (h_im - h_col * stride_h);
        int w_k = (w_im - w_col * stride_w);
        if (h_k % dilation == 0 && w_k % dilation == 0) {
          h_k /= dilation, w_k /= dilation;
          int col_index =
              (((c_im * kernel_size_h + h_k) * kernel_size_w + w_k) * out_h +
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
            int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
            int pad_h, int pad_w, int dilation, const VecInt &out_shape,
            T *in_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * in_h * in_w;
  KernelCol2Im<T><<<GetBlocks(count), NumThreads>>>(
      col_data, offset, count, in_c, in_h, in_w, kernel_size_h, kernel_size_w,
      stride_h, stride_w, pad_h, pad_w, dilation, out_h, out_w, in_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Col2Im(const float *, const VecInt &, int, int, int, int, int,
                     int, int, int, const VecInt &, float *);

}  // namespace Vision

}  // namespace Shadow
