#include "pooling_op.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
__global__ void KernelPooling(const T *in_data, int count, int in_c, int in_h,
                              int in_w, int kernel_size_h, int kernel_size_w,
                              int stride_h, int stride_w, int pad_h, int pad_w,
                              int mode, int out_h, int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int j_out = globalid % out_w;
    int i_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % in_c;
    int b_out = temp / in_c;

    int kistart = i_out * stride_h - pad_h, kjstart = j_out * stride_w - pad_w;
    int kiend = min(kistart + kernel_size_h, in_h + pad_h);
    int kjend = min(kjstart + kernel_size_w, in_w + pad_w);
    int pool_size = (kiend - kistart) * (kjend - kjstart);
    kistart = max(kistart, 0), kjstart = max(kjstart, 0);
    kiend = min(kiend, in_h), kjend = min(kjend, in_w);

    in_data += (b_out * in_c + c_out) * in_h * in_w;

    T max_val = -FLT_MAX, sum_val = T(0);
    for (int ki = kistart; ki < kiend; ++ki) {
      for (int kj = kjstart; kj < kjend; ++kj) {
        T value = in_data[ki * in_w + kj];
        max_val = max(max_val, value);
        sum_val += value;
      }
    }
    out_data[globalid] = (mode == 0) ? max_val : sum_val / pool_size;
  }
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size_h,
             int kernel_size_w, int stride_h, int stride_w, int pad_h,
             int pad_w, int mode, const VecInt &out_shape, T *out_data,
             Context *context) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling<T><<<GetBlocks(count), NumThreads, 0,
                     cudaStream_t(context->cuda_stream())>>>(
      in_data, count, in_c, in_h, in_w, kernel_size_h, kernel_size_w, stride_h,
      stride_w, pad_h, pad_w, mode, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Pooling(const float *, const VecInt &, int, int, int, int, int,
                      int, int, const VecInt &, float *, Context *);

}  // namespace Vision

}  // namespace Shadow
