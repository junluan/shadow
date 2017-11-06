#include "pooling_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelPooling(const T *in_data, int count, int in_c, int in_h,
                              int in_w, int kernel_size, int stride, int pad,
                              int mode, int out_h, int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int j_out = globalid % out_w;
    int i_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % in_c;
    int b_out = temp / in_c;

    int kistart = i_out * stride - pad, kjstart = j_out * stride - pad;
    int kiend = min(kistart + kernel_size, in_h + pad);
    int kjend = min(kjstart + kernel_size, in_w + pad);
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
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, kernel_size, stride, pad, mode, out_h,
      out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Pooling(const float *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow