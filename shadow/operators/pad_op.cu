#include "pad_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelPad(const T* in_data, int count, int channel, int in_h,
                          int in_w, int out_h, int out_w, int pad_top,
                          int pad_left, T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w_in = globalid % in_w;
    int h_in = temp % in_h;
    temp = temp / in_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int dst_offset =
        ((b_out * channel + c_out) * out_h + h_in + pad_top) * out_w + w_in +
        pad_left;
    out_data[dst_offset] = in_data[globalid];
  }
}

template <typename T>
void Pad(const T* in_data, const VecInt& in_shape, const VecInt& paddings,
         const VecInt& out_shape, T* out_data) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int pad_top = paddings[0], pad_left = paddings[2];
  int count = batch * channel * in_h * in_w;
  KernelPad<T><<<GetBlocks(count), NumThreads>>>(in_data, count, channel, in_h,
                                                 in_w, out_h, out_w, pad_top,
                                                 pad_left, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Pad(const float* in_data, const VecInt& in_shape,
                  const VecInt& paddings, const VecInt& out_shape,
                  float* out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
