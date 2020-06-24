#include "pad.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPad(const float* in_data, int count, int channel,
                          int in_h, int in_w, int out_h, int out_w, int pad_t,
                          int pad_b, int pad_l, int pad_r, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w_in = globalid % in_w;
    int h_in = temp % in_h;
    temp = temp / in_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int dst_offset =
        ((b_out * channel + c_out) * out_h + h_in + pad_t) * out_w + w_in +
        pad_l;
    if (h_in + pad_t >= 0 && h_in < in_h + pad_b) {
      if (w_in + pad_l >= 0 && w_in < in_w + pad_r) {
        out_data[dst_offset] = in_data[globalid];
      }
    }
  }
}

template <>
void Pad<DeviceType::kGPU, float>(const float* in_data, const VecInt& in_shape,
                                  const VecInt& paddings,
                                  const VecInt& out_shape, float* out_data,
                                  Context* context) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * channel * in_h * in_w;
  KernelPad<<<GetBlocks(count), NumThreads, 0,
              cudaStream_t(context->cuda_stream())>>>(
      in_data, count, channel, in_h, in_w, out_h, out_w, paddings[0],
      paddings[1], paddings[2], paddings[3], out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PadGPU, PadKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
