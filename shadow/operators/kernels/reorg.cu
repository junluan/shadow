#include "reorg.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelReorgDarknet(const float* in_data, int count, int in_c,
                                   int in_h, int in_w, int stride,
                                   float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w = globalid % in_w;
    int h = temp % in_h;
    temp = temp / in_h;
    int c = temp % in_c;
    int b = temp / in_c;

    int out_c = in_c / (stride * stride);

    int c2 = c % out_c;
    int offset = c / out_c;
    int h2 = h * stride + offset / stride;
    int w2 = w * stride + offset % stride;
    int out_index =
        ((b * out_c + c2) * in_h * stride + h2) * in_w * stride + w2;

    out_data[globalid] = in_data[out_index];
  }
}

__global__ void KernelReorgNatural(const float* in_data, int count, int in_c,
                                   int in_h, int in_w, int stride,
                                   float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int out_c = in_c * stride * stride;
    int out_h = in_h / stride, out_w = in_w / stride;

    int temp = globalid / out_w;
    int w = globalid % out_w;
    int h = temp % out_h;
    temp = temp / out_h;
    int c = temp % out_c;
    int b = temp / out_c;

    int offset = c / stride;
    int c2 = offset / stride;
    int h2 = h * stride + offset % stride;
    int w2 = w * stride + c % stride;
    int in_index = ((b * in_c + c2) * in_h + h2) * in_w + w2;

    out_data[globalid] = in_data[in_index];
  }
}

template <>
void Reorg<DeviceType::kGPU, float>(const float* in_data,
                                    const VecInt& in_shape, int type,
                                    int stride, float* out_data,
                                    Context* context) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_c * in_h * in_w;
  if (type == kDarknet) {
    KernelReorgDarknet<<<GetBlocks(count), NumThreads, 0,
                         cudaStream_t(context->stream())>>>(
        in_data, count, in_c, in_h, in_w, stride, out_data);
  } else if (type == kNatural) {
    KernelReorgNatural<<<GetBlocks(count), NumThreads, 0,
                         cudaStream_t(context->stream())>>>(
        in_data, count, in_c, in_h, in_w, stride, out_data);
  } else {
    LOG(FATAL) << "Unsupported reorg type: " << type;
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ReorgGPU, ReorgKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
