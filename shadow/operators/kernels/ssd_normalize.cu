#include "ssd_normalize.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelChannelSum(const float* out_data, int val_count,
                                 int channel, int inner_num, float eps,
                                 float* val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const auto* out_data_offset = out_data + n * channel * inner_num + s;
    double sum = 0;
    for (int c = 0; c < channel; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[globalid] = sqrtf(static_cast<float>(sum) + eps);
  }
}

__global__ void KernelChannelDiv(const float* in_data, const float* val_data,
                                 int count, int channel, int inner_num,
                                 float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / channel / inner_num, s = globalid % inner_num;
    out_data[globalid] = in_data[globalid] / val_data[n * inner_num + s];
  }
}

template <>
void SSDNormalize<DeviceType::kGPU, float>(const float* in_data, int outer_num,
                                           int channel, int inner_num,
                                           float eps, float* val_data,
                                           float* out_data, Context* context) {
  int val_count = outer_num * inner_num, count = val_count * channel;
  KernelChannelSum<<<GetBlocks(val_count), NumThreads, 0,
                     cudaStream_t(context->stream())>>>(
      out_data, val_count, channel, inner_num, eps, val_data);
  KernelChannelDiv<<<GetBlocks(count), NumThreads, 0,
                     cudaStream_t(context->stream())>>>(
      in_data, val_data, count, channel, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SSDNormalizeGPU,
                           SSDNormalizeKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
