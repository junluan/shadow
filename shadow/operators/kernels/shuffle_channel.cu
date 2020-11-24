#include "shuffle_channel.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelShuffleChannel(const float* in_data, int count,
                                     int channel, int spatial_dim, int group,
                                     float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / spatial_dim;
    int sp_in = globalid % spatial_dim;
    int c_in = temp % channel;
    int b = temp / channel;
    temp = channel / group;
    int c_out = (c_in % temp) * group + c_in / temp;

    int dst_offset = (b * channel + c_out) * spatial_dim + sp_in;
    out_data[dst_offset] = in_data[globalid];
  }
}

template <>
void ShuffleChannel<DeviceType::kGPU, float>(const float* in_data, int batch,
                                             int channel, int spatial_dim,
                                             int group, float* out_data,
                                             Context* context) {
  int count = batch * channel * spatial_dim;
  KernelShuffleChannel<<<GetBlocks(count), NumThreads, 0,
                         cudaStream_t(context->stream())>>>(
      in_data, count, channel, spatial_dim, group, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ShuffleChannelGPU,
                           ShuffleChannelKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
