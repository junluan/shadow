#include "shuffle_channel_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelShuffleChannel(const T *in_data, int count, int channel,
                                     int spatial_dim, int group, T *out_data) {
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

template <typename T>
void ShuffleChannel(const T *in_data, int batch, int channel, int spatial_dim,
                    int group, T *out_data) {
  int count = batch * channel * spatial_dim;
  KernelShuffleChannel<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, channel, spatial_dim, group, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void ShuffleChannel(const float *, int, int, int, int, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
