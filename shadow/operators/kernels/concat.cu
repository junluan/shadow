#include "concat.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelConcat(const float* in_data, int count, int num_concats,
                             int concat_size, int out_concat_axis,
                             int in_concat_axis, int offset_concat_axis,
                             float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int total_concat_size = concat_size * in_concat_axis;
    int concat_num = globalid / total_concat_size;
    int concat_index = globalid % total_concat_size;
    int out_index =
        concat_index +
        (concat_num * out_concat_axis + offset_concat_axis) * concat_size;
    out_data[out_index] = in_data[globalid];
  }
}

template <>
void Concat<DeviceType::kGPU, float>(const float* in_data, int count,
                                     int num_concats, int concat_size,
                                     int out_concat_axis, int in_concat_axis,
                                     int offset_concat_axis, float* out_data,
                                     Context* context) {
  KernelConcat<<<GetBlocks(count), NumThreads, 0,
                 cudaStream_t(context->stream())>>>(
      in_data, count, num_concats, concat_size, out_concat_axis, in_concat_axis,
      offset_concat_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ConcatGPU, ConcatKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
