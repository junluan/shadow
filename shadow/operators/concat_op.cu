#include "concat_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelConcat(const T *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int total_concat_size = concat_size * bottom_concat_axis;
    int concat_num = globalid / total_concat_size;
    int concat_index = globalid % total_concat_size;
    int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[globalid];
  }
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  KernelConcat<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_concats, concat_size, top_concat_axis,
      bottom_concat_axis, offset_concat_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Concat(const float *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
