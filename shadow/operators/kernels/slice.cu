#include "slice.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelSlice(const float* in_data, int count, int num_slices,
                            int slice_size, int in_slice_axis,
                            int out_slice_axis, int offset_slice_axis,
                            float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int total_slice_size = slice_size * out_slice_axis;
    int slice_num = globalid / total_slice_size;
    int slice_index = globalid % total_slice_size;
    int in_index = slice_index +
                   (slice_num * in_slice_axis + offset_slice_axis) * slice_size;
    out_data[globalid] = in_data[in_index];
  }
}

template <>
void Slice<DeviceType::kGPU, float>(const float* in_data, int count,
                                    int num_slices, int slice_size,
                                    int in_slice_axis, int out_slice_axis,
                                    int offset_slice_axis, float* out_data,
                                    Context* context) {
  KernelSlice<<<GetBlocks(count), NumThreads, 0,
                cudaStream_t(context->stream())>>>(
      in_data, count, num_slices, slice_size, in_slice_axis, out_slice_axis,
      offset_slice_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SliceGPU, SliceKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
