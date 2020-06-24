#include "slice.hpp"

namespace Shadow {

namespace Vision {

template <>
void Slice<DeviceType::kCPU, float>(const float* in_data, int count,
                                    int num_slices, int slice_size,
                                    int in_slice_axis, int out_slice_axis,
                                    int offset_slice_axis, float* out_data,
                                    Context* context) {
  for (int n = 0; n < num_slices; ++n) {
    memcpy(out_data + n * out_slice_axis * slice_size,
           in_data + (n * in_slice_axis + offset_slice_axis) * slice_size,
           out_slice_axis * slice_size * sizeof(float));
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(SliceCPU, SliceKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
