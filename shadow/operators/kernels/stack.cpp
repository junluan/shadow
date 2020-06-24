#include "stack.hpp"

namespace Shadow {

namespace Vision {

template <>
void Stack<DeviceType::kCPU, float>(const float* in_data, int count,
                                    int num_stacks, int stack_size,
                                    int out_stack_axis, int offset_stack_axis,
                                    float* out_data, Context* context) {
  for (int n = 0; n < num_stacks; ++n) {
    memcpy(out_data + (n * out_stack_axis + offset_stack_axis) * stack_size,
           in_data + n * stack_size, stack_size * sizeof(float));
  }
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(StackCPU, StackKernelDefault<DeviceType::kCPU>);

}  // namespace Shadow
