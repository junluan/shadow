#include "stack.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelStack(const float* in_data, int count, int num_stacks,
                            int stack_size, int out_stack_axis,
                            int offset_stack_axis, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int stack_num = globalid / stack_size;
    int stack_index = globalid % stack_size;
    int top_index =
        stack_index +
        (stack_num * out_stack_axis + offset_stack_axis) * stack_size;
    out_data[top_index] = in_data[globalid];
  }
}

template <>
void Stack<DeviceType::kGPU, float>(const float* in_data, int count,
                                    int num_stacks, int stack_size,
                                    int out_stack_axis, int offset_stack_axis,
                                    float* out_data, Context* context) {
  KernelStack<<<GetBlocks(count), NumThreads, 0,
                cudaStream_t(context->cuda_stream())>>>(
      in_data, count, num_stacks, stack_size, out_stack_axis, offset_stack_axis,
      out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(StackGPU, StackKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
