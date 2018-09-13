#include "stack_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelStack(const T *in_data, int count, int num_stacks,
                            int stack_size, int top_stack_axis,
                            int offset_stack_axis, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int stack_num = globalid / stack_size;
    int stack_index = globalid % stack_size;
    int top_index =
        stack_index +
        (stack_num * top_stack_axis + offset_stack_axis) * stack_size;
    out_data[top_index] = in_data[globalid];
  }
}

template <typename T>
void Stack(const T *in_data, int count, int num_stacks, int stack_size,
           int top_stack_axis, int offset_stack_axis, T *out_data) {
  KernelStack<T><<<GetBlocks(count), NumThreads>>>(in_data, count, num_stacks,
                                                   stack_size, top_stack_axis,
                                                   offset_stack_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Stack(const float *in_data, int count, int num_stacks,
                    int stack_size, int top_stack_axis, int offset_concat_axis,
                    float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
