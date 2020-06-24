#include "permute.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelPermute(const float* in_data, int count, int num_axes,
                              const int* order, const int* old_steps,
                              const int* new_steps, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int old_idx = 0;
    int idx = globalid;
    for (int j = 0; j < num_axes; ++j) {
      old_idx += (idx / new_steps[j]) * old_steps[order[j]];
      idx %= new_steps[j];
    }
    out_data[globalid] = in_data[old_idx];
  }
}

template <>
void Permute<DeviceType::kGPU, float>(const float* in_data, int count,
                                      int num_axes, const int* order,
                                      const int* old_steps,
                                      const int* new_steps, float* out_data,
                                      Context* context) {
  KernelPermute<<<GetBlocks(count), NumThreads, 0,
                  cudaStream_t(context->cuda_stream())>>>(
      in_data, count, num_axes, order, old_steps, new_steps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(PermuteGPU, PermuteKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
