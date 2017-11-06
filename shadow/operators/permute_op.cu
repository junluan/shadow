#include "permute_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelPermute(const T *in_data, int count, int num_axes,
                              const int *permute_order, const int *old_steps,
                              const int *new_steps, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int old_idx = 0;
    int idx = globalid;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    out_data[globalid] = in_data[old_idx];
  }
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  KernelPermute<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_axes, permute_order, old_steps, new_steps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Permute(const float *in_data, int count, int num_axes,
                      const int *permute_order, const int *old_steps,
                      const int *new_steps, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow