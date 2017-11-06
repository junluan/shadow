#include "bias_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelBias(const T *in_data, int count, const T *bias_data,
                           int bias_dim, int inner_dim, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % bias_dim;
    out_data[globalid] = in_data[globalid] + bias_data[index];
  }
}

template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data) {
  KernelBias<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, bias_data, bias_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Bias(const float *in_data, int count, const float *bias_data,
                   int bias_dim, int inner_dim, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow