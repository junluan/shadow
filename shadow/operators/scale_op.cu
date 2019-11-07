#include "scale_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelScale(const T *in_data, int count, const T *scale_data,
                            const T *bias_data, int scale_dim, int inner_dim,
                            T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % scale_dim;
    out_data[globalid] =
        in_data[globalid] * scale_data[index] + bias_data[index];
  }
}

template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  KernelScale<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, scale_data, bias_data, scale_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Scale(const float *, int, const float *, const float *, int, int,
                    float *);
#endif

}  // namespace Vision

}  // namespace Shadow
