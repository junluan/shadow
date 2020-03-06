#include "axpy_op.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
__global__ void KernelAxpy(int count, int spatial_dim, const T *scale_data,
                           const T *x_data, const T *y_data, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] = scale_data[globalid / spatial_dim] * x_data[globalid] +
                         y_data[globalid];
  }
}

template <typename T>
void Axpy(const T *scale_data, const T *x_data, const T *y_data,
          const VecInt &in_shape, T *out_data, Context *context) {
  int spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_shape[1] * spatial_dim;
  KernelAxpy<T><<<GetBlocks(count), NumThreads, 0,
                  cudaStream_t(context->cuda_stream())>>>(
      count, spatial_dim, scale_data, x_data, y_data, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Axpy(const float *, const float *, const float *, const VecInt &,
                   float *, Context *);

}  // namespace Vision

}  // namespace Shadow
