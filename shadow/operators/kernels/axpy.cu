#include "axpy.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelAxpy(int count, int inner_num, const float* alpha_data,
                           const float* x_data, const float* y_data,
                           float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        alpha_data[globalid / inner_num] * x_data[globalid] + y_data[globalid];
  }
}

template <>
void Axpy<DeviceType::kGPU, float>(const float* alpha_data, const float* x_data,
                                   const float* y_data, int outer_num,
                                   int inner_num, float* out_data,
                                   Context* context) {
  int count = outer_num * inner_num;
  KernelAxpy<<<GetBlocks(count), NumThreads, 0,
               cudaStream_t(context->cuda_stream())>>>(
      count, inner_num, alpha_data, x_data, y_data, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(AxpyGPU, AxpyKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
