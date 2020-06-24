#include "gather.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelGather(const float* in_data, const int* indexes_data,
                             int num_indexes, int gather_dim, int inner_num,
                             int count, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int gather_num = num_indexes * inner_num;
    int gather_index = indexes_data[(globalid / inner_num) % num_indexes];
    int in_index =
        (gather_index + globalid / gather_num * gather_dim) * inner_num +
        globalid % inner_num;
    out_data[globalid] = in_data[in_index];
  }
}

template <>
void Gather<DeviceType::kGPU, float>(const float* in_data,
                                     const int* indexes_data, int num_indexes,
                                     int gather_dim, int inner_num, int count,
                                     float* out_data, Context* context) {
  KernelGather<<<GetBlocks(count), NumThreads, 0,
                 cudaStream_t(context->cuda_stream())>>>(
      in_data, indexes_data, num_indexes, gather_dim, inner_num, count,
      out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(GatherGPU, GatherKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
