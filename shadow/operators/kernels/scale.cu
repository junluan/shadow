#include "scale.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelScaleBias(const float* in_data, int count,
                                const float* scale_data, const float* bias_data,
                                int scale_num, int inner_num, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_num) % scale_num;
    out_data[globalid] =
        in_data[globalid] * scale_data[index] + bias_data[index];
  }
}

template <>
void ScaleBias<DeviceType::kGPU, float>(const float* in_data, int count,
                                        const float* scale_data,
                                        const float* bias_data, int scale_num,
                                        int inner_num, float* out_data,
                                        Context* context) {
  KernelScaleBias<<<GetBlocks(count), NumThreads, 0,
                    cudaStream_t(context->stream())>>>(
      in_data, count, scale_data, bias_data, scale_num, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

__global__ void KernelScale(const float* in_data, int count,
                            const float* scale_data, int scale_num,
                            int inner_num, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_num) % scale_num;
    out_data[globalid] = in_data[globalid] * scale_data[index];
  }
}

template <>
void Scale<DeviceType::kGPU, float>(const float* in_data, int count,
                                    const float* scale_data, int scale_num,
                                    int inner_num, float* out_data,
                                    Context* context) {
  KernelScale<<<GetBlocks(count), NumThreads, 0,
                cudaStream_t(context->stream())>>>(
      in_data, count, scale_data, scale_num, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelBias(const T* in_data, int count, const T* bias_data,
                           int scale_num, int inner_num, T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_num) % scale_num;
    out_data[globalid] = in_data[globalid] + bias_data[index];
  }
}

template <>
void Bias<DeviceType::kGPU, float>(const float* in_data, int count,
                                   const float* bias_data, int scale_num,
                                   int inner_num, float* out_data,
                                   Context* context) {
  KernelBias<<<GetBlocks(count), NumThreads, 0,
               cudaStream_t(context->stream())>>>(
      in_data, count, bias_data, scale_num, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ScaleGPU, ScaleKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
