#include "normalize.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelLpNorm(const float* in_data, int val_count, int dim,
                             int inner_num, float p, float* val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const auto* in_data_offset = in_data + n * dim * inner_num + s;
    double val = 0;
    if (p == 1) {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        val += fabsf(*in_data_offset);
      }
      val_data[globalid] = static_cast<float>(val);
    } else if (p == 2) {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        auto abs_data = fabsf(*in_data_offset);
        val += abs_data * abs_data;
      }
      val_data[globalid] = sqrtf(static_cast<float>(val));
    } else {
      for (int c = 0; c < dim; ++c, in_data_offset += inner_num) {
        val += powf(fabsf(*in_data_offset), p);
      }
      val_data[globalid] = powf(static_cast<float>(val), 1.f / p);
    }
  }
}

__global__ void KernelDivLpNorm(const float* in_data, const float* val_data,
                                int count, int dim, int inner_num, float eps,
                                float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / dim / inner_num, s = globalid % inner_num;
    out_data[globalid] =
        in_data[globalid] / fmaxf(val_data[n * inner_num + s], eps);
  }
}

template <>
void Normalize<DeviceType::kGPU, float>(const float* in_data, int outer_num,
                                        int dim, int inner_num, float* val_data,
                                        float p, float eps, float* out_data,
                                        Context* context) {
  int val_count = outer_num * inner_num, count = val_count * dim;
  KernelLpNorm<<<GetBlocks(val_count), NumThreads, 0,
                 cudaStream_t(context->cuda_stream())>>>(
      in_data, val_count, dim, inner_num, p, val_data);
  KernelDivLpNorm<<<GetBlocks(count), NumThreads, 0,
                    cudaStream_t(context->cuda_stream())>>>(
      in_data, val_data, count, dim, inner_num, eps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(NormalizeGPU,
                           NormalizeKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
