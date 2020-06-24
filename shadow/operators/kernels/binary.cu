#include "binary.hpp"

namespace Shadow {

namespace Vision {

__device__ float Binary(float a, float b, int operation) {
  switch (operation) {
    case kAdd:
      return a + b;
    case kSub:
      return a - b;
    case kMul:
      return a * b;
    case kDiv:
      return a / b;
    case kPow:
      return powf(a, b);
    case kMax:
      return fmaxf(a, b);
    case kMin:
      return fminf(a, b);
    default:
      return 0;
  }
}

__global__ void KernelBroadcastBinary(const float* in_data, const int* in_shape,
                                      const float* scalar_data,
                                      const int* scalar_shape, int operation,
                                      int num_axes, int count,
                                      const int* out_shape, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int in_shape_acc[8], scalar_shape_acc[8];
    in_shape_acc[num_axes - 1] = 1, scalar_shape_acc[num_axes - 1] = 1;
    for (int n = num_axes - 1; n > 0; --n) {
      in_shape_acc[n - 1] = in_shape[n] * in_shape_acc[n];
      scalar_shape_acc[n - 1] = scalar_shape[n] * scalar_shape_acc[n];
    }

    int in_index = 0, scalar_index = 0, cc = globalid;
    for (int n = num_axes - 1; n >= 0; --n) {
      int dim = cc % out_shape[n];
      in_index += (dim % in_shape[n]) * in_shape_acc[n];
      scalar_index += (dim % scalar_shape[n]) * scalar_shape_acc[n];
      cc /= out_shape[n];
    }

    out_data[globalid] =
        Binary(in_data[in_index], scalar_data[scalar_index], operation);
  }
}

template <>
void BroadcastBinary<DeviceType::kGPU, float>(
    const float* in_data, const int* in_shape, const float* scalar_data,
    const int* scalar_shape, int operation, int num_axes, int count,
    const int* out_shape, float* out_data, Context* context) {
  KernelBroadcastBinary<<<GetBlocks(count), NumThreads, 0,
                          cudaStream_t(context->cuda_stream())>>>(
      in_data, in_shape, scalar_data, scalar_shape, operation, num_axes, count,
      out_shape, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(BinaryGPU, BinaryKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
