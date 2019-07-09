#include "binary_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__device__ T Binary(T a, T b, int operation) {
  switch (operation) {
    case BinaryOp::kAdd:
      return a + b;
    case BinaryOp::kSub:
      return a - b;
    case BinaryOp::kMul:
      return a * b;
    case BinaryOp::kDiv:
      return a / b;
    case BinaryOp::kPow:
      return powf(a, b);
    case BinaryOp::kMax:
      return fmaxf(a, b);
    case BinaryOp::kMin:
      return fminf(a, b);
    default:
      return 0;
  }
}

template <typename T>
__global__ void KernelBroadcastBinary(const T *in_data, const int *in_shape,
                                      const T *scalar_data,
                                      const int *scalar_shape, int operation,
                                      int num_axes, int count,
                                      const int *out_shape, T *out_data) {
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

template <typename T>
void BroadcastBinary(const T *in_data, const int *in_shape,
                     const T *scalar_data, const int *scalar_shape,
                     int operation, int num_axes, int count,
                     const int *out_shape, T *out_data) {
  KernelBroadcastBinary<T><<<GetBlocks(count), NumThreads>>>(
      in_data, in_shape, scalar_data, scalar_shape, operation, num_axes, count,
      out_shape, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void BroadcastBinary(const float *in_data, const int *in_shape,
                              const float *scalar_data, const int *scalar_shape,
                              int operation, int num_axes, int count,
                              const int *out_shape, float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
