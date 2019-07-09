#include "activate_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__device__ T ActivateValue(T x, int type, float slope) {
  switch (type) {
    case ActivateOp::kRelu:
      return x > 0 ? x : 0;
    case ActivateOp::kLeaky:
      return x > 0 ? x : T(slope * x);
    case ActivateOp::kSigmoid:
      return 1 / (1 + expf(-x));
    case ActivateOp::kSoftPlus:
      return logf(1 + expf(x));
    case ActivateOp::kTanh: {
      T exp_2x = expf(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    default:
      return x;
  }
}

template <typename T>
__global__ void KernelActivate(const T *in_data, T *out_data, int count,
                               int type, float slope) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] = ActivateValue(in_data[globalid], type, slope);
  }
}

template <typename T>
void Activate(const T *in_data, T *out_data, int count, int type, float slope) {
  KernelActivate<T>
      <<<GetBlocks(count), NumThreads>>>(in_data, out_data, count, type, slope);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPRelu(const T *in_data, T *out_data, int count,
                            int channels, int dim, int div_factor,
                            const T *slope_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c = (globalid / dim) % channels / div_factor;
    T value = in_data[globalid];
    out_data[globalid] = value > 0 ? value : value * slope_data[c];
  }
}

template <typename T>
void PRelu(const T *in_data, T *out_data, const VecInt &in_shape,
           bool channel_shared, const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  KernelPRelu<T><<<GetBlocks(count), NumThreads>>>(
      in_data, out_data, count, channels, dim, div_factor, slope_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Activate(const float *in_data, float *out_data, int count,
                       int type, float slope);
template void PRelu(const float *in_data, float *out_data,
                    const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);
#endif

}  // namespace Vision

}  // namespace Shadow
