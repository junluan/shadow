#include "activate_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__device__ float ActivateValue(T x, int type, float slope) {
  // PRelu: 0, Relu: 1, Leaky: 2, Sigmoid: 3, SoftPlus: 4, Tanh: 5
  switch (type) {
    case 1:
      return x * (x > 0);
    case 2:
      return x > 0 ? x : T(slope * x);
    case 3:
      return 1 / (1 + expf(-x));
    case 4:
      return logf(1 + expf(x));
    case 5: {
      T exp_2x = expf(2 * x);
      return (exp_2x - 1) / (exp_2x + 1);
    }
    default:
      return x;
  }
}

template <typename T>
__global__ void KernelActivate(T *data, int count, int type, float slope) {
  CUDA_KERNEL_LOOP(globalid, count) {
    data[globalid] = ActivateValue(data[globalid], type, slope);
  }
}

template <typename T>
void Activate(T *data, int count, int type, float slope) {
  KernelActivate<T><<<GetBlocks(count), NumThreads>>>(data, count, type, slope);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPRelu(T *data, int count, int channels, int dim,
                            int div_factor, const T *slope_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c = (globalid / dim) % channels / div_factor;
    T value = data[globalid];
    data[globalid] = value > 0 ? value : value * slope_data[c];
  }
}

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  KernelPRelu<T><<<GetBlocks(count), NumThreads>>>(data, count, channels, dim,
                                                   div_factor, slope_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Activate(float *data, int count, int type, float slope);
template void PRelu(float *data, const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);
#endif

}  // namespace Vision

}  // namespace Shadow
