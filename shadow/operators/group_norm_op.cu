#include "group_norm_op.hpp"

namespace Shadow {

namespace Vision {

template <typename T>
__global__ void KernelComputeGroup(const T *in_data, int count, int channel,
                                   int group, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int g_out = globalid % group;
    int b_out = globalid / group;

    int num_val = channel / group;

    const T *in_data_off = in_data + b_out * channel + g_out * num_val;

    T sum = T(0);
    for (int n = 0; n < num_val; ++n, in_data_off++) {
      sum += *in_data_off;
    }

    out_data[globalid] = sum / num_val;
  }
}

template <typename T>
void ComputeGroup(const T *in_data, int batch, int channel, int group,
                  T *out_data) {
  int count = batch * group;
  KernelComputeGroup<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, channel, group, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelSubtractMean(const T *in_data, const T *mean_data,
                                   int count, int channel, int spatial_dim,
                                   int group, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / spatial_dim;
    int c_out = temp % channel;
    int g_out = c_out / (channel / group);
    int b_out = temp / channel;

    out_data[globalid] = in_data[globalid] - mean_data[b_out * group + g_out];
  }
}

template <typename T>
void SubtractMean(const T *in_data, const T *mean_data, int batch, int channel,
                  int spatial_dim, int group, T *out_data) {
  int count = batch * channel * spatial_dim;
  KernelSubtractMean<T><<<GetBlocks(count), NumThreads>>>(
      in_data, mean_data, count, channel, spatial_dim, group, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelDivideVariance(const T *in_data, const T *variance_data,
                                     int count, int channel, int spatial_dim,
                                     int group, float eps, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / spatial_dim;
    int c_out = temp % channel;
    int g_out = c_out / (channel / group);
    int b_out = temp / channel;

    out_data[globalid] =
        in_data[globalid] / sqrtf(variance_data[b_out * group + g_out] + eps);
  }
}

template <typename T>
void DivideVariance(const T *in_data, const T *variance_data, int batch,
                    int channel, int spatial_dim, int group, float eps,
                    T *out_data) {
  int count = batch * channel * spatial_dim;
  KernelDivideVariance<T>
      <<<GetBlocks(count), NumThreads>>>(in_data, variance_data, count, channel,
                                         spatial_dim, group, eps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void ComputeGroup(const float *, int, int, int, float *);
template void SubtractMean(const float *, const float *, int, int, int, int,
                           float *);
template void DivideVariance(const float *, const float *, int, int, int, int,
                             float, float *);

}  // namespace Vision

}  // namespace Shadow
