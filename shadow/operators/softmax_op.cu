#include "softmax_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelChannelMax(const T *in_data, int val_count, int channels,
                                 int inner_num, T *val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const T *in_data_offset = in_data + n * channels * inner_num + s;
    T max_val = T(-FLT_MAX);
    for (int c = 0; c < channels; ++c, in_data_offset += inner_num) {
      max_val = fmaxf(*in_data_offset, max_val);
    }
    val_data[globalid] = max_val;
  }
}

template <typename T>
__global__ void KernelChannelSubExp(const T *in_data, const T *val_data,
                                    int count, int channels, int inner_num,
                                    T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / channels / inner_num, s = globalid % inner_num;
    out_data[globalid] = expf(in_data[globalid] - val_data[n * inner_num + s]);
  }
}

template <typename T>
__global__ void KernelChannelSum(const T *out_data, int val_count, int channels,
                                 int inner_num, T *val_data) {
  CUDA_KERNEL_LOOP(globalid, val_count) {
    int n = globalid / inner_num, s = globalid % inner_num;
    const T *out_data_offset = out_data + n * channels * inner_num + s;
    T sum = T(0);
    for (int c = 0; c < channels; ++c, out_data_offset += inner_num) {
      sum += *out_data_offset;
    }
    val_data[globalid] = sum;
  }
}

template <typename T>
__global__ void KernelChannelDiv(const T *val_data, int count, int channels,
                                 int inner_num, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int n = globalid / channels / inner_num, s = globalid % inner_num;
    out_data[globalid] /= val_data[n * inner_num + s];
  }
}

template <typename T>
void Softmax(const T *in_data, int outer_num, int channels, int inner_num,
             T *val_data, T *out_data) {
  int val_count = outer_num * inner_num, count = val_count * channels;
  KernelChannelMax<T><<<GetBlocks(val_count), NumThreads>>>(
      in_data, val_count, channels, inner_num, val_data);
  KernelChannelSubExp<T><<<GetBlocks(count), NumThreads>>>(
      in_data, val_data, count, channels, inner_num, out_data);
  KernelChannelSum<T><<<GetBlocks(val_count), NumThreads>>>(
      out_data, val_count, channels, inner_num, val_data);
  KernelChannelDiv<T><<<GetBlocks(count), NumThreads>>>(
      val_data, count, channels, inner_num, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Softmax(const float *, int, int, int, float *, float *);

#endif

}  // namespace Vision

}  // namespace Shadow
