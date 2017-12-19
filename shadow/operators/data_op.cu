#include "data_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelDataTransform(const T *in_data, int count, int in_c,
                                    int spatial_dim, int num_mean,
                                    const T *mean_value, int num_scale,
                                    const T *scale_value, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c_out = (globalid / spatial_dim) % in_c;

    if (num_mean == 1 && num_scale == 1) {
      out_data[globalid] = (in_data[globalid] - mean_value[0]) * scale_value[0];
    } else if (num_mean == in_c && num_scale == 1) {
      out_data[globalid] =
          (in_data[globalid] - mean_value[c_out]) * scale_value[0];
    } else if (num_mean == 1 && num_scale == in_c) {
      out_data[globalid] =
          (in_data[globalid] - mean_value[0]) * scale_value[c_out];
    } else if (num_mean == in_c && num_scale == in_c) {
      out_data[globalid] =
          (in_data[globalid] - mean_value[c_out]) * scale_value[c_out];
    }
  }
}

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, int num_mean,
                   const T *mean_value, int num_scale, const T *scale_value,
                   T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;
  KernelDataTransform<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, spatial_dim, num_mean, mean_value, num_scale,
      scale_value, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void DataTransform(const float *in_data, const VecInt &in_shape,
                            int num_mean, const float *mean_value,
                            int num_scale, const float *scale_value,
                            float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
