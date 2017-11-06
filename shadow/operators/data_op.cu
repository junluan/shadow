#include "data_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelDataTransform(const T *in_data, int count, int in_c,
                                    int spatial_dim, float scale, int num_mean,
                                    const T *mean_value, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c_out = (globalid / spatial_dim) % in_c;
    int s_out = globalid % spatial_dim;

    if (num_mean == 1) {
      out_data[globalid] = (in_data[globalid] - mean_value[0]) * scale;
    } else if (num_mean == in_c) {
      out_data[globalid] = (in_data[globalid] - mean_value[c_out]) * scale;
    } else if (num_mean == in_c * spatial_dim) {
      out_data[globalid] =
          (in_data[globalid] - mean_value[c_out * spatial_dim + s_out]) * scale;
    }
  }
}

template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;
  KernelDataTransform<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, spatial_dim, scale, num_mean, mean_value, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void DataTransform(const float *in_data, const VecInt &in_shape,
                            float scale, int num_mean, const float *mean_value,
                            float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow