#include "reorg_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelReorg(const T *in_data, int count, int in_c, int in_h,
                            int in_w, int out_c, int stride, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w = globalid % in_w;
    int h = temp % in_h;
    temp = temp / in_h;
    int c = temp % in_c;
    int b = temp / in_c;

    int c2 = c % out_c;
    int offset = c / out_c;
    int h2 = h * stride + offset / stride;
    int w2 = w * stride + offset % stride;
    int in_index = ((b * in_c + c) * in_h + h) * in_w + w;
    int out_index =
        ((b * out_c + c2) * in_h * stride + h2) * in_w * stride + w2;
    out_data[in_index] = in_data[out_index];
  }
}

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c / (stride * stride);
  int count = batch * in_c * in_h * in_w;
  KernelReorg<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, out_c, stride, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow
