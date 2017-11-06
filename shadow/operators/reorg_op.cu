#include "reorg_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelReorg(const T *in_data, int count, int in_c, int in_h,
                            int in_w, int out_c, int out_h, int out_w,
                            int stride, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w = globalid % out_w;
    int h = temp % out_h;
    temp = temp / out_h;
    int c = temp % out_c;
    int b = temp / out_c;

    int c_in = c % in_c;
    int area = c / in_c;
    int h_in = h * stride + area / stride;
    int w_in = w * stride + area % stride;
    int in_index = ((b * in_c + c_in) * in_h + h_in) * in_w + w_in;
    int out_index = ((b * out_c + c) * out_h + h) * out_w + w;
    out_data[out_index] = in_data[in_index];
  }
}

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_c = in_c * stride * stride;
  int out_h = in_h / stride, out_w = in_w / stride;
  int count = batch * out_c * out_h * out_w;
  KernelReorg<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, out_c, out_h, out_w, stride, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);
#endif

}  // namespace Vision

}  // namespace Shadow