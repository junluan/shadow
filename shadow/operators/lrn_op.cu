#include "lrn_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelLRNFillScale(const T *in_data, int count, int in_c,
                                   int in_h, int in_w, int size,
                                   float alpha_over_size, float k,
                                   T *scale_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w = globalid % in_w;
    int h = temp % in_h;
    int b = temp / in_h;

    int offset = (b * in_c * in_h + h) * in_w + w, head = 0;
    const T *in_off = in_data + offset;
    T *scale_off = scale_data + offset;
    T accum_scale = T(0);
    int step = in_h * in_w;
    int pre_pad = (size - 1) / 2, post_pad = size - pre_pad - 1;
    while (head < post_pad && head < in_c) {
      accum_scale += in_off[head * step] * in_off[head * step];
      head++;
    }
    while (head < in_c) {
      accum_scale += in_off[head * step] * in_off[head * step];
      if (head - size >= 0) {
        accum_scale -=
            in_off[(head - size) * step] * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      head++;
    }
    while (head < in_c + post_pad) {
      if (head - size >= 0) {
        accum_scale -=
            in_off[(head - size) * step] * in_off[(head - size) * step];
      }
      scale_off[(head - post_pad) * step] = k + accum_scale * alpha_over_size;
      head++;
    }
  }
}

template <typename T>
__global__ void KernelLRN(const T *in_data, int count, const T *scale_data,
                          float negative_beta, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        in_data[globalid] * pow(scale_data[globalid], negative_beta);
  }
}

template <typename T>
void LRN(const T *in_data, const VecInt &in_shape, int size, float alpha,
         float beta, float k, T *scale_data, T *out_data) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_h * in_w;
  KernelLRNFillScale<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, size, alpha / size, k, scale_data);
  CUDA_CHECK(cudaPeekAtLastError());
  count *= in_c;
  KernelLRN<T><<<GetBlocks(count), NumThreads>>>(in_data, count, scale_data,
                                                 -beta, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template void LRN(const float *, const VecInt &, int, float, float, float,
                  float *, float *);
#endif

}  // namespace Vision

}  // namespace Shadow
