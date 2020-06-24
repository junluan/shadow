#include "lrn.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelLRNFillScale(const float* in_data, int count, int in_c,
                                   int in_h, int in_w, int size,
                                   float alpha_over_size, float k,
                                   float* scale_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / in_w;
    int w = globalid % in_w;
    int h = temp % in_h;
    int b = temp / in_h;

    int offset = (b * in_c * in_h + h) * in_w + w, head = 0;
    const auto* in_off = in_data + offset;
    auto* scale_off = scale_data + offset;
    auto accum_scale = 0.f;
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

__global__ void KernelLRN(const float* in_data, int count,
                          const float* scale_data, float negative_beta,
                          float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    out_data[globalid] =
        in_data[globalid] * pow(scale_data[globalid], negative_beta);
  }
}

template <>
void LRN<DeviceType::kGPU, float>(const float* in_data, const VecInt& in_shape,
                                  int size, float alpha, float beta, float k,
                                  float* scale_data, float* out_data,
                                  Context* context) {
  int batch = in_shape[0], in_c = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int count = batch * in_h * in_w;
  KernelLRNFillScale<<<GetBlocks(count), NumThreads, 0,
                       cudaStream_t(context->cuda_stream())>>>(
      in_data, count, in_c, in_h, in_w, size, alpha / size, k, scale_data);
  CUDA_CHECK(cudaPeekAtLastError());
  count *= in_c;
  KernelLRN<<<GetBlocks(count), NumThreads, 0,
              cudaStream_t(context->cuda_stream())>>>(
      in_data, count, scale_data, -beta, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(LRNGPU, LRNKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
