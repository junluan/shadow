#include "resize.hpp"

namespace Shadow {

namespace Vision {

__global__ void KernelResizeNearest2D(const float* in_data, int count, int in_h,
                                      int in_w, int out_h, int out_w,
                                      float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    int outer_num = temp / out_h;

    float fh = static_cast<float>(in_h) / out_h;
    float fw = static_cast<float>(in_w) / out_w;
    int src_h = static_cast<int>(h_out * fh);
    int src_w = static_cast<int>(w_out * fw);

    int src_index = (outer_num * in_h + src_h) * in_w + src_w;

    out_data[globalid] = in_data[src_index];
  }
}

__global__ void KernelResizeBilinear2D(const float* in_data, int count,
                                       int in_h, int in_w, int out_h, int out_w,
                                       bool align_corners, float* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    int outer_num = temp / out_h;

    float fh = align_corners ? static_cast<float>(in_h - 1) / (out_h - 1)
                             : static_cast<float>(in_h) / out_h;
    float fw = align_corners ? static_cast<float>(in_w - 1) / (out_w - 1)
                             : static_cast<float>(in_w) / out_w;

    float src_h_f, src_w_f;
    if (align_corners) {
      src_h_f = h_out * fh;
      src_w_f = w_out * fw;
    } else {
      src_h_f = (h_out + 0.5f) * fh - 0.5f;
      src_h_f = src_h_f < 0 ? 0 : src_h_f;
      src_w_f = (w_out + 0.5f) * fw - 0.5f;
      src_w_f = src_w_f < 0 ? 0 : src_w_f;
    }

    int src_h_l = static_cast<int>(src_h_f), src_h_h = src_h_l + 1;
    if (src_h_l >= in_h - 1) {
      src_h_h = src_h_l = in_h - 1;
    }
    float sh = src_h_f - src_h_l;

    int src_w_l = static_cast<int>(src_w_f), src_w_h = src_w_l + 1;
    if (src_w_l >= in_w - 1) {
      src_w_h = src_w_l = in_w - 1;
    }
    float sw = src_w_f - src_w_l;

    int src_index_off = outer_num * in_h;

    int src_index_0 = (src_index_off + src_h_l) * in_w + src_w_l;
    int src_index_1 = (src_index_off + src_h_h) * in_w + src_w_l;
    int src_index_2 = (src_index_off + src_h_l) * in_w + src_w_h;
    int src_index_3 = (src_index_off + src_h_h) * in_w + src_w_h;

    out_data[globalid] = (1 - sh) * (1 - sw) * in_data[src_index_0] +
                         sh * (1 - sw) * in_data[src_index_1] +
                         (1 - sh) * sw * in_data[src_index_2] +
                         sh * sw * in_data[src_index_3];
  }
}

template <>
void ResizeNearest2D<DeviceType::kGPU, float>(const float* in_data,
                                              const VecInt& in_shape,
                                              const VecInt& out_shape,
                                              float* out_data,
                                              Context* context) {
  int outer_num = in_shape[0] * in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = outer_num * out_h * out_w;
  KernelResizeNearest2D<<<GetBlocks(count), NumThreads, 0,
                          cudaStream_t(context->stream())>>>(
      in_data, count, in_h, in_w, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <>
void ResizeBilinear2D<DeviceType::kGPU, float>(
    const float* in_data, const VecInt& in_shape, bool align_corners,
    const VecInt& out_shape, float* out_data, Context* context) {
  int outer_num = in_shape[0] * in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = outer_num * out_h * out_w;
  KernelResizeBilinear2D<<<GetBlocks(count), NumThreads, 0,
                           cudaStream_t(context->stream())>>>(
      in_data, count, in_h, in_w, out_h, out_w, align_corners, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

REGISTER_OP_KERNEL_DEFAULT(ResizeGPU, ResizeKernelDefault<DeviceType::kGPU>);

}  // namespace Shadow
