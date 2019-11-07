#include "resize_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelResizeNearest(const T* in_data, int count, int channel,
                                    int in_h, int in_w, int out_h, int out_w,
                                    T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    float fh = static_cast<float>(in_h) / out_h;
    float fw = static_cast<float>(in_w) / out_w;
    int src_h = static_cast<int>(h_out * fh);
    int src_w = static_cast<int>(w_out * fw);

    int src_index = ((b_out * channel + c_out) * in_h + src_h) * in_w + src_w;

    out_data[globalid] = in_data[src_index];
  }
}

template <typename T>
__global__ void KernelResizeBilinear(const T* in_data, int count, int channel,
                                     int in_h, int in_w, int out_h, int out_w,
                                     T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    float fh = static_cast<float>(in_h) / out_h;
    float fw = static_cast<float>(in_w) / out_w;

    float src_h_f = (h_out + 0.5f) * fh - 0.5f;
    int src_h = static_cast<int>(src_h_f);
    float sh = src_h_f - src_h;
    src_h = src_h < in_h - 1 ? src_h : in_h - 2;
    src_h = src_h < 0 ? 0 : src_h;

    float src_w_f = (w_out + 0.5f) * fw - 0.5f;
    int src_w = static_cast<int>(src_w_f);
    float sw = src_w_f - src_w;
    src_w = src_w < in_w - 1 ? src_w : in_w - 2;
    src_w = src_w < 0 ? 0 : src_w;

    int src_h_off = (b_out * channel + c_out) * in_h + src_h;

    int src_index_0 = src_h_off * in_w + src_w;
    int src_index_1 = (src_h_off + 1) * in_w + src_w;
    int src_index_2 = src_h_off * in_w + src_w + 1;
    int src_index_3 = (src_h_off + 1) * in_w + src_w + 1;

    out_data[globalid] = static_cast<T>(
        (1 - sh) * (1 - sw) * in_data[src_index_0] +
        sh * (1 - sw) * in_data[src_index_1] +
        (1 - sh) * sw * in_data[src_index_2] + sh * sw * in_data[src_index_3]);
  }
}

template <typename T>
void Resize(const T* in_data, const VecInt& in_shape, int type,
            const VecInt& out_shape, T* out_data) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * channel * out_h * out_w;
  if (type == 0) {
    KernelResizeNearest<T><<<GetBlocks(count), NumThreads>>>(
        in_data, count, channel, in_h, in_w, out_h, out_w, out_data);
  } else if (type == 1) {
    KernelResizeBilinear<T><<<GetBlocks(count), NumThreads>>>(
        in_data, count, channel, in_h, in_w, out_h, out_w, out_data);
  } else {
    LOG(FATAL) << "Unsupported resize type: " << type;
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template void Resize(const float*, const VecInt&, int, const VecInt&, float*);
#endif

}  // namespace Vision

}  // namespace Shadow
