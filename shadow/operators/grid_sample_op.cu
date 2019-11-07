#include "grid_sample_op.hpp"

namespace Shadow {

namespace Vision {

#if defined(USE_CUDA)
template <typename T>
__global__ void KernelGridSampleNearest(const T* in_data,
                                        const float* grid_data, int count,
                                        int channel, int in_h, int in_w,
                                        int out_h, int out_w, int padding_mode,
                                        T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int grid_offset = ((b_out * out_h + h_out) * out_w + w_out) * 2;
    float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

    int src_h = static_cast<int>(round((y + 1) / 2.f * (in_h - 1)));
    int src_w = static_cast<int>(round((x + 1) / 2.f * (in_w - 1)));

    if (padding_mode == 1) {
      src_h = min(max(src_h, 0), in_h - 1);
      src_w = min(max(src_w, 0), in_w - 1);
    } else if (padding_mode == 0) {
      if (src_h < 0 || src_w < 0 || src_h > in_h - 1 || src_w > in_w - 1) {
        *out_data++ = T(0);
        continue;
      }
    }

    int src_index = ((b_out * channel + c_out) * in_h + src_h) * in_w + src_w;
    out_data[globalid] = in_data[src_index];
  }
}

template <typename T>
__global__ void KernelGridSampleBilinear(const T* in_data,
                                         const float* grid_data, int count,
                                         int channel, int in_h, int in_w,
                                         int out_h, int out_w, int padding_mode,
                                         T* out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int w_out = globalid % out_w;
    int h_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % channel;
    int b_out = temp / channel;

    int grid_offset = ((b_out * out_h + h_out) * out_w + w_out) * 2;
    float x = grid_data[grid_offset], y = grid_data[grid_offset + 1];

    float src_h_f = (y + 1) / 2.f * (in_h - 1);
    float src_w_f = (x + 1) / 2.f * (in_w - 1);

    if (padding_mode == 1) {
      src_h_f = min(max(src_h_f, 0.f), in_h - 1.f);
      src_w_f = min(max(src_w_f, 0.f), in_w - 1.f);
    } else if (padding_mode == 0) {
      if (src_h_f < 0 || src_w_f < 0 || src_h_f > in_h - 1 ||
          src_w_f > in_w - 1) {
        *out_data++ = T(0);
        continue;
      }
    }

    int src_h_0 = max(static_cast<int>(floor(src_h_f)), 0);
    int src_h_1 = min(static_cast<int>(ceil(src_h_f)), in_h - 1);
    int src_w_0 = max(static_cast<int>(floor(src_w_f)), 0);
    int src_w_1 = min(static_cast<int>(ceil(src_w_f)), in_w - 1);
    float sh = src_h_f - src_h_0, sw = src_w_f - src_w_0;

    int h_offset = (b_out * channel + c_out) * in_h;
    int src_index_0 = (h_offset + src_h_0) * in_w + src_w_0;
    int src_index_1 = (h_offset + src_h_1) * in_w + src_w_0;
    int src_index_2 = (h_offset + src_h_0) * in_w + src_w_1;
    int src_index_3 = (h_offset + src_h_1) * in_w + src_w_1;

    out_data[globalid] = static_cast<T>(
        (1 - sh) * (1 - sw) * in_data[src_index_0] +
        sh * (1 - sw) * in_data[src_index_1] +
        (1 - sh) * sw * in_data[src_index_2] + sh * sw * in_data[src_index_3]);
  }
}

template <typename T>
void GridSample(const T* in_data, const VecInt& in_shape,
                const float* grid_data, int mode, int padding_mode,
                const VecInt& out_shape, T* out_data) {
  int batch = in_shape[0], channel = in_shape[1];
  int in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * channel * out_h * out_w;
  if (mode == 0) {
    KernelGridSampleNearest<T><<<GetBlocks(count), NumThreads>>>(
        in_data, grid_data, count, channel, in_h, in_w, out_h, out_w,
        padding_mode, out_data);
  } else if (mode == 1) {
    KernelGridSampleBilinear<T><<<GetBlocks(count), NumThreads>>>(
        in_data, grid_data, count, channel, in_h, in_w, out_h, out_w,
        padding_mode, out_data);
  } else {
    LOG(FATAL) << "Unsupported grid sample mode: " << mode;
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template void GridSample(const float*, const VecInt&, const float*, int, int,
                         const VecInt&, float*);
#endif

}  // namespace Vision

}  // namespace Shadow
