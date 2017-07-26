#include "image.hpp"
#include "kernel.hpp"
#include "util/log.hpp"

#include "math_functions.h"

namespace Shadow {

namespace Image {

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

template <typename T>
__global__ void KernelIm2Col(const T *in_data, int offset, int count, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int dilation, int zero_point, int out_h,
                             int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int h_index = globalid / out_w;
    int h_col = h_index % out_h;
    int w_col = globalid % out_w;
    int c_im = h_index / out_h;
    int c_col = c_im * kernel_size * kernel_size;
    int h_offset = h_col * stride - pad;
    int w_offset = w_col * stride - pad;
    out_data += (c_col * out_h + h_col) * out_w + w_col;
    in_data += offset + (c_im * in_h + h_offset) * in_w + w_offset;
    for (int i = 0; i < kernel_size; ++i) {
      for (int j = 0; j < kernel_size; ++j) {
        int h_im = h_offset + i * dilation;
        int w_im = w_offset + j * dilation;
        *out_data = (h_im >= 0 && w_im >= 0 && h_im < in_h && w_im < in_w)
                        ? in_data[i * dilation * in_w + j * dilation]
                        : static_cast<T>(zero_point);
        out_data += out_h * out_w;
      }
    }
  }
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation, int zero_point,
            const VecInt &out_shape, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = in_c * out_h * out_w;
  KernelIm2Col<T><<<GetBlocks(count), NumThreads>>>(
      in_data, offset, count, in_c, in_h, in_w, kernel_size, stride, pad,
      dilation, zero_point, out_h, out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPooling(const T *in_data, int count, int in_c, int in_h,
                              int in_w, int kernel_size, int stride, int pad,
                              int mode, int out_h, int out_w, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int temp = globalid / out_w;
    int j_out = globalid % out_w;
    int i_out = temp % out_h;
    temp = temp / out_h;
    int c_out = temp % in_c;
    int b_out = temp / in_c;

    int kistart = i_out * stride - pad, kjstart = j_out * stride - pad;
    int kiend = min(kistart + kernel_size, in_h);
    int kjend = min(kjstart + kernel_size, in_w);
    int pool_size = (kiend - kistart) * (kjend - kjstart);
    kistart = max(kistart, 0), kjstart = max(kjstart, 0);
    kiend = min(kiend, in_h), kjend = min(kjend, in_w);

    T max = -FLT_MAX;
    T sum = T(0);
    for (int ki = kistart; ki < kiend; ++ki) {
      for (int kj = kjstart; kj < kjend; ++kj) {
        int index = kj + in_w * (ki + in_h * (c_out + in_c * b_out));
        T value = in_data[index];
        max = (value > max) ? value : max;
        sum += value;
      }
    }
    out_data[globalid] = (mode == 0) ? max : sum / pool_size;
  }
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int count = batch * in_c * out_h * out_w;
  KernelPooling<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, in_c, in_h, in_w, kernel_size, stride, pad, mode, out_h,
      out_w, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelConcat(const T *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int total_concat_size = concat_size * bottom_concat_axis;
    int concat_num = globalid / total_concat_size;
    int concat_index = globalid % total_concat_size;
    int top_index =
        concat_index +
        (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
    out_data[top_index] = in_data[globalid];
  }
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  KernelConcat<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_concats, concat_size, top_concat_axis,
      bottom_concat_axis, offset_concat_axis, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPermute(const T *in_data, int count, int num_axes,
                              const int *permute_order, const int *old_steps,
                              const int *new_steps, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int old_idx = 0;
    int idx = globalid;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    out_data[globalid] = in_data[old_idx];
  }
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  KernelPermute<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, num_axes, permute_order, old_steps, new_steps, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelScale(const T *in_data, int count, const T *scale_data,
                            const T *bias_data, int scale_dim, int inner_dim,
                            T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % scale_dim;
    out_data[globalid] =
        in_data[globalid] * scale_data[index] + bias_data[index];
  }
}

template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data) {
  KernelScale<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, scale_data, bias_data, scale_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelBias(const T *in_data, int count, const T *bias_data,
                           int bias_dim, int inner_dim, T *out_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int index = (globalid / inner_dim) % bias_dim;
    out_data[globalid] = in_data[globalid] + bias_data[index];
  }
}

template <typename T>
void Bias(const T *in_data, int count, const T *bias_data, int bias_dim,
          int inner_dim, T *out_data) {
  KernelBias<T><<<GetBlocks(count), NumThreads>>>(
      in_data, count, bias_data, bias_dim, inner_dim, out_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

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

template <typename T>
__device__ float ActivateValue(T x, int type) {
  switch (type) {
    case 1:
      return x * (x > 0); /*relu*/
    case 2:
      return (x > 0) ? x : T(.1) * x; /*leaky*/
    default:
      return x;
  }
}

template <typename T>
__global__ void KernelActivate(T *data, int count, int type) {
  CUDA_KERNEL_LOOP(globalid, count) {
    data[globalid] = ActivateValue(data[globalid], type);
  }
}

template <typename T>
void Activate(T *data, int count, int type) {
  KernelActivate<T><<<GetBlocks(count), NumThreads>>>(data, count, type);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
__global__ void KernelPRelu(T *data, int count, int channels, int dim,
                            int div_factor, const T *slope_data) {
  CUDA_KERNEL_LOOP(globalid, count) {
    int c = (globalid / dim) % channels / div_factor;
    T value = data[globalid];
    data[globalid] = value > 0 ? value : value * slope_data[c];
  }
}

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data) {
  int channels = in_shape[1], dim = 1;
  for (int i = 2; i < in_shape.size(); ++i) dim *= in_shape[i];
  int count = in_shape[0] * channels * dim;
  int div_factor = channel_shared ? channels : 1;
  KernelPRelu<T><<<GetBlocks(count), NumThreads>>>(data, count, channels, dim,
                                                   div_factor, slope_data);
  CUDA_CHECK(cudaPeekAtLastError());
}

// Explicit instantiation
template void DataTransform(const float *in_data, const VecInt &in_shape,
                            float scale, int num_mean, const float *mean_value,
                            float *out_data);
template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     int zero_point, const VecInt &out_shape, float *out_data);
template void Pooling(const float *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, float *out_data);
template void Concat(const float *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     float *out_data);
template void Permute(const float *in_data, int count, int num_axes,
                      const int *permute_order, const int *old_steps,
                      const int *new_steps, float *out_data);
template void Scale(const float *in_data, int count, const float *scale_data,
                    const float *bias_data, int scale_dim, int inner_dim,
                    float *out_data);
template void Bias(const float *in_data, int count, const float *bias_data,
                   int bias_dim, int inner_dim, float *out_data);
template void Reorg(const float *in_data, const VecInt &in_shape, int stride,
                    float *out_data);
template void LRN(const float *in_data, const VecInt &in_shape, int size,
                  float alpha, float beta, float k, float *scale_data,
                  float *out_data);
template void Activate(float *data, int count, int type);
template void PRelu(float *data, const VecInt &in_shape, bool channel_shared,
                    const float *slope_data);
#endif

}  // namespace Image

}  // namespace Shadow
