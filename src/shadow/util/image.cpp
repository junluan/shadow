#include "shadow/util/image.hpp"

#include <cfloat>
#include <cstring>

namespace Image {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void DataTransform(const T *in_data, int count, float scale, float mean_value,
                   T *out_data) {
  for (int i = 0; i < count; ++i) {
    out_data[i] = (in_data[i] - mean_value) * scale;
  }
}

template <typename T>
inline T Im2ColGetPixel(const T *image, int in_h, int in_w, int im_row,
                        int im_col, int channel, int pad) {
  im_row -= pad;
  im_col -= pad;
  if (im_row < 0 || im_col < 0 || im_row >= in_h || im_col >= in_w) return (T)0;
  return image[im_col + in_w * (im_row + in_h * channel)];
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, const VecInt &out_shape,
            T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  const T *im_data_offset = in_data + offset;
  int kernel_num_ = in_c * kernel_size * kernel_size;
  for (int c = 0; c < kernel_num_; ++c) {
    int w_offset = c % kernel_size;
    int h_offset = (c / kernel_size) % kernel_size;
    int c_im = c / kernel_size / kernel_size;
    for (int h = 0; h < out_h; ++h) {
      for (int w = 0; w < out_w; ++w) {
        int im_row = h_offset + h * stride;
        int im_col = w_offset + w * stride;
        int col_index = (c * out_h + h) * out_w + w;
        out_data[col_index] = Im2ColGetPixel(im_data_offset, in_h, in_w, im_row,
                                             im_col, c_im, pad);
      }
    }
  }
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int kistart = h * stride - pad, kjstart = w * stride - pad;
          int kiend = std::min(kistart + kernel_size, in_h);
          int kjend = std::min(kjstart + kernel_size, in_w);
          int pool_size = (kiend - kistart) * (kjend - kjstart);
          kistart = std::max(kistart, 0), kjstart = std::max(kjstart, 0);
          kiend = std::min(kiend, in_h), kjend = std::min(kjend, in_w);
          float max = -FLT_MAX;
          float sum = 0.f;
          for (int ki = kistart; ki < kiend; ++ki) {
            for (int kj = kjstart; kj < kjend; ++kj) {
              int index = kj + in_w * (ki + in_h * (c + in_c * b));
              float value = in_data[index];
              max = (value > max) ? value : max;
              sum += value;
            }
          }
          int out_index = w + out_w * (h + out_h * (c + in_c * b));
          if (mode == 0) {
            out_data[out_index] = max;
          } else {
            out_data[out_index] = sum / pool_size;
          }
        }
      }
    }
  }
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  for (int n = 0; n < num_concats; ++n) {
    memcpy(out_data + (n * top_concat_axis + offset_concat_axis) * concat_size,
           in_data + n * bottom_concat_axis * concat_size,
           bottom_concat_axis * concat_size * sizeof(T));
  }
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  for (int i = 0; i < count; ++i) {
    int old_idx = 0;
    int idx = i;
    for (int j = 0; j < num_axes; ++j) {
      int order = permute_order[j];
      old_idx += (idx / new_steps[j]) * old_steps[order];
      idx %= new_steps[j];
    }
    out_data[i] = in_data[old_idx];
  }
}

template <typename T>
inline T Activate(T x, int type) {
  switch (type) {
    case 0:
      return x;
    case 1:
      return x * (x > 0);
    case 2:
      return (x > 0) ? x : T(.1) * x;
    default:
      return x;
  }
}

template <typename T>
void Activate(T *data, int count, int type) {
  for (int i = 0; i < count; ++i) {
    data[i] = Activate(data[i], type);
  }
}

// Explicit instantiation
template void DataTransform<float>(const float *in_data, int count, float scale,
                                   float mean_value, float *out_data);
template void Im2Col<float>(const float *in_data, const VecInt &in_shape,
                            int offset, int kernel_size, int stride, int pad,
                            const VecInt &out_shape, float *out_data);
template void Pooling<float>(const float *in_data, const VecInt &in_shape,
                             int kernel_size, int stride, int pad, int mode,
                             const VecInt &out_shape, float *out_data);
template void Concat<float>(const float *in_data, int count, int num_concats,
                            int concat_size, int top_concat_axis,
                            int bottom_concat_axis, int offset_concat_axis,
                            float *out_data);
template void Permute<float, int>(const float *in_data, int count, int num_axes,
                                  const int *permute_order,
                                  const int *old_steps, const int *new_steps,
                                  float *out_data);
template void Activate<float>(float *data, int count, int type);

#else
template <typename T>
void DataTransform(const T *in_data, int count, float scale, float mean_value,
                   T *out_data) {
  Kernel::DataTransform(in_data, count, scale, mean_value, out_data);
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, const VecInt &out_shape,
            T *out_data) {
  Kernel::Im2Col(in_data, offset, in_shape[1], in_shape[2], in_shape[3],
                 kernel_size, stride, pad, out_shape[2], out_shape[3],
                 out_data);
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  Kernel::Pooling(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                  kernel_size, stride, pad, mode, out_shape[2], out_shape[3],
                  out_data);
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  Kernel::Concat(in_data, count, num_concats, concat_size, top_concat_axis,
                 bottom_concat_axis, offset_concat_axis, out_data);
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  Kernel::Permute(in_data, count, num_axes, permute_order, old_steps, new_steps,
                  out_data);
}

template <typename T>
void Activate(T *data, int count, int type) {
  Kernel::Activate(data, count, type);
}

#if defined(USE_CUDA)
// Explicit instantiation
template void DataTransform<float>(const float *in_data, int count, float scale,
                                   float mean_value, float *out_data);
template void Im2Col<float>(const float *in_data, const VecInt &in_shape,
                            int offset, int kernel_size, int stride, int pad,
                            const VecInt &out_shape, float *out_data);
template void Pooling<float>(const float *in_data, const VecInt &in_shape,
                             int kernel_size, int stride, int pad, int mode,
                             const VecInt &out_shape, float *out_data);
template void Concat<float>(const float *in_data, int count, int num_concats,
                            int concat_size, int top_concat_axis,
                            int bottom_concat_axis, int offset_concat_axis,
                            float *out_data);
template void Permute<float, int>(const float *in_data, int count, int num_axes,
                                  const int *permute_order,
                                  const int *old_steps, const int *new_steps,
                                  float *out_data);
template void Activate<float>(float *data, int count, int type);

#else
// Explicit instantiation
template void DataTransform<cl_mem>(const cl_mem *in_data, int count,
                                    float scale, float mean_value,
                                    cl_mem *out_data);
template void Im2Col<cl_mem>(const cl_mem *in_data, const VecInt &in_shape,
                             int offset, int kernel_size, int stride, int pad,
                             const VecInt &out_shape, cl_mem *out_data);
template void Pooling<cl_mem>(const cl_mem *in_data, const VecInt &in_shape,
                              int kernel_size, int stride, int pad, int mode,
                              const VecInt &out_shape, cl_mem *out_data);
template void Concat<cl_mem>(const cl_mem *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             cl_mem *out_data);
template void Permute<cl_mem, cl_mem>(const cl_mem *in_data, int count,
                                      int num_axes, const cl_mem *permute_order,
                                      const cl_mem *old_steps,
                                      const cl_mem *new_steps,
                                      cl_mem *out_data);
template void Activate<cl_mem>(cl_mem *data, int count, int type);
#endif
#endif

}  // namespace Image
