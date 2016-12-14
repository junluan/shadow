#include "shadow/util/image.hpp"
#include "shadow/kernel.hpp"

namespace Image {

#if !defined(USE_CUDA) & !defined(USE_CL)
template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;
  if (num_mean == 1) {
    for (int i = 0; i < count; ++i) {
      out_data[i] = (in_data[i] - mean_value[0]) * scale;
    }
  } else if (num_mean == in_c) {
    for (int i = 0; i < count; ++i) {
      int c_out = (i / spatial_dim) % in_c;
      out_data[i] = (in_data[i] - mean_value[c_out]) * scale;
    }
  } else if (num_mean == in_c * spatial_dim) {
    for (int i = 0; i < count; ++i) {
      int c_out = (i / spatial_dim) % in_c;
      int s_out = i % spatial_dim;
      out_data[i] =
          (in_data[i] - mean_value[c_out * spatial_dim + s_out]) * scale;
    }
  }
}

// check for 0 <= a < b
inline bool check_border(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *out_data) {
  in_data += offset;
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];
  int spatial_dim = in_h * in_w;
  for (int k_c = 0; k_c < in_c; ++k_c, in_data += spatial_dim) {
    for (int k_s = 0; k_s < kernel_size * kernel_size; ++k_s) {
      int k_h = k_s / kernel_size;
      int k_w = k_s % kernel_size;
      int im_row = -pad + k_h * dilation;
      for (int h = 0; h < out_h; ++h, im_row += stride) {
        if (check_border(im_row, in_h)) {
          int im_col = -pad + k_w * dilation;
          for (int w = 0; w < out_w; ++w, im_col += stride) {
            if (check_border(im_col, in_w)) {
              *(out_data++) = in_data[im_row * in_w + im_col];
            } else {
              *(out_data++) = 0;
            }
          }
        } else {
          for (int w = 0; w < out_w; ++w) {
            *(out_data++) = 0;
          }
        }
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
template void DataTransform(const float *in_data, const VecInt &in_shape,
                            float scale, int num_mean, const float *mean_value,
                            float *out_data);
template void Im2Col(const float *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     const VecInt &out_shape, float *out_data);
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
template void Activate(float *data, int count, int type);

#elif defined(USE_CL)
template <typename T>
void DataTransform(const T *in_data, const VecInt &in_shape, float scale,
                   int num_mean, const T *mean_value, T *out_data) {
  int in_c = in_shape[1], spatial_dim = in_shape[2] * in_shape[3];
  int count = in_shape[0] * in_c * spatial_dim;

  cl_kernel kernel = (*Kernel::cl_datatransform_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &spatial_dim);
  clSetKernelArg(kernel, 4, sizeof(float), &scale);
  clSetKernelArg(kernel, 5, sizeof(int), &num_mean);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), mean_value);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  cl_kernel kernel = (*Kernel::cl_im2col_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &kernel_size);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &dilation);
  clSetKernelArg(kernel, 9, sizeof(int), &out_h);
  clSetKernelArg(kernel, 10, sizeof(int), &out_w);
  clSetKernelArg(kernel, 11, sizeof(cl_mem), out_data);
  size_t global = in_c * out_h * out_w;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  cl_kernel kernel = (*Kernel::cl_pooling_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &kernel_size);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &mode);
  clSetKernelArg(kernel, 9, sizeof(int), &out_h);
  clSetKernelArg(kernel, 10, sizeof(int), &out_w);
  clSetKernelArg(kernel, 11, sizeof(cl_mem), out_data);
  size_t global = batch * in_c * out_h * out_w;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  cl_kernel kernel = (*Kernel::cl_concat_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &num_concats);
  clSetKernelArg(kernel, 3, sizeof(int), &concat_size);
  clSetKernelArg(kernel, 4, sizeof(int), &top_concat_axis);
  clSetKernelArg(kernel, 5, sizeof(int), &bottom_concat_axis);
  clSetKernelArg(kernel, 6, sizeof(int), &offset_concat_axis);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  cl_kernel kernel = (*Kernel::cl_permute_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &num_axes);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), permute_order);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), old_steps);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), new_steps);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

template <typename T>
void Activate(T *data, int count, int type) {
  cl_kernel kernel = (*Kernel::cl_activate_kernel_)();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &type);
  size_t global = count;
  clEnqueueNDRangeKernel((*Kernel::queue_)(), kernel, 1, nullptr, &global,
                         nullptr, 0, nullptr, nullptr);
  clFinish((*Kernel::queue_)());
}

// Explicit instantiation
template void DataTransform(const cl_mem *in_data, const VecInt &in_shape,
                            float scale, int num_mean, const cl_mem *mean_value,
                            cl_mem *out_data);
template void Im2Col(const cl_mem *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     const VecInt &out_shape, cl_mem *out_data);
template void Pooling(const cl_mem *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, cl_mem *out_data);
template void Concat(const cl_mem *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     cl_mem *out_data);
template void Permute(const cl_mem *in_data, int count, int num_axes,
                      const cl_mem *permute_order, const cl_mem *old_steps,
                      const cl_mem *new_steps, cl_mem *out_data);
template void Activate(cl_mem *data, int count, int type);
#endif

}  // namespace Image
