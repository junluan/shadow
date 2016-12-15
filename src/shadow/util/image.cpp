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

  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_datatransform_kernel_;
  kernel->SetArguments(*in_data, count, in_c, spatial_dim, scale, num_mean,
                       *mean_value, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Im2Col(const T *in_data, const VecInt &in_shape, int offset,
            int kernel_size, int stride, int pad, int dilation,
            const VecInt &out_shape, T *out_data) {
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  size_t global = in_c * out_h * out_w;
  EasyCL::Kernel *kernel = Kernel::cl_im2col_kernel_;
  kernel->SetArguments(*in_data, offset, in_c, in_h, in_w, kernel_size, stride,
                       pad, dilation, out_h, out_w, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Pooling(const T *in_data, const VecInt &in_shape, int kernel_size,
             int stride, int pad, int mode, const VecInt &out_shape,
             T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  size_t global = batch * in_c * out_h * out_w;
  EasyCL::Kernel *kernel = Kernel::cl_pooling_kernel_;
  kernel->SetArguments(*in_data, batch, in_c, in_h, in_w, kernel_size, stride,
                       pad, mode, out_h, out_w, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_concat_kernel_;
  kernel->SetArguments(*in_data, count, num_concats, concat_size,
                       top_concat_axis, bottom_concat_axis, offset_concat_axis,
                       *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_permute_kernel_;
  kernel->SetArguments(*in_data, count, num_axes, *permute_order, *old_steps,
                       *new_steps, *out_data);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

template <typename T>
void Activate(T *data, int count, int type) {
  size_t global = count;
  EasyCL::Kernel *kernel = Kernel::cl_activate_kernel_;
  kernel->SetArguments(*data, count, type);
  kernel->Launch(*Kernel::queue_, {global}, Kernel::event_);
  Kernel::queue_->Finish();
}

// Explicit instantiation
template void DataTransform(const BufferF *in_data, const VecInt &in_shape,
                            float scale, int num_mean,
                            const BufferF *mean_value, BufferF *out_data);
template void Im2Col(const BufferF *in_data, const VecInt &in_shape, int offset,
                     int kernel_size, int stride, int pad, int dilation,
                     const VecInt &out_shape, BufferF *out_data);
template void Pooling(const BufferF *in_data, const VecInt &in_shape,
                      int kernel_size, int stride, int pad, int mode,
                      const VecInt &out_shape, BufferF *out_data);
template void Concat(const BufferF *in_data, int count, int num_concats,
                     int concat_size, int top_concat_axis,
                     int bottom_concat_axis, int offset_concat_axis,
                     BufferF *out_data);
template void Permute(const BufferF *in_data, int count, int num_axes,
                      const BufferI *permute_order, const BufferI *old_steps,
                      const BufferI *new_steps, BufferF *out_data);
template void Activate(BufferF *data, int count, int type);
#endif

}  // namespace Image
