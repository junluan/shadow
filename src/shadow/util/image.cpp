#include "shadow/util/image.hpp"

namespace Image {

#if defined(USE_CUDA)
template <typename T>
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data) {
  Kernel::DataTransform(N, in_data, scale, mean_value, out_data);
}

template <typename T>
void Im2Col(const std::vector<int> &in_shape, const T *in_data, int offset,
            int kernel_size, int stride, int pad,
            const std::vector<int> &out_shape, T *out_data) {
  Kernel::Im2Col(in_data, offset, in_shape[1], in_shape[2], in_shape[3],
                 kernel_size, stride, pad, out_shape[2], out_shape[3],
                 out_data);
}

template <typename T>
void Pooling(const std::vector<int> &in_shape, const T *in_data,
             int kernel_size, int stride, int mode,
             const std::vector<int> &out_shape, T *out_data) {
  Kernel::Pooling(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                  kernel_size, stride, out_shape[2], out_shape[3], mode,
                  out_data);
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  Kernel::Permute(in_data, count, num_axes, permute_order, old_steps, new_steps,
                  out_data);
}

// Explicit instantiation
template void DataTransform<float>(int N, const float *in_data, float scale,
                                   float mean_value, float *out_data);
template void Im2Col<float>(const std::vector<int> &in_shape,
                            const float *in_data, int offset, int kernel_size,
                            int stride, int pad,
                            const std::vector<int> &out_shape, float *out_data);
template void Pooling<float>(const std::vector<int> &in_shape,
                             const float *in_data, int kernel_size, int stride,
                             int mode, const std::vector<int> &out_shape,
                             float *out_data);
template void Permute<float, int>(const float *in_data, int count, int num_axes,
                                  const int *permute_order,
                                  const int *old_steps, const int *new_steps,
                                  float *out_data);

#elif defined(USE_CL)
template <typename T>
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data) {
  Kernel::DataTransform(N, in_data, scale, mean_value, out_data);
}

template <typename T>
void Im2Col(const std::vector<int> &in_shape, const T *in_data, int offset,
            int kernel_size, int stride, int pad,
            const std::vector<int> &out_shape, T *out_data) {
  Kernel::Im2Col(in_data, offset, in_shape[1], in_shape[2], in_shape[3],
                 kernel_size, stride, pad, out_shape[2], out_shape[3],
                 out_data);
}

template <typename T>
void Pooling(const std::vector<int> &in_shape, const T *in_data,
             int kernel_size, int stride, int mode,
             const std::vector<int> &out_shape, T *out_data) {
  Kernel::Pooling(in_data, in_shape[0], in_shape[1], in_shape[2], in_shape[3],
                  kernel_size, stride, out_shape[2], out_shape[3], mode,
                  out_data);
}

template <typename T>
void Permute(const T *in_data, int count, int num_axes,
             const std::vector<int> &permute_order,
             const std::vector<int> &old_steps,
             const std::vector<int> &new_steps, T *out_data) {}

// Explicit instantiation
template void DataTransform<cl_mem>(int N, const cl_mem *in_data, float scale,
                                    float mean_value, cl_mem *out_data);
template void Im2Col<cl_mem>(const std::vector<int> &in_shape,
                             const cl_mem *in_data, int offset, int kernel_size,
                             int stride, int pad,
                             const std::vector<int> &out_shape,
                             cl_mem *out_data);
template void Pooling<cl_mem>(const std::vector<int> &in_shape,
                              const cl_mem *in_data, int kernel_size,
                              int stride, int mode,
                              const std::vector<int> &out_shape,
                              cl_mem *out_data);
template void Permute<cl_mem>(const cl_mem *in_data, int count, int num_axes,
                              const std::vector<int> &permute_order,
                              const std::vector<int> &old_steps,
                              const std::vector<int> &new_steps,
                              cl_mem *out_data);

#else
template <typename T>
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data) {
  for (int i = 0; i < N; ++i) {
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
void Im2Col(const std::vector<int> &in_shape, const T *in_data, int offset,
            int kernel_size, int stride, int pad,
            const std::vector<int> &out_shape, T *out_data) {
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

  //#pragma omp parallel for
  //  for (int p = 0; p < in_c * out_h * out_w; ++p) {
  //    int c_out = (p / out_h / out_w) % in_c;
  //    int i_out = (p / out_w) % out_h;
  //    int j_out = p % out_w;
  //    int i_inp = -pad + i_out * stride;
  //    int j_inp = -pad + j_out * stride;
  //
  //    int im_offset = c_out * in_h * in_w;
  //    int col_offset = (c_out * ksize * ksize * out_h + i_out) * out_w +
  //    j_out;
  //    for (int ki = 0; ki < ksize; ++ki) {
  //      for (int kj = 0; kj < ksize; ++kj) {
  //        int i = i_inp + ki;
  //        int j = j_inp + kj;
  //        int col_index = col_offset + (ki * ksize + kj) * out_h * out_w;
  //        col_data[col_index] = (i >= 0 && j >= 0 && i < in_h && j < in_w)
  //                                  ? im_data[im_offset + i * in_w + j]
  //                                  : 0;
  //      }
  //    }
  //  }
}

template <typename T>
void Pooling(const std::vector<int> &in_shape, const T *in_data,
             int kernel_size, int stride, int mode,
             const std::vector<int> &out_shape, T *out_data) {
  int batch = in_shape[0];
  int in_c = in_shape[1], in_h = in_shape[2], in_w = in_shape[3];
  int out_h = out_shape[2], out_w = out_shape[3];

  int h_offset = ((in_h - kernel_size) % stride) / 2;
  int w_offset = ((in_w - kernel_size) % stride) / 2;

  for (int b = 0; b < batch; ++b) {
    for (int c = 0; c < in_c; ++c) {
      for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
          int out_index = w + out_w * (h + out_h * (c + in_c * b));
          float max = -10000.0f;
          float sum = 0.f;
          for (int ki = 0; ki < kernel_size; ++ki) {
            for (int kj = 0; kj < kernel_size; ++kj) {
              int cur_h = h_offset + h * stride + ki;
              int cur_w = w_offset + w * stride + kj;
              int index = cur_w + in_w * (cur_h + in_h * (c + b * in_c));
              bool valid =
                  (cur_h >= 0 && cur_h < in_h && cur_w >= 0 && cur_w < in_w);
              float value = valid ? in_data[index] : -10000.0f;
              max = (value > max) ? value : max;
              sum += valid ? in_data[index] : 0.f;
            }
          }
          if (mode == 0)
            out_data[out_index] = max;
          else
            out_data[out_index] = sum / (kernel_size * kernel_size);
        }
      }
    }
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

// Explicit instantiation
template void DataTransform<float>(int N, const float *in_data, float scale,
                                   float mean_value, float *out_data);
template void Im2Col<float>(const std::vector<int> &in_shape,
                            const float *in_data, int offset, int kernel_size,
                            int stride, int pad,
                            const std::vector<int> &out_shape, float *out_data);
template void Pooling<float>(const std::vector<int> &in_shape,
                             const float *in_data, int kernel_size, int stride,
                             int mode, const std::vector<int> &out_shape,
                             float *out_data);
template void Permute<float, int>(const float *in_data, int count, int num_axes,
                                  const int *permute_order,
                                  const int *old_steps, const int *new_steps,
                                  float *out_data);
#endif

}  // namespace Image
