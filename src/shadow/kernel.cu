#include "shadow/kernel.hpp"

namespace Kernel {

__global__ void DataTransformKernel(int N, const float *in_data, float scale,
                                    float mean_value, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid > N) return;

  out_data[globalid] = (in_data[globalid] - mean_value) * scale;
}

template <typename T>
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data) {
  DataTransformKernel<<<GridDim(N), BLOCK>>>(N, in_data, scale, mean_value,
                                             out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void Im2ColKernel(const float *im_data, int offset, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int out_h, int out_w, float *col_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= in_c * out_h * out_w) return;

  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int i_inp = -pad + i_out * stride;
  int j_inp = -pad + j_out * stride;

  im_data += offset + c_out * in_h * in_w;
  col_data +=
      (c_out * kernel_size * kernel_size * out_h + i_out) * out_w + j_out;

  for (int ki = 0; ki < kernel_size; ++ki) {
    for (int kj = 0; kj < kernel_size; ++kj) {
      int i = i_inp + ki;
      int j = j_inp + kj;
      *col_data = (i >= 0 && j >= 0 && i < in_h && j < in_w)
                      ? im_data[i * in_w + j]
                      : 0.f;
      col_data += out_h * out_w;
    }
  }
}

template <typename T>
void Im2Col(const T *in_data, int offset, int in_c, int in_h, int in_w,
            int kernel_size, int stride, int pad, int out_h, int out_w,
            T *out_data) {
  int N = in_c * out_h * out_w;
  Im2ColKernel<<<GridDim(N), BLOCK>>>(in_data, offset, in_c, in_h, in_w,
                                      kernel_size, stride, pad, out_h, out_w,
                                      out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void PoolingKernel(const float *in_data, int batch, int in_c,
                              int in_h, int in_w, int kernel_size, int stride,
                              int out_h, int out_w, int mode, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= batch * in_c * out_h * out_w) return;

  int h_offset = ((in_h - kernel_size) % stride) / 2;
  int w_offset = ((in_w - kernel_size) % stride) / 2;

  int b_out = (globalid / (out_w * out_h * in_c)) % batch;
  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int i_inp = h_offset + i_out * stride;
  int j_inp = w_offset + j_out * stride;

  int offset = ((b_out * in_c + c_out) * in_h + i_inp) * in_w + j_inp;

  float max = -10000.0f;
  float sum = 0.f;
  for (int ki = 0; ki < kernel_size; ++ki) {
    for (int kj = 0; kj < kernel_size; ++kj) {
      int in = offset + ki * in_w + kj;
      bool valid = in < batch * in_c * in_h * in_w;
      float value = valid ? in_data[in] : -10000.0f;
      max = (value > max) ? value : max;
      sum += valid ? in_data[in] : 0.f;
    }
  }
  if (mode == 0)
    out_data[globalid] = max;
  else
    out_data[globalid] = sum / (kernel_size * kernel_size);
}

template <typename T>
void Pooling(const T *in_data, int batch, int in_c, int in_h, int in_w,
             int kernel_size, int stride, int out_h, int out_w, int mode,
             T *out_data) {
  int N = batch * in_c * out_h * out_w;
  PoolingKernel<<<GridDim(N), BLOCK>>>(in_data, batch, in_c, in_h, in_w,
                                       kernel_size, stride, out_h, out_w, mode,
                                       out_data);
  CheckError(cudaPeekAtLastError());
}

__device__ float Activate(float x, int mode) {
  switch (mode) {
    case 0:
      return x; /*linear*/
    case 1:
      return x * (x > 0); /*relu*/
    case 2:
      return (x > 0) ? x : .1f * x; /*leaky*/
    default:
      return x;
  }
}

__global__ void ActivateArrayKernel(int N, int mode, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N) return;

  out_data[globalid] = Activate(out_data[globalid], mode);
}

template <typename T>
void ActivateArray(int N, const shadow::ActivateType &type, T *out_data) {
  ActivateArrayKernel<<<GridDim(N), BLOCK>>>(N, type, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void SetArrayKernel(int N, float value, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N) return;

  out_data[globalid] = value;
}

template <typename T>
void SetArray(int N, float value, T *out_data) {
  float val = {value};
  SetArrayKernel<<<GridDim(N), BLOCK>>>(N, val, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void SetArrayRepeatKernel(int N, const float *value, int value_size,
                                     float *out_data, int offset) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N * value_size) return;

  int value_index = globalid / N;
  out_data[offset + globalid] = value[value_index];
}

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset) {
  SetArrayRepeatKernel<<<GridDim(N * value_size), BLOCK>>>(N, value, value_size,
                                                           out_data, offset);
  CheckError(cudaPeekAtLastError());
}

// Explicit instantiation
template void DataTransform<float>(int N, const float *in_data, float scale,
                                   float mean_value, float *out_data);

template void Im2Col<float>(const float *in_data, int offset, int in_c,
                            int in_h, int in_w, int kernel_size, int stride,
                            int pad, int out_h, int out_w, float *out_data);

template void Pooling<float>(const float *in_data, int batch, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int out_h, int out_w, int mode, float *out_data);

template void ActivateArray<float>(int N, const shadow::ActivateType &type,
                                   float *out_data);

template void SetArray<float>(int N, float value, float *out_data);

template void SetArrayRepeat<float>(int N, const float *value, int value_size,
                                    float *out_data, int offset);

}  // namespace Kernel
