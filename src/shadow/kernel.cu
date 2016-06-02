#include "shadow/kernel.hpp"

__global__ void DataTransformKernel(int N, const float *in_data, float scale,
                                    float mean_value, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid > N)
    return;

  out_data[globalid] = (in_data[globalid] - mean_value) * scale;
}

void Kernel::DataTransform(int N, const float *in_data, float scale,
                           float mean_value, float *out_data) {
  DataTransformKernel<<<CUDA::CUDAGridDim(N), BLOCK>>>(N, in_data, scale,
                                                       mean_value, out_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
}

__global__ void Im2ColKernel(const float *im_data, int offset, int in_c,
                             int in_h, int in_w, int ksize, int stride, int pad,
                             int out_h, int out_w, float *col_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= in_c * out_h * out_w)
    return;

  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int i_inp = -pad + i_out * stride;
  int j_inp = -pad + j_out * stride;

  im_data += offset + c_out * in_h * in_w;
  col_data += (c_out * ksize * ksize * out_h + i_out) * out_w + j_out;

  for (int ki = 0; ki < ksize; ++ki) {
    for (int kj = 0; kj < ksize; ++kj) {
      int i = i_inp + ki;
      int j = j_inp + kj;
      *col_data = (i >= 0 && j >= 0 && i < in_h && j < in_w)
                      ? im_data[i * in_w + j]
                      : 0.f;
      col_data += out_h * out_w;
    }
  }
}

void Kernel::Im2Col(const float *im_data, int offset, int in_c, int in_h,
                    int in_w, int ksize, int stride, int pad, int out_h,
                    int out_w, float *col_data) {
  int N = in_c * out_h * out_w;
  Im2ColKernel<<<CUDA::CUDAGridDim(N), BLOCK>>>(im_data, offset, in_c, in_h,
                                                in_w, ksize, stride, pad, out_h,
                                                out_w, col_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
}

__global__ void PoolingKernel(const float *in_data, int batch, int in_c,
                              int in_h, int in_w, int ksize, int stride,
                              int out_h, int out_w, int mode, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= batch * in_c * out_h * out_w)
    return;

  int h_offset = ((in_h - ksize) % stride) / 2;
  int w_offset = ((in_w - ksize) % stride) / 2;

  int b_out = (globalid / (out_w * out_h * in_c)) % batch;
  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int i_inp = h_offset + i_out * stride;
  int j_inp = w_offset + j_out * stride;

  int offset = ((b_out * in_c + c_out) * in_h + i_inp) * in_w + j_inp;

  float max = -10000.0f;
  float sum = 0.f;
  for (int ki = 0; ki < ksize; ++ki) {
    for (int kj = 0; kj < ksize; ++kj) {
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
    out_data[globalid] = sum / (ksize * ksize);
}

void Kernel::Pooling(const float *in_data, int batch, int in_c, int in_h,
                     int in_w, int ksize, int stride, int out_h, int out_w,
                     int mode, float *out_data) {
  int N = batch * in_c * out_h * out_w;
  PoolingKernel<<<CUDA::CUDAGridDim(N), BLOCK>>>(in_data, batch, in_c, in_h,
                                                 in_w, ksize, stride, out_h,
                                                 out_w, mode, out_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
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
  if (globalid >= N)
    return;

  out_data[globalid] = Activate(out_data[globalid], mode);
}

void Kernel::ActivateArray(int N, shadow::ActivateType a, float *out_data) {
  ActivateArrayKernel<<<CUDA::CUDAGridDim(N), BLOCK>>>(N, a, out_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
}

__global__ void SetArrayKernel(int N, float value, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N)
    return;

  out_data[globalid] = value;
}

void Kernel::SetArray(int N, float value, float *out_data) {
  float val = {value};
  SetArrayKernel<<<CUDA::CUDAGridDim(N), BLOCK>>>(N, val, out_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
}

__global__ void SetArrayRepeatKernel(int N, const float *value, int value_size,
                                     float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= N * value_size)
    return;

  int value_index = globalid / N;
  out_data[globalid] = value[value_index];
}

void Kernel::SetArrayRepeat(int N, const float *value, int value_size,
                            float *out_data) {
  SetArrayRepeatKernel<<<CUDA::CUDAGridDim(N * value_size), BLOCK>>>(
      N, value, value_size, out_data);
  CUDA::CUDACheckError(cudaPeekAtLastError());
}
