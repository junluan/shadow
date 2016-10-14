#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

namespace Kernel {

#if defined(USE_CUDA)
void Setup(int device_id) { CheckError(cudaSetDevice(device_id)); }

void Release() {}

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr) {
  T *buffer;
  CheckError(cudaMalloc(&buffer, size * sizeof(Dtype)));
  if (host_ptr != nullptr) {
    WriteBuffer(size, host_ptr, buffer);
  }
  return buffer;
}

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToHost));
}

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyHostToDevice));
}

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des) {
  CheckError(
      cudaMemcpy(des, src, size * sizeof(Dtype), cudaMemcpyDeviceToDevice));
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  CheckError(cudaFree(buffer));
}

__global__ void DataTransformKernel(const float *in_data, int count,
                                    float scale, float mean_value,
                                    float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid > count) return;

  out_data[globalid] = (in_data[globalid] - mean_value) * scale;
}

template <typename T>
void DataTransform(const T *in_data, int count, float scale, float mean_value,
                   T *out_data) {
  DataTransformKernel<<<GridDim(count), BLOCK>>>(in_data, count, scale,
                                                 mean_value, out_data);
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

  float max = FLT_MIN;
  float sum = 0.f;
  for (int ki = 0; ki < kernel_size; ++ki) {
    for (int kj = 0; kj < kernel_size; ++kj) {
      int in = offset + ki * in_w + kj;
      bool valid = in < batch * in_c * in_h * in_w;
      float value = valid ? in_data[in] : FLT_MIN;
      max = (value > max) ? value : max;
      sum += valid ? in_data[in] : 0.f;
    }
  }
  if (mode == 0) {
    out_data[globalid] = max;
  } else {
    out_data[globalid] = sum / (kernel_size * kernel_size);
  }
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

__global__ void ConcatKernel(const float *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= count) return;

  int total_concat_size = concat_size * bottom_concat_axis;
  int concat_num = globalid / total_concat_size;
  int concat_index = globalid % total_concat_size;
  int top_index =
      concat_index +
      (concat_num * top_concat_axis + offset_concat_axis) * concat_size;
  out_data[top_index] = in_data[globalid];
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  ConcatKernel<<<GridDim(count), BLOCK>>>(
      in_data, count, num_concats, concat_size, top_concat_axis,
      bottom_concat_axis, offset_concat_axis, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void PermuteKernel(const float *in_data, int count, int num_axes,
                              const int *permute_order, const int *old_steps,
                              const int *new_steps, float *out_data) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= count) return;

  int old_idx = 0;
  int idx = globalid;
  for (int j = 0; j < num_axes; ++j) {
    int order = permute_order[j];
    old_idx += (idx / new_steps[j]) * old_steps[order];
    idx %= new_steps[j];
  }
  out_data[globalid] = in_data[old_idx];
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  PermuteKernel<<<GridDim(count), BLOCK>>>(
      in_data, count, num_axes, permute_order, old_steps, new_steps, out_data);
  CheckError(cudaPeekAtLastError());
}

__device__ float ActivateValue(float x, int type) {
  switch (type) {
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

__global__ void ActivateKernel(float *data, int count, int type) {
  int globalid =
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
  if (globalid >= count) return;

  data[globalid] = ActivateValue(data[globalid], type);
}

template <typename T>
void Activate(T *data, int count, int type) {
  ActivateKernel<<<GridDim(count), BLOCK>>>(data, count, type);
  CheckError(cudaPeekAtLastError());
}

// Explicit instantiation
template int *MakeBuffer<int, int>(int size, int *host_ptr);
template float *MakeBuffer<float, float>(int size, float *host_ptr);

template void ReadBuffer<int, int>(int size, const int *src, int *des);
template void ReadBuffer<float, float>(int size, const float *src, float *des);

template void WriteBuffer<int, int>(int size, const int *src, int *des);
template void WriteBuffer<float, float>(int size, const float *src, float *des);

template void CopyBuffer<int, int>(int size, const int *src, int *des);
template void CopyBuffer<float, float>(int size, const float *src, float *des);

template void ReleaseBuffer<int>(int *buffer);
template void ReleaseBuffer<float>(float *buffer);

template void DataTransform<float>(const float *in_data, int count, float scale,
                                   float mean_value, float *out_data);
template void Im2Col<float>(const float *in_data, int offset, int in_c,
                            int in_h, int in_w, int kernel_size, int stride,
                            int pad, int out_h, int out_w, float *out_data);
template void Pooling<float>(const float *in_data, int batch, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int out_h, int out_w, int mode, float *out_data);
template void Concat<float>(const float *in_data, int count, int num_concats,
                            int concat_size, int top_concat_axis,
                            int bottom_concat_axis, int offset_concat_axis,
                            float *out_data);
template void Permute<float, int>(const float *in_data, int count, int num_axes,
                                  const int *permute_order,
                                  const int *old_steps, const int *new_steps,
                                  float *out_data);
template void Activate<float>(float *data, int count, int type);

dim3 GridDim(int size) {
  unsigned int k = (unsigned int)(size - 1) / BLOCK + 1;
  unsigned int x = k;
  unsigned int y = 1;
  if (x > 65535) {
    x = (unsigned int)std::ceil(std::sqrt(k));
    y = (size - 1) / (x * BLOCK) + 1;
  }
  return dim3(x, y, 1);
}
#endif

}  // namespace Kernel
