#include <cstdlib>
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

#define CUDA_KERNEL_LOOP(globalid, count)                               \
  const int globalid =                                                  \
      (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x; \
  if (globalid >= count) return;

__global__ void KernelDataTransform(const float *in_data, int count, int in_c,
                                    int spatial_dim, float scale, int num_mean,
                                    const float *mean_value, float *out_data) {
  CUDA_KERNEL_LOOP(globalid, count);

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

template <typename T>
void DataTransform(const T *in_data, int count, int in_c, int spatial_dim,
                   float scale, int num_mean, const T *mean_value,
                   T *out_data) {
  KernelDataTransform<<<GridDim(count), BLOCK>>>(
      in_data, count, in_c, spatial_dim, scale, num_mean, mean_value, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelIm2Col(const float *im_data, int offset, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int out_h, int out_w, float *col_data) {
  CUDA_KERNEL_LOOP(globalid, in_c * out_h * out_w);

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
  KernelIm2Col<<<GridDim(N), BLOCK>>>(in_data, offset, in_c, in_h, in_w,
                                      kernel_size, stride, pad, out_h, out_w,
                                      out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelPooling(const float *in_data, int batch, int in_c,
                              int in_h, int in_w, int kernel_size, int stride,
                              int pad, int mode, int out_h, int out_w,
                              float *out_data) {
  CUDA_KERNEL_LOOP(globalid, batch * in_c * out_h * out_w);

  int b_out = (globalid / (out_w * out_h * in_c)) % batch;
  int c_out = (globalid / (out_w * out_h)) % in_c;
  int i_out = (globalid / out_w) % out_h;
  int j_out = globalid % out_w;

  int kistart = i_out * stride - pad, kjstart = j_out * stride - pad;
  int kiend = min(kistart + kernel_size, in_h);
  int kjend = min(kjstart + kernel_size, in_w);
  int pool_size = (kiend - kistart) * (kjend - kjstart);
  kistart = max(kistart, 0), kjstart = max(kjstart, 0);
  kiend = min(kiend, in_h), kjend = min(kjend, in_w);

  float max = -FLT_MAX;
  float sum = 0.f;
  for (int ki = kistart; ki < kiend; ++ki) {
    for (int kj = kjstart; kj < kjend; ++kj) {
      int index = kj + in_w * (ki + in_h * (c_out + in_c * b_out));
      float value = in_data[index];
      max = (value > max) ? value : max;
      sum += value;
    }
  }
  if (mode == 0) {
    out_data[globalid] = max;
  } else {
    out_data[globalid] = sum / pool_size;
  }
}

template <typename T>
void Pooling(const T *in_data, int batch, int in_c, int in_h, int in_w,
             int kernel_size, int stride, int pad, int mode, int out_h,
             int out_w, T *out_data) {
  int N = batch * in_c * out_h * out_w;
  KernelPooling<<<GridDim(N), BLOCK>>>(in_data, batch, in_c, in_h, in_w,
                                       kernel_size, stride, pad, mode, out_h,
                                       out_w, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelConcat(const float *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             float *out_data) {
  CUDA_KERNEL_LOOP(globalid, count);

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
  KernelConcat<<<GridDim(count), BLOCK>>>(
      in_data, count, num_concats, concat_size, top_concat_axis,
      bottom_concat_axis, offset_concat_axis, out_data);
  CheckError(cudaPeekAtLastError());
}

__global__ void KernelPermute(const float *in_data, int count, int num_axes,
                              const int *permute_order, const int *old_steps,
                              const int *new_steps, float *out_data) {
  CUDA_KERNEL_LOOP(globalid, count);

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
  KernelPermute<<<GridDim(count), BLOCK>>>(
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

__global__ void KernelActivate(float *data, int count, int type) {
  CUDA_KERNEL_LOOP(globalid, count);

  data[globalid] = ActivateValue(data[globalid], type);
}

template <typename T>
void Activate(T *data, int count, int type) {
  KernelActivate<<<GridDim(count), BLOCK>>>(data, count, type);
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

template void DataTransform<float>(const float *in_data, int count, int in_c,
                                   int spatial_dim, float scale, int num_mean,
                                   const float *mean_value, float *out_data);
template void Im2Col<float>(const float *in_data, int offset, int in_c,
                            int in_h, int in_w, int kernel_size, int stride,
                            int pad, int out_h, int out_w, float *out_data);
template void Pooling<float>(const float *in_data, int batch, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int mode, int out_h, int out_w,
                             float *out_data);
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
