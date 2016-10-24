#ifndef SHADOW_KERNEL_HPP
#define SHADOW_KERNEL_HPP

#if defined(USE_CUDA)
#include "cuda_runtime.h"
const int BLOCK = 512;

#define CheckError(status)                                        \
  {                                                               \
    if (status != cudaSuccess) {                                  \
      Fatal("Error: " + std::string(cudaGetErrorString(status))); \
    }                                                             \
  }

#elif defined(USE_CL)
#include <EasyCL.h>
#endif

namespace Kernel {

void Setup(int device_id = 0);
void Release();

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr);

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des);

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des);

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des);

template <typename T>
void ReleaseBuffer(T *buffer);

template <typename T>
void DataTransform(const T *in_data, int count, float scale, float mean_value,
                   T *out_data);

template <typename T>
void Im2Col(const T *in_data, int offset, int in_c, int in_h, int in_w,
            int kernel_size, int stride, int pad, int out_h, int out_w,
            T *out_data);

template <typename T>
void Pooling(const T *in_data, int batch, int in_c, int in_h, int in_w,
             int kernel_size, int stride, int pad, int mode, int out_h,
             int out_w, T *out_data);

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data);

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

template <typename T>
void Activate(T *data, int count, int type);

#if defined(USE_CUDA)
dim3 GridDim(int size);

#elif defined(USE_CL)
extern EasyCL *easyCL;
#endif

}  // namespace Kernel

#endif  // SHADOW_KERNEL_HPP
