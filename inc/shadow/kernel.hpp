#ifndef SHADOW_KERNEL_HPP
#define SHADOW_KERNEL_HPP

#include "shadow/proto/shadow.pb.h"

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
const int BLOCK = 512;
#endif

#if defined(USE_CL)
#include <EasyCL.h>
#include <clBLAS.h>
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
void DataTransform(int N, const T *in_data, float scale, float mean_value,
                   T *out_data);

template <typename T>
void Im2Col(const T *in_data, int offset, int in_c, int in_h, int in_w,
            int kernel_size, int stride, int pad, int out_h, int out_w,
            T *out_data);

template <typename T>
void Pooling(const T *in_data, int batch, int in_c, int in_h, int in_w,
             int kernel_size, int stride, int out_h, int out_w, int mode,
             T *out_data);

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

template <typename T>
void ActivateArray(int N, const shadow::ActivateType &type, T *out_data);

template <typename T>
void SetArray(int N, float value, T *out_data);

template <typename T>
void SetArrayRepeat(int N, const T *value, int value_size, T *out_data,
                    int offset);

#if defined(USE_CUDA)
dim3 GridDim(int size);
void CheckError(cudaError_t status);

extern cublasHandle_t cublas_handle_;

#elif defined(USE_CL)
extern EasyCL *easyCL;
static CLKernel *cl_datatransform_kernel_ = nullptr;
static CLKernel *cl_im2col_kernel_ = nullptr;
static CLKernel *cl_pooling_kernel_ = nullptr;
static CLKernel *cl_activations_kernel_ = nullptr;
static CLKernel *cl_setarray_kernel_ = nullptr;
static CLKernel *cl_setarrayrepeat_kernel_ = nullptr;
#endif

}  // namespace Kernel

#endif  // SHADOW_KERNEL_HPP
