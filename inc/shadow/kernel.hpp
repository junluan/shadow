#ifndef SHADOW_KERNEL_HPP
#define SHADOW_KERNEL_HPP

#include "shadow/proto/shadow.pb.h"

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
const int BLOCK = 512;
#define BType float

#elif defined(USE_CL)
#include <EasyCL.h>
#include <clBLAS.h>
#define BType cl_mem

#else
#define BType float
#endif

class Kernel {
 public:
  static void Setup(int device_id = 0);
  static void Release();

  static BType *MakeBuffer(int size, float *host_ptr);
  static void ReadBuffer(int size, const BType *src, float *des);
  static void WriteBuffer(int size, const float *src, BType *des);
  static void CopyBuffer(int size, const BType *src, BType *des);
  static void ReleaseBuffer(BType *buffer);

  static void DataTransform(int N, const BType *in_data, float scale,
                            float mean_value, BType *out_data);
  static void Im2Col(const BType *im_data, int offset, int in_c, int in_h,
                     int in_w, int ksize, int stride, int pad, int out_h,
                     int out_w, BType *col_data);
  static void Pooling(const BType *in_data, int batch, int in_c, int in_h,
                      int in_w, int ksize, int stride, int out_h, int out_w,
                      int mode, BType *out_data);
  static void ActivateArray(int N, const shadow::ActivateType &type,
                            BType *out_data);
  static void SetArray(int N, float value, BType *out_data);
  static void SetArrayRepeat(int N, const BType *value, int value_size,
                             BType *out_data, int offset);

  static void *GetHandle();
  static void *GetQueue();

 private:
#if defined(USE_CUDA)
  static dim3 GridDim(int size);
  static void CheckError(cudaError_t status);

  static cublasHandle_t cublas_handle_;

#elif defined(USE_CL)
  static EasyCL *easyCL;
  static CLKernel *cl_datatransform_kernel_;
  static CLKernel *cl_im2col_kernel_;
  static CLKernel *cl_pooling_kernel_;
  static CLKernel *cl_activations_kernel_;
  static CLKernel *cl_setarray_kernel_;
  static CLKernel *cl_setarrayrepeat_kernel_;
#endif
};

#endif  // SHADOW_KERNEL_HPP
