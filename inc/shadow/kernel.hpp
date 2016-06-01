#ifndef SHADOW_KERNEL_HPP
#define SHADOW_KERNEL_HPP

#include "shadow/util/activations.hpp"

#if defined(USE_CUDA)
#include "cublas_v2.h"
#include "cuda_runtime.h"
#define BLOCK 512
#endif

#if defined(USE_CL)
#include <EasyCL.h>
#include <clBLAS.h>
#define BType cl_mem
#else
#define BType float
#endif

class Kernel {
public:
  static void KernelSetup(int device_id = 0);
  static void KernelRelease();

  static void DataTransform(int N, const BType *in_data, float scale,
                            float mean_value, BType *out_data);
  static void Im2Col(const BType *im_data, int offset, int in_c, int in_h,
                     int in_w, int ksize, int stride, int pad, int out_h,
                     int out_w, BType *col_data);
  static void Pooling(const BType *in_data, int batch, int in_c, int in_h,
                      int in_w, int ksize, int stride, int out_h, int out_w,
                      int mode, BType *out_data);
  static void ActivateArray(int N, shadow::ActivateType a, BType *out_data);
  static void BiasOutput(const BType *biases, int batch, int num, int size,
                         BType *out_data);
};

#if defined(USE_CUDA)
class CUDA {
public:
  static float *CUDAMakeBuffer(int size, float *host_ptr);
  static void CUDAReadBuffer(int size, const float *src, float *des);
  static void CUDAWriteBuffer(int size, float *des, const float *src);
  static void CUDAReleaseBuffer(float *buffer);
  static void CUDACheckError(cudaError_t status);
  static dim3 CUDAGridDim(int size);

  static cublasHandle_t BlasHandle;
};
#endif

#if defined(USE_CL)
class CL {
public:
  static cl_mem CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr);
  static void CLReadBuffer(int size, const cl_mem *src, float *des);
  static void CLWriteBuffer(int size, cl_mem *des, const float *src);
  static void CLCopyBuffer(int size, const cl_mem *src, cl_mem des);
  static void CLReleaseBuffer(cl_mem *buffer);

  static EasyCL *easyCL;
  static CLKernel *cl_activations_kernel_;
  static CLKernel *cl_im2col_kernel_;
  static CLKernel *cl_biasoutput_kernel_;
  static CLKernel *cl_pooling_kernel_;
  static CLKernel *cl_veccopy_kernel_;
  static CLKernel *cl_datatransform_kernel_;
};
#endif

#endif // SHADOW_KERNEL_HPP
