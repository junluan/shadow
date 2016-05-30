#ifndef SHADOW_KERNEL_HPP
#define SHADOW_KERNEL_HPP

#include "shadow/util/activations.hpp"

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"
#define BLOCK 512
#endif

#ifdef USE_CL
#include <EasyCL.h>
#include <clBLAS.h>
#endif

class Kernel {
public:
  static void KernelSetup(int device_id = 0);
  static void KernelRelease();

#ifdef USE_CUDA
public:
  static void CUDADataTransform(int N, const float *in_data, float scale,
                                float mean_value, float *out_data);
  static void CUDAIm2Col(const float *im_data, int offset, int in_c, int in_h,
                         int in_w, int ksize, int stride, int pad, int out_h,
                         int out_w, float *col_data);
  static void CUDAPooling(const float *in_data, int batch, int in_c, int in_h,
                          int in_w, int ksize, int stride, int out_h, int out_w,
                          int mode, float *out_data);
  static void CUDAActivateArray(int N, shadow::ActivateType a, float *out_data);
  static void CUDABiasOutput(const float *biases, int batch, int num, int size,
                             float *out_data);
#endif

#ifdef USE_CL
public:
  static void CLDataTransform(int N, const cl_mem in_data, float scale,
                              float mean_value, cl_mem out_data);
  static void CLIm2Col(const cl_mem im_data, int offset, int in_c, int in_h,
                       int in_w, int ksize, int stride, int pad, int out_h,
                       int out_w, cl_mem col_data);
  static void CLPooling(const cl_mem in_data, int batch, int in_c, int in_h,
                        int in_w, int ksize, int stride, int out_h, int out_w,
                        int mode, cl_mem out_data);
  static void CLActivateArray(int N, shadow::ActivateType a, cl_mem out_data);
  static void CLBiasOutput(const cl_mem biases, int batch, int num, int size,
                           cl_mem out_data);
#endif
};

#ifdef USE_CUDA
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

#ifdef USE_CL
class CL {
public:
  static cl_mem CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr);
  static void CLReadBuffer(int size, const cl_mem src, float *des);
  static void CLWriteBuffer(int size, cl_mem des, const float *src);
  static void CLCopyBuffer(int size, const cl_mem src, cl_mem des);
  static void CLReleaseBuffer(cl_mem buffer);

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
