#ifndef SHADOW_KERNEL_H
#define SHADOW_KERNEL_H

#include "activations.h"
#include "cl.h"
#include "cuda.h"

class Kernel {
#ifdef USE_CUDA
public:
  static void CUDADataTransform(int N, float *in_data, float scale,
                                float mean_value, float *out_data);
  static void CUDAIm2Col(float *im_data, int offset, int in_c, int in_h,
                         int in_w, int ksize, int stride, int pad, int out_h,
                         int out_w, float *col_data);
  static void CUDAPooling(float *in_data, int batch, int in_c, int in_h,
                          int in_w, int ksize, int stride, int out_h, int out_w,
                          int mode, float *out_data);
  static void CUDAActivateArray(int N, Activation a, float *out_data);
  static void CUDABiasOutput(float *biases, int batch, int num, int size,
                             float *out_data);
#endif

#ifdef USE_CL
public:
  static void CLKernelSetup();
  static void CLKernelRelease();

  static void CLDataTransform(int N, cl_mem in_data, float scale,
                              float mean_value, cl_mem out_data);
  static void CLIm2Col(cl_mem im_data, int offset, int in_c, int in_h, int in_w,
                       int ksize, int stride, int pad, int out_h, int out_w,
                       cl_mem col_data);
  static void CLPooling(cl_mem in_data, int batch, int in_c, int in_h, int in_w,
                        int ksize, int stride, int out_h, int out_w, int mode,
                        cl_mem out_data);
  static void CLActivateArray(int N, Activation a, cl_mem out_data);
  static void CLBiasOutput(cl_mem biases, int batch, int num, int size,
                           cl_mem out_data);

  static CLKernel *cl_activations_kernel_;
  static CLKernel *cl_im2col_kernel_;
  static CLKernel *cl_biasoutput_kernel_;
  static CLKernel *cl_pooling_kernel_;
  static CLKernel *cl_veccopy_kernel_;
  static CLKernel *cl_datatransform_kernel_;
#endif
};

#endif // SHADOW_KERNEL_H
