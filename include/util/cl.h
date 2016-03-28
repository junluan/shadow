#ifndef SHADOW_CL_H
#define SHADOW_CL_H

//#ifdef USE_CL
#include <EasyCL.h>
//#endif

class CL {
public:
  static void CLSetup();
  static void CLRelease();

  //#ifdef USE_CL
  static void CLDataTransform(int N, cl_mem in_data, float scale,
                              float mean_value, cl_mem out_data);
  static void CLBiasOutput(cl_mem biases, int batch, int num, int size,
                           cl_mem out_data);
  static void CLPooling(cl_mem in_data, int batch, int in_c, int in_h, int in_w,
                        int ksize, int stride, int out_h, int out_w, int mode,
                        cl_mem out_data);

  static cl_mem CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr);
  static void CLReadBuffer(int size, const cl_mem &src, float *des);
  static void CLWriteBuffer(int size, cl_mem &des, float *src);
  static void CLCopyBuffer(int size, const cl_mem &src, cl_mem &des);
  static void CLReleaseBuffer(cl_mem buffer);

  static EasyCL *easyCL;
  static CLKernel *cl_activations_kernel_;
  static CLKernel *cl_im2col_kernel_;
  static CLKernel *cl_biasoutput_kernel_;
  static CLKernel *cl_pool_kernel_;
  static CLKernel *cl_veccopy_kernel_;
  static CLKernel *cl_datatransform_kernel_;
  //#endif
};

#endif // SHADOW_CL_H
