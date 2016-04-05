#ifdef USE_CL

#include "cl.h"
#include <clBLAS.h>

EasyCL *CL::easyCL = NULL;

void CL::CLSetup() {
  clblasSetup();
  easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);
}

void CL::CLRelease() {
  clblasTeardown();
  easyCL->~EasyCL();
}

cl_mem CL::CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr) {
  return clCreateBuffer(*easyCL->context, flags, size * sizeof(float), host_ptr,
                        NULL);
}

void CL::CLReadBuffer(int size, const cl_mem &src, float *des) {
  clEnqueueReadBuffer(*easyCL->queue, src, CL_TRUE, 0, size * sizeof(float),
                      des, 0, NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLWriteBuffer(int size, cl_mem &des, float *src) {
  clEnqueueWriteBuffer(*easyCL->queue, des, CL_TRUE, 0, size * sizeof(float),
                       src, 0, NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLCopyBuffer(int size, const cl_mem &src, cl_mem &des) {
  clEnqueueCopyBuffer(*easyCL->queue, src, des, 0, 0, size * sizeof(float), 0,
                      NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLReleaseBuffer(cl_mem buffer) { clReleaseMemObject(buffer); }

#endif
