#ifndef SHADOW_CL_H
#define SHADOW_CL_H

#ifdef USE_CL
#include <EasyCL.h>

class CL {
public:
  static void CLSetup();
  static void CLRelease();

  static cl_mem CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr);
  static void CLReadBuffer(int size, const cl_mem &src, float *des);
  static void CLWriteBuffer(int size, cl_mem &des, float *src);
  static void CLCopyBuffer(int size, const cl_mem &src, cl_mem &des);
  static void CLReleaseBuffer(cl_mem buffer);

  static EasyCL *easyCL;
};
#endif

#endif // SHADOW_CL_H
