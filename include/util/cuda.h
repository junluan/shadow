#ifndef SHADOW_CUDA_H
#define SHADOW_CUDA_H

#ifdef USE_CUDA
#include "cublas_v2.h"
#include "cuda_runtime.h"

#define BLOCK 512

class CUDA {
public:
  static void CUDASetup(int device_id = 0);
  static void CUDARelease();

  static float *CUDAMakeBuffer(int size, float *host_ptr);
  static void CUDAReadBuffer(int size, const float *src, float *des);
  static void CUDAWriteBuffer(int size, float *des, const float *src);
  static void CUDAReleaseBuffer(float *buffer);
  static void CUDACheckError(cudaError_t status);
  static dim3 CUDAGridDim(int size);

  static cublasHandle_t BlasHandle;
};
#endif

#endif // SHADOW_CUDA_H
