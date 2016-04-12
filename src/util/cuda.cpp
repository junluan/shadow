#ifdef USE_CUDA

#include "cuda.h"
#include "util.h"

cublasHandle_t CUDA::BlasHandle = NULL;

void CUDA::CUDASetup(int device_id) {
  CUDACheckError(cudaSetDevice(device_id));
  cublasCreate(&BlasHandle);
}

void CUDA::CUDARelease() {
  if (BlasHandle != NULL)
    cublasDestroy(BlasHandle);
}

float *CUDA::CUDAMakeBuffer(int size, float *host_ptr) {
  float *buffer;
  CUDACheckError(cudaMalloc(&buffer, sizeof(float) * size));
  if (host_ptr) {
    CUDAWriteBuffer(size, buffer, host_ptr);
  }
  return buffer;
}

void CUDA::CUDAReadBuffer(int size, const float *src, float *des) {
  CUDACheckError(
      cudaMemcpy(des, src, sizeof(float) * size, cudaMemcpyDeviceToHost));
}

void CUDA::CUDAWriteBuffer(int size, float *des, const float *src) {
  CUDACheckError(
      cudaMemcpy(des, src, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void CUDA::CUDAReleaseBuffer(float *buffer) {
  CUDACheckError(cudaFree(buffer));
}

dim3 CUDA::CUDAGridDim(int size) {
  unsigned int k = (unsigned int)(size - 1) / BLOCK + 1;
  unsigned int x = k;
  unsigned int y = 1;
  if (x > 65535) {
    x = (unsigned int)ceilf(sqrtf(k));
    y = (size - 1) / (x * BLOCK) + 1;
  }
  return dim3(x, y, 1);
}

void CUDA::CUDACheckError(cudaError_t status) {
  cudaError_t status2 = cudaGetLastError();
  if (status != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::string message(s);
    error("CUDA Error: " + message);
  }
  if (status2 != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::string message(s);
    error("CUDA Error Prev: " + message);
  }
}

#endif
