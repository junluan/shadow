#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

void Kernel::KernelSetup(int device_id) {
#if defined(USE_CUDA)
  CUDA::CUDACheckError(cudaSetDevice(device_id));
  cublasCreate(&CUDA::BlasHandle);
#endif

#if defined(USE_CL)
  CL::easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);
  std::string kernelfile = "./src/shadow/kernel.cl";
  CL::cl_datatransform_kernel_ =
      CL::easyCL->buildKernel(kernelfile, "DataTransform");
  CL::cl_im2col_kernel_ = CL::easyCL->buildKernel(kernelfile, "Im2Col");
  CL::cl_pooling_kernel_ = CL::easyCL->buildKernel(kernelfile, "Pooling");
  CL::cl_activations_kernel_ =
      CL::easyCL->buildKernel(kernelfile, "ActivateArray");
  CL::cl_setarray_kernel_ = CL::easyCL->buildKernel(kernelfile, "SetArray");
  CL::cl_setarrayrepeat_kernel_ =
      CL::easyCL->buildKernel(kernelfile, "SetArrayRepeat");
  clblasSetup();
#endif
}

void Kernel::KernelRelease() {
#if defined(USE_CUDA)
  if (CUDA::BlasHandle != NULL)
    cublasDestroy(CUDA::BlasHandle);
#endif

#if defined(USE_CL)
  CL::cl_datatransform_kernel_->~CLKernel();
  CL::cl_im2col_kernel_->~CLKernel();
  CL::cl_pooling_kernel_->~CLKernel();
  CL::cl_activations_kernel_->~CLKernel();
  CL::cl_setarray_kernel_->~CLKernel();
  CL::cl_setarrayrepeat_kernel_->~CLKernel();
  CL::easyCL->~EasyCL();
  clblasTeardown();
#endif
}

#if defined(USE_CUDA)
cublasHandle_t CUDA::BlasHandle = NULL;

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
    Fatal("CUDA Error: " + message);
  }
  if (status2 != cudaSuccess) {
    std::string message(cudaGetErrorString(status));
    Fatal("CUDA Error Prev: " + message);
  }
}
#endif

#if defined(USE_CL)
EasyCL *CL::easyCL = NULL;
CLKernel *CL::cl_datatransform_kernel_ = NULL;
CLKernel *CL::cl_im2col_kernel_ = NULL;
CLKernel *CL::cl_pooling_kernel_ = NULL;
CLKernel *CL::cl_activations_kernel_ = NULL;
CLKernel *CL::cl_setarray_kernel_ = NULL;
CLKernel *CL::cl_setarrayrepeat_kernel_ = NULL;

cl_mem CL::CLMakeBuffer(int size, cl_mem_flags flags, void *host_ptr) {
  return clCreateBuffer(*easyCL->context, flags, size * sizeof(float), host_ptr,
                        NULL);
}

void CL::CLReadBuffer(int size, const cl_mem *src, float *des) {
  clEnqueueReadBuffer(*easyCL->queue, *src, CL_TRUE, 0, size * sizeof(float),
                      des, 0, NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLWriteBuffer(int size, cl_mem *des, const float *src) {
  clEnqueueWriteBuffer(*easyCL->queue, *des, CL_TRUE, 0, size * sizeof(float),
                       src, 0, NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLCopyBuffer(int size, const cl_mem *src, cl_mem des) {
  clEnqueueCopyBuffer(*easyCL->queue, *src, des, 0, 0, size * sizeof(float), 0,
                      NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLReleaseBuffer(cl_mem *buffer) { clReleaseMemObject(*buffer); }

void Kernel::DataTransform(int N, const cl_mem *in_data, float scale,
                           float mean_value, cl_mem *out_data) {
  cl_kernel kernel = CL::cl_datatransform_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 2, sizeof(float), &scale);
  clSetKernelArg(kernel, 3, sizeof(float), &mean_value);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::Im2Col(const cl_mem *im_data, int offset, int in_c, int in_h,
                    int in_w, int ksize, int stride, int pad, int out_h,
                    int out_w, cl_mem *col_data) {
  cl_kernel kernel = CL::cl_im2col_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), im_data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &ksize);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &out_h);
  clSetKernelArg(kernel, 9, sizeof(int), &out_w);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), col_data);
  size_t global = in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::Pooling(const cl_mem *in_data, int batch, int in_c, int in_h,
                     int in_w, int ksize, int stride, int out_h, int out_w,
                     int mode, cl_mem *out_data) {
  cl_kernel kernel = CL::cl_pooling_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &ksize);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &out_h);
  clSetKernelArg(kernel, 8, sizeof(int), &out_w);
  clSetKernelArg(kernel, 9, sizeof(int), &mode);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), out_data);
  size_t global = batch * in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::ActivateArray(int N, shadow::ActivateType a, cl_mem *out_data) {
  cl_kernel kernel = CL::cl_activations_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(int), &a);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::SetArray(int N, float value, cl_mem *out_data) {
  cl_kernel kernel = CL::cl_setarray_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(float), &value);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::SetArrayRepeat(int N, const cl_mem *value, int value_size,
                            cl_mem *out_data) {
  cl_kernel kernel = CL::cl_setarrayrepeat_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), value);
  clSetKernelArg(kernel, 2, sizeof(int), &value_size);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), out_data);
  size_t global = N * value_size;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}
#endif
