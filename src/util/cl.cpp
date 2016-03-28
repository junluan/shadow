#include "cl.h"
#include <clBLAS.h>

EasyCL *CL::easyCL = NULL;
CLKernel *CL::cl_activations_kernel_ = NULL;
CLKernel *CL::cl_im2col_kernel_ = NULL;
CLKernel *CL::cl_biasoutput_kernel_ = NULL;
CLKernel *CL::cl_pool_kernel_ = NULL;
CLKernel *CL::cl_veccopy_kernel_ = NULL;
CLKernel *CL::cl_datatransform_kernel_ = NULL;

void CL::CLSetup() {
  clblasSetup();
  easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);
  std::string kernelfile = "./src/ocl/kernels.cl";
  cl_activations_kernel_ = easyCL->buildKernel(kernelfile, "ActivateArray");
  cl_im2col_kernel_ = easyCL->buildKernel(kernelfile, "Im2Col");
  cl_biasoutput_kernel_ = easyCL->buildKernel(kernelfile, "BiasOutput");
  cl_pool_kernel_ = easyCL->buildKernel(kernelfile, "Pool");
  cl_veccopy_kernel_ = easyCL->buildKernel(kernelfile, "VecCopy");
  cl_datatransform_kernel_ = easyCL->buildKernel(kernelfile, "DataTransform");
}

void CL::CLRelease() {
  clblasTeardown();
  cl_activations_kernel_->~CLKernel();
  cl_im2col_kernel_->~CLKernel();
  cl_biasoutput_kernel_->~CLKernel();
  cl_pool_kernel_->~CLKernel();
  cl_veccopy_kernel_->~CLKernel();
  cl_datatransform_kernel_->~CLKernel();
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

void CL::CLDataTransform(int N, cl_mem in_data, float scale, float mean_value,
                         cl_mem out_data) {
  cl_kernel kernel = cl_datatransform_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_data);
  clSetKernelArg(kernel, 2, sizeof(float), &scale);
  clSetKernelArg(kernel, 3, sizeof(float), &mean_value);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLBiasOutput(cl_mem biases, int batch, int num, int size,
                      cl_mem out_data) {
  cl_kernel kernel = cl_biasoutput_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &biases);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &num);
  clSetKernelArg(kernel, 3, sizeof(int), &size);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_data);
  size_t global = batch * num * size;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*easyCL->queue);
}

void CL::CLPooling(cl_mem in_data, int batch, int in_c, int in_h, int in_w,
                   int ksize, int stride, int out_h, int out_w, int mode,
                   cl_mem out_data) {
  cl_kernel kernel = cl_pool_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &ksize);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &out_h);
  clSetKernelArg(kernel, 8, sizeof(int), &out_w);
  clSetKernelArg(kernel, 9, sizeof(int), &mode);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), &out_data);
  size_t global = batch * in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*easyCL->queue);
}
