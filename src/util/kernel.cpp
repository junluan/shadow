#include "kernel.h"

#ifdef USE_CL
CLKernel *Kernel::cl_activations_kernel_ = NULL;
CLKernel *Kernel::cl_im2col_kernel_ = NULL;
CLKernel *Kernel::cl_biasoutput_kernel_ = NULL;
CLKernel *Kernel::cl_pooling_kernel_ = NULL;
CLKernel *Kernel::cl_veccopy_kernel_ = NULL;
CLKernel *Kernel::cl_datatransform_kernel_ = NULL;

void Kernel::CLKernelSetup() {
  std::string kernelfile = "./src/util/kernel.cl";
  cl_activations_kernel_ = CL::easyCL->buildKernel(kernelfile, "ActivateArray");
  cl_im2col_kernel_ = CL::easyCL->buildKernel(kernelfile, "Im2Col");
  cl_biasoutput_kernel_ = CL::easyCL->buildKernel(kernelfile, "BiasOutput");
  cl_pooling_kernel_ = CL::easyCL->buildKernel(kernelfile, "Pooling");
  cl_veccopy_kernel_ = CL::easyCL->buildKernel(kernelfile, "VecCopy");
  cl_datatransform_kernel_ =
      CL::easyCL->buildKernel(kernelfile, "DataTransform");
}

void Kernel::CLKernelRelease() {
  cl_activations_kernel_->~CLKernel();
  cl_im2col_kernel_->~CLKernel();
  cl_biasoutput_kernel_->~CLKernel();
  cl_pooling_kernel_->~CLKernel();
  cl_veccopy_kernel_->~CLKernel();
  cl_datatransform_kernel_->~CLKernel();
}

void Kernel::CLDataTransform(int N, cl_mem in_data, float scale,
                             float mean_value, cl_mem out_data) {
  cl_kernel kernel = cl_datatransform_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &in_data);
  clSetKernelArg(kernel, 2, sizeof(float), &scale);
  clSetKernelArg(kernel, 3, sizeof(float), &mean_value);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::CLIm2Col(cl_mem im_data, int offset, int in_c, int in_h, int in_w,
                      int ksize, int stride, int pad, int out_h, int out_w,
                      cl_mem col_data) {
  cl_kernel kernel = cl_im2col_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &im_data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &ksize);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &out_h);
  clSetKernelArg(kernel, 9, sizeof(int), &out_w);
  clSetKernelArg(kernel, 10, sizeof(cl_mem), &col_data);
  size_t global = in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::CLPooling(cl_mem in_data, int batch, int in_c, int in_h, int in_w,
                       int ksize, int stride, int out_h, int out_w, int mode,
                       cl_mem out_data) {
  cl_kernel kernel = cl_pooling_kernel_->GetKernel();
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
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::CLActivateArray(int N, Activation a, cl_mem out_data) {
  cl_kernel kernel = Kernel::cl_activations_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(int), &a);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}

void Kernel::CLBiasOutput(cl_mem biases, int batch, int num, int size,
                          cl_mem out_data) {
  cl_kernel kernel = cl_biasoutput_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &biases);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &num);
  clSetKernelArg(kernel, 3, sizeof(int), &size);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &out_data);
  size_t global = batch * num * size;
  clEnqueueNDRangeKernel(*CL::easyCL->queue, kernel, 1, NULL, &global, NULL, 0,
                         NULL, NULL);
  clFinish(*CL::easyCL->queue);
}
#endif
