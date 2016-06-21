#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

void Kernel::Setup(int device_id) {
#if defined(USE_CUDA)
  CheckError(cudaSetDevice(device_id));
  cublasCreate(&cublas_handle_);

#elif defined(USE_CL)
  easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);
  std::string cl_file = "./src/shadow/kernel.cl";
  cl_datatransform_kernel_ = easyCL->buildKernel(cl_file, "DataTransform");
  cl_im2col_kernel_ = easyCL->buildKernel(cl_file, "Im2Col");
  cl_pooling_kernel_ = easyCL->buildKernel(cl_file, "Pooling");
  cl_activations_kernel_ = easyCL->buildKernel(cl_file, "ActivateArray");
  cl_setarray_kernel_ = easyCL->buildKernel(cl_file, "SetArray");
  cl_setarrayrepeat_kernel_ = easyCL->buildKernel(cl_file, "SetArrayRepeat");
  clblasSetup();
#endif
}

void Kernel::Release() {
#if defined(USE_CUDA)
  if (cublas_handle_ != nullptr) cublasDestroy(cublas_handle_);

#elif defined(USE_CL)
  cl_datatransform_kernel_->~CLKernel();
  cl_im2col_kernel_->~CLKernel();
  cl_pooling_kernel_->~CLKernel();
  cl_activations_kernel_->~CLKernel();
  cl_setarray_kernel_->~CLKernel();
  cl_setarrayrepeat_kernel_->~CLKernel();
  easyCL->~EasyCL();
  clblasTeardown();
#endif
}

BType *Kernel::MakeBuffer(int size, float *host_ptr) {
#if defined(USE_CUDA)
  float *buffer;
  CheckError(cudaMalloc(&buffer, sizeof(float) * size));
  if (host_ptr != nullptr) {
    WriteBuffer(size, host_ptr, buffer);
  }
  return buffer;

#elif defined(USE_CL)
  cl_mem *buffer = new cl_mem();
  *buffer = clCreateBuffer(*easyCL->context, CL_MEM_READ_WRITE,
                           size * sizeof(float), host_ptr, nullptr);
  return buffer;

#else
  return nullptr;
#endif
}

void Kernel::ReadBuffer(int size, const BType *src, float *des) {
#if defined(USE_CUDA)
  CheckError(
      cudaMemcpy(des, src, sizeof(float) * size, cudaMemcpyDeviceToHost));

#elif defined(USE_CL)
  clEnqueueReadBuffer(*easyCL->queue, *src, CL_TRUE, 0, size * sizeof(float),
                      des, 0, nullptr, nullptr);
  clFinish(*easyCL->queue);
#endif
}

void Kernel::WriteBuffer(int size, const float *src, BType *des) {
#if defined(USE_CUDA)
  CheckError(
      cudaMemcpy(des, src, sizeof(float) * size, cudaMemcpyHostToDevice));

#elif defined(USE_CL)
  clEnqueueWriteBuffer(*easyCL->queue, *des, CL_TRUE, 0, size * sizeof(float),
                       src, 0, nullptr, nullptr);
  clFinish(*easyCL->queue);
#endif
}

void Kernel::CopyBuffer(int size, const BType *src, BType *des) {
#if defined(USE_CUDA)
  CheckError(
      cudaMemcpy(des, src, sizeof(float) * size, cudaMemcpyDeviceToDevice));

#elif defined(USE_CL)
  clEnqueueCopyBuffer(*easyCL->queue, *src, *des, 0, 0, size * sizeof(float), 0,
                      nullptr, nullptr);
  clFinish(*easyCL->queue);
#endif
}

void Kernel::ReleaseBuffer(BType *buffer) {
#if defined(USE_CUDA)
  CheckError(cudaFree(buffer));

#elif defined(USE_CL)
  clReleaseMemObject(*buffer);
#endif
}

void *Kernel::GetHandle() {
#if defined(USE_CUDA)
  return cublas_handle_;

#else
  return nullptr;
#endif
}

void *Kernel::GetQueue() {
#if defined(USE_CL)
  return easyCL->queue;

#else
  return nullptr;
#endif
}

#if defined(USE_CUDA)
cublasHandle_t Kernel::cublas_handle_ = nullptr;

dim3 Kernel::GridDim(int size) {
  unsigned int k = (unsigned int)(size - 1) / BLOCK + 1;
  unsigned int x = k;
  unsigned int y = 1;
  if (x > 65535) {
    x = (unsigned int)std::ceil(std::sqrt(k));
    y = (size - 1) / (x * BLOCK) + 1;
  }
  return dim3(x, y, 1);
}

void Kernel::CheckError(cudaError_t status) {
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
EasyCL *Kernel::easyCL = nullptr;
CLKernel *Kernel::cl_datatransform_kernel_ = nullptr;
CLKernel *Kernel::cl_im2col_kernel_ = nullptr;
CLKernel *Kernel::cl_pooling_kernel_ = nullptr;
CLKernel *Kernel::cl_activations_kernel_ = nullptr;
CLKernel *Kernel::cl_setarray_kernel_ = nullptr;
CLKernel *Kernel::cl_setarrayrepeat_kernel_ = nullptr;

void Kernel::DataTransform(int N, const cl_mem *in_data, float scale,
                           float mean_value, cl_mem *out_data) {
  cl_kernel kernel = cl_datatransform_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 2, sizeof(float), &scale);
  clSetKernelArg(kernel, 3, sizeof(float), &mean_value);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

void Kernel::Im2Col(const cl_mem *im_data, int offset, int in_c, int in_h,
                    int in_w, int ksize, int stride, int pad, int out_h,
                    int out_w, cl_mem *col_data) {
  cl_kernel kernel = cl_im2col_kernel_->GetKernel();
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
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

void Kernel::Pooling(const cl_mem *in_data, int batch, int in_c, int in_h,
                     int in_w, int ksize, int stride, int out_h, int out_w,
                     int mode, cl_mem *out_data) {
  cl_kernel kernel = cl_pooling_kernel_->GetKernel();
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
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

void Kernel::ActivateArray(int N, const shadow::ActivateType &type,
                           cl_mem *out_data) {
  cl_kernel kernel = cl_activations_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(int), &type);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

void Kernel::SetArray(int N, float value, cl_mem *out_data) {
  cl_kernel kernel = cl_setarray_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(float), &value);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), out_data);
  size_t global = N;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

void Kernel::SetArrayRepeat(int N, const cl_mem *value, int value_size,
                            cl_mem *out_data, int offset) {
  cl_kernel kernel = cl_setarrayrepeat_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(int), &N);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), value);
  clSetKernelArg(kernel, 2, sizeof(int), &value_size);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), out_data);
  clSetKernelArg(kernel, 4, sizeof(int), &offset);
  size_t global = N * value_size;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}
#endif
