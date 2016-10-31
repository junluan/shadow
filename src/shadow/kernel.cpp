#include "shadow/kernel.hpp"

namespace Kernel {

#if defined(USE_CL)
EasyCL *easyCL = nullptr;
static CLKernel *cl_datatransform_kernel_ = nullptr;
static CLKernel *cl_im2col_kernel_ = nullptr;
static CLKernel *cl_pooling_kernel_ = nullptr;
static CLKernel *cl_concat_kernel_ = nullptr;
static CLKernel *cl_permute_kernel_ = nullptr;
static CLKernel *cl_activate_kernel_ = nullptr;

void Setup(int device_id) {
  easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);
  std::string cl_file = "./src/shadow/kernel.cl";
  cl_datatransform_kernel_ = easyCL->buildKernel(cl_file, "DataTransform");
  cl_im2col_kernel_ = easyCL->buildKernel(cl_file, "Im2Col");
  cl_pooling_kernel_ = easyCL->buildKernel(cl_file, "Pooling");
  cl_concat_kernel_ = easyCL->buildKernel(cl_file, "Concat");
  cl_permute_kernel_ = easyCL->buildKernel(cl_file, "Permute");
  cl_activate_kernel_ = easyCL->buildKernel(cl_file, "Activate");
}

void Release() {
  cl_datatransform_kernel_->~CLKernel();
  cl_im2col_kernel_->~CLKernel();
  cl_pooling_kernel_->~CLKernel();
  cl_concat_kernel_->~CLKernel();
  cl_permute_kernel_->~CLKernel();
  cl_activate_kernel_->~CLKernel();
  easyCL->~EasyCL();
}

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr) {
  T *buffer = new cl_mem();
  *buffer = clCreateBuffer(*easyCL->context, CL_MEM_READ_WRITE,
                           size * sizeof(Dtype), host_ptr, nullptr);
  return buffer;
}

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des) {
  clEnqueueReadBuffer(*easyCL->queue, *src, CL_TRUE, 0, size * sizeof(Dtype),
                      des, 0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des) {
  clEnqueueWriteBuffer(*easyCL->queue, *des, CL_TRUE, 0, size * sizeof(Dtype),
                       src, 0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des) {
  clEnqueueCopyBuffer(*easyCL->queue, *src, *des, 0, 0, size * sizeof(Dtype), 0,
                      nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  clReleaseMemObject(*buffer);
}

template <typename T>
void DataTransform(const T *in_data, int count, int in_c, int spatial_dim,
                   float scale, int num_mean, const T *mean_value,
                   T *out_data) {
  cl_kernel kernel = cl_datatransform_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &spatial_dim);
  clSetKernelArg(kernel, 4, sizeof(float), &scale);
  clSetKernelArg(kernel, 5, sizeof(int), &num_mean);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), mean_value);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T>
void Im2Col(const T *in_data, int offset, int in_c, int in_h, int in_w,
            int kernel_size, int stride, int pad, int dilation, int out_h,
            int out_w, T *out_data) {
  cl_kernel kernel = cl_im2col_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &offset);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &kernel_size);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &dilation);
  clSetKernelArg(kernel, 9, sizeof(int), &out_h);
  clSetKernelArg(kernel, 10, sizeof(int), &out_w);
  clSetKernelArg(kernel, 11, sizeof(cl_mem), out_data);
  size_t global = in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T>
void Pooling(const T *in_data, int batch, int in_c, int in_h, int in_w,
             int kernel_size, int stride, int pad, int mode, int out_h,
             int out_w, T *out_data) {
  cl_kernel kernel = cl_pooling_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &batch);
  clSetKernelArg(kernel, 2, sizeof(int), &in_c);
  clSetKernelArg(kernel, 3, sizeof(int), &in_h);
  clSetKernelArg(kernel, 4, sizeof(int), &in_w);
  clSetKernelArg(kernel, 5, sizeof(int), &kernel_size);
  clSetKernelArg(kernel, 6, sizeof(int), &stride);
  clSetKernelArg(kernel, 7, sizeof(int), &pad);
  clSetKernelArg(kernel, 8, sizeof(int), &mode);
  clSetKernelArg(kernel, 9, sizeof(int), &out_h);
  clSetKernelArg(kernel, 10, sizeof(int), &out_w);
  clSetKernelArg(kernel, 11, sizeof(cl_mem), out_data);
  size_t global = batch * in_c * out_h * out_w;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T>
void Concat(const T *in_data, int count, int num_concats, int concat_size,
            int top_concat_axis, int bottom_concat_axis, int offset_concat_axis,
            T *out_data) {
  cl_kernel kernel = cl_concat_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &num_concats);
  clSetKernelArg(kernel, 3, sizeof(int), &concat_size);
  clSetKernelArg(kernel, 4, sizeof(int), &top_concat_axis);
  clSetKernelArg(kernel, 5, sizeof(int), &bottom_concat_axis);
  clSetKernelArg(kernel, 6, sizeof(int), &offset_concat_axis);
  clSetKernelArg(kernel, 7, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data) {
  cl_kernel kernel = cl_permute_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), in_data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &num_axes);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), permute_order);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), old_steps);
  clSetKernelArg(kernel, 5, sizeof(cl_mem), new_steps);
  clSetKernelArg(kernel, 6, sizeof(cl_mem), out_data);
  size_t global = count;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

template <typename T>
void Activate(T *data, int count, int type) {
  cl_kernel kernel = cl_activate_kernel_->GetKernel();
  clSetKernelArg(kernel, 0, sizeof(cl_mem), data);
  clSetKernelArg(kernel, 1, sizeof(int), &count);
  clSetKernelArg(kernel, 2, sizeof(int), &type);
  size_t global = count;
  clEnqueueNDRangeKernel(*easyCL->queue, kernel, 1, nullptr, &global, nullptr,
                         0, nullptr, nullptr);
  clFinish(*easyCL->queue);
}

// Explicit instantiation
template cl_mem *MakeBuffer<cl_mem, int>(int size, int *host_ptr);
template cl_mem *MakeBuffer<cl_mem, float>(int size, float *host_ptr);

template void ReadBuffer<cl_mem, int>(int size, const cl_mem *src, int *des);
template void ReadBuffer<cl_mem, float>(int size, const cl_mem *src,
                                        float *des);

template void WriteBuffer<cl_mem, int>(int size, const int *src, cl_mem *des);
template void WriteBuffer<cl_mem, float>(int size, const float *src,
                                         cl_mem *des);

template void CopyBuffer<cl_mem, int>(int size, const cl_mem *src, cl_mem *des);

template void ReleaseBuffer<cl_mem>(cl_mem *buffer);

template void DataTransform<cl_mem>(const cl_mem *in_data, int count, int in_c,
                                    int spatial_dim, float scale, int num_mean,
                                    const cl_mem *mean_value, cl_mem *out_data);
template void Im2Col<cl_mem>(const cl_mem *in_data, int offset, int in_c,
                             int in_h, int in_w, int kernel_size, int stride,
                             int pad, int dilation, int out_h, int out_w,
                             cl_mem *out_data);
template void Pooling<cl_mem>(const cl_mem *in_data, int batch, int in_c,
                              int in_h, int in_w, int kernel_size, int stride,
                              int pad, int mode, int out_h, int out_w,
                              cl_mem *out_data);
template void Concat<cl_mem>(const cl_mem *in_data, int count, int num_concats,
                             int concat_size, int top_concat_axis,
                             int bottom_concat_axis, int offset_concat_axis,
                             cl_mem *out_data);
template void Permute<cl_mem, cl_mem>(const cl_mem *in_data, int count,
                                      int num_axes, const cl_mem *permute_order,
                                      const cl_mem *old_steps,
                                      const cl_mem *new_steps,
                                      cl_mem *out_data);
template void Activate<cl_mem>(cl_mem *data, int count, int type);
#endif

}  // namespace Kernel
