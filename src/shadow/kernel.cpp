#include "shadow/kernel.hpp"

namespace Kernel {

#if defined(USE_CL)
#include <clBLAS.h>

EasyCL *easyCL = nullptr;

CLKernel *cl_channelmax_kernel_ = nullptr;
CLKernel *cl_channelsub_kernel_ = nullptr;
CLKernel *cl_channelsum_kernel_ = nullptr;
CLKernel *cl_channeldiv_kernel_ = nullptr;
CLKernel *cl_set_kernel_ = nullptr;
CLKernel *cl_add_kernel_ = nullptr;
CLKernel *cl_sub_kernel_ = nullptr;
CLKernel *cl_mul_kernel_ = nullptr;
CLKernel *cl_div_kernel_ = nullptr;
CLKernel *cl_sqr_kernel_ = nullptr;
CLKernel *cl_exp_kernel_ = nullptr;
CLKernel *cl_log_kernel_ = nullptr;
CLKernel *cl_abs_kernel_ = nullptr;
CLKernel *cl_pow_kernel_ = nullptr;

CLKernel *cl_datatransform_kernel_ = nullptr;
CLKernel *cl_im2col_kernel_ = nullptr;
CLKernel *cl_pooling_kernel_ = nullptr;
CLKernel *cl_concat_kernel_ = nullptr;
CLKernel *cl_permute_kernel_ = nullptr;
CLKernel *cl_activate_kernel_ = nullptr;

void Setup(int device_id) {
  easyCL = EasyCL::createForFirstGpuOtherwiseCpu(true);

  std::string cl_blas = "./src/shadow/util/blas.cl";
  cl_channelmax_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "ChannelMax");
  cl_channelsub_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "ChannelSub");
  cl_channelsum_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "ChannelSum");
  cl_channeldiv_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "ChannelDiv");
  cl_set_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Set");
  cl_add_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Add");
  cl_sub_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Sub");
  cl_mul_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Mul");
  cl_div_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Div");
  cl_sqr_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Sqr");
  cl_exp_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Exp");
  cl_log_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Log");
  cl_abs_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Abs");
  cl_pow_kernel_ = Kernel::easyCL->buildKernel(cl_blas, "Pow");

  std::string cl_image = "src/shadow/util/image.cl";
  cl_datatransform_kernel_ = easyCL->buildKernel(cl_image, "DataTransform");
  cl_im2col_kernel_ = easyCL->buildKernel(cl_image, "Im2Col");
  cl_pooling_kernel_ = easyCL->buildKernel(cl_image, "Pooling");
  cl_concat_kernel_ = easyCL->buildKernel(cl_image, "Concat");
  cl_permute_kernel_ = easyCL->buildKernel(cl_image, "Permute");
  cl_activate_kernel_ = easyCL->buildKernel(cl_image, "Activate");

  clblasSetup();
}

void Release() {
  cl_channelmax_kernel_->~CLKernel();
  cl_channelsub_kernel_->~CLKernel();
  cl_channelsum_kernel_->~CLKernel();
  cl_channeldiv_kernel_->~CLKernel();
  cl_set_kernel_->~CLKernel();
  cl_add_kernel_->~CLKernel();
  cl_sub_kernel_->~CLKernel();
  cl_mul_kernel_->~CLKernel();
  cl_div_kernel_->~CLKernel();
  cl_sqr_kernel_->~CLKernel();
  cl_exp_kernel_->~CLKernel();
  cl_log_kernel_->~CLKernel();
  cl_abs_kernel_->~CLKernel();
  cl_pow_kernel_->~CLKernel();

  cl_datatransform_kernel_->~CLKernel();
  cl_im2col_kernel_->~CLKernel();
  cl_pooling_kernel_->~CLKernel();
  cl_concat_kernel_->~CLKernel();
  cl_permute_kernel_->~CLKernel();
  cl_activate_kernel_->~CLKernel();

  easyCL->~EasyCL();

  clblasTeardown();
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
#endif

}  // namespace Kernel
