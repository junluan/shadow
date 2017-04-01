#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

namespace Kernel {

#if defined(USE_CL)
#include <clBLAS.h>

EasyCL::Device *device_ = nullptr;
EasyCL::Context *context_ = nullptr;
EasyCL::Queue *queue_ = nullptr;
EasyCL::Event *event_ = nullptr;

EasyCL::KernelSet cl_kernels_ = {};

void Setup(int device_id) {
  device_ = EasyCL::CreateForIndexedGPU(device_id);
  context_ = new EasyCL::Context(*device_);
  queue_ = new EasyCL::Queue(*context_, *device_);
  event_ = new EasyCL::Event();

  auto compiler_options = std::vector<std::string>{};

  const std::string cl_blas = "src/shadow/util/blas.cl";
  auto program_blas =
      EasyCL::Program(*context_, Util::read_text_from_file(cl_blas));
  program_blas.Build(*device_, compiler_options);

  cl_kernels_.set_kernel(program_blas, "ChannelMax");
  cl_kernels_.set_kernel(program_blas, "ChannelSub");
  cl_kernels_.set_kernel(program_blas, "ChannelSum");
  cl_kernels_.set_kernel(program_blas, "ChannelDiv");
  cl_kernels_.set_kernel(program_blas, "Set");
  cl_kernels_.set_kernel(program_blas, "AddScalar");
  cl_kernels_.set_kernel(program_blas, "Add");
  cl_kernels_.set_kernel(program_blas, "Sub");
  cl_kernels_.set_kernel(program_blas, "Mul");
  cl_kernels_.set_kernel(program_blas, "Div");
  cl_kernels_.set_kernel(program_blas, "Sqr");
  cl_kernels_.set_kernel(program_blas, "Exp");
  cl_kernels_.set_kernel(program_blas, "Log");
  cl_kernels_.set_kernel(program_blas, "Abs");
  cl_kernels_.set_kernel(program_blas, "Pow");

  const std::string cl_image = "src/shadow/util/image.cl";
  auto program_image =
      EasyCL::Program(*context_, Util::read_text_from_file(cl_image));
  program_image.Build(*device_, compiler_options);

  cl_kernels_.set_kernel(program_image, "DataTransform");
  cl_kernels_.set_kernel(program_image, "Im2Col");
  cl_kernels_.set_kernel(program_image, "Pooling");
  cl_kernels_.set_kernel(program_image, "Concat");
  cl_kernels_.set_kernel(program_image, "Permute");
  cl_kernels_.set_kernel(program_image, "Scale");
  cl_kernels_.set_kernel(program_image, "Bias");
  cl_kernels_.set_kernel(program_image, "Reorg");
  cl_kernels_.set_kernel(program_image, "LRN");
  cl_kernels_.set_kernel(program_image, "LRNFillScale");
  cl_kernels_.set_kernel(program_image, "Activate");
  cl_kernels_.set_kernel(program_image, "PRelu");

  clblasSetup();
}

void Release() {
  delete device_;
  delete context_;
  delete queue_;
  delete event_;

  clblasTeardown();
}

template <typename T, typename Dtype>
T *MakeBuffer(int size, Dtype *host_ptr) {
  return new EasyCL::Buffer<Dtype>(*Kernel::context_, size);
}

template <typename T, typename Dtype>
void ReadBuffer(int size, const T *src, Dtype *des) {
  src->Read(*Kernel::queue_, size, des);
}

template <typename T, typename Dtype>
void WriteBuffer(int size, const Dtype *src, T *des) {
  des->Write(*Kernel::queue_, size, src);
}

template <typename T, typename Dtype>
void CopyBuffer(int size, const T *src, T *des) {
  src->CopyTo(*Kernel::queue_, size, *des);
}

template <typename T>
void ReleaseBuffer(T *buffer) {
  delete buffer;
}

// Explicit instantiation
template BufferI *MakeBuffer(int size, int *host_ptr);
template BufferF *MakeBuffer(int size, float *host_ptr);

template void ReadBuffer(int size, const BufferI *src, int *des);
template void ReadBuffer(int size, const BufferF *src, float *des);

template void WriteBuffer(int size, const int *src, BufferI *des);
template void WriteBuffer(int size, const float *src, BufferF *des);

template void CopyBuffer(int size, const BufferI *src, BufferI *des);
template void CopyBuffer(int size, const BufferF *src, BufferF *des);

template void ReleaseBuffer(BufferI *buffer);
template void ReleaseBuffer(BufferF *buffer);
#endif

}  // namespace Kernel
