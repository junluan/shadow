#include "kernel.hpp"

#include "util/log.hpp"
#include "util/util.hpp"

#if defined(USE_CL)
#include "clBLAS.h"
#endif

namespace Shadow {

namespace Kernel {

#if !defined(USE_CUDA) & !defined(USE_CL)
#if defined(USE_NNPACK)
pthreadpool_t nnp_pthreadpool_ = nullptr;
#endif

void Setup(int device_id) {
#if defined(USE_NNPACK)
  if (nnp_pthreadpool_ == nullptr) {
    CHECK_EQ(nnp_initialize(), nnp_status_success);
    nnp_pthreadpool_ = pthreadpool_create(NumThreads);
    CHECK_NOTNULL(nnp_pthreadpool_);
  }
#endif
}

void Release() {
#if defined(USE_NNPACK)
  if (nnp_pthreadpool_ != nullptr) {
    CHECK_EQ(nnp_deinitialize(), nnp_status_success);
    pthreadpool_destroy(nnp_pthreadpool_);
    nnp_pthreadpool_ = nullptr;
  }
#endif
}

void Synchronize() {}

#elif defined(USE_CL)
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

  const std::string cl_blas("shadow/core/blas.cl");
  auto program_blas =
      EasyCL::Program(*context_, Util::read_text_from_file(cl_blas));
  program_blas.Build(*device_, compiler_options);

  const std::vector<std::string> cl_blas_kernels{
      "ChannelMax", "ChannelSub", "ChannelSum", "ChannelDiv", "Set",
      "Add",        "Sub",        "Mul",        "Div",        "Pow",
      "Max",        "Min",        "AddScalar",  "SubScalar",  "MulScalar",
      "DivScalar",  "PowScalar",  "MaxScalar",  "MinScalar",  "Abs",
      "Square",     "Sqrt",       "Log",        "Exp",        "Sin",
      "Cos",        "Tan",        "Asin",       "Acos",       "Atan",
      "Floor",      "Ceil"};

  cl_kernels_.set_kernel(program_blas, cl_blas_kernels);

  const std::string cl_vision("shadow/core/vision.cl");
  auto program_vision =
      EasyCL::Program(*context_, Util::read_text_from_file(cl_vision));
  program_vision.Build(*device_, compiler_options);

  const std::vector<std::string> cl_vision_kernels{
      "Im2Col",   "Pooling",  "Concat",       "Permute", "Scale",
      "Bias",     "Reorg",    "LRNFillScale", "LRN",     "ROIPooling",
      "Proposal", "Activate", "PRelu"};

  cl_kernels_.set_kernel(program_vision, cl_vision_kernels);

  clblasSetup();
}

void Release() {
  delete device_;
  delete context_;
  delete queue_;
  delete event_;

  clblasTeardown();
}

void Synchronize() {}

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
template BufferUC *MakeBuffer(int size, unsigned char *host_ptr);

template void ReadBuffer(int size, const BufferI *src, int *des);
template void ReadBuffer(int size, const BufferF *src, float *des);
template void ReadBuffer(int size, const BufferUC *src, unsigned char *des);

template void WriteBuffer(int size, const int *src, BufferI *des);
template void WriteBuffer(int size, const float *src, BufferF *des);
template void WriteBuffer(int size, const unsigned char *src, BufferUC *des);

template void CopyBuffer(int size, const BufferI *src, BufferI *des);
template void CopyBuffer(int size, const BufferF *src, BufferF *des);
template void CopyBuffer(int size, const BufferUC *src, BufferUC *des);

template void ReleaseBuffer(BufferI *buffer);
template void ReleaseBuffer(BufferF *buffer);
template void ReleaseBuffer(BufferUC *buffer);
#endif

}  // namespace Kernel

}  // namespace Shadow
