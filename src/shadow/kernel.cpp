#include "shadow/kernel.hpp"
#include "shadow/util/util.hpp"

namespace Kernel {

#if defined(USE_CL)
#include <clBLAS.h>

EasyCL::Device *device_ = nullptr;
EasyCL::Context *context_ = nullptr;
EasyCL::Queue *queue_ = nullptr;
EasyCL::Event *event_ = nullptr;

EasyCL::Kernel *cl_channelmax_kernel_ = nullptr;
EasyCL::Kernel *cl_channelsub_kernel_ = nullptr;
EasyCL::Kernel *cl_channelsum_kernel_ = nullptr;
EasyCL::Kernel *cl_channeldiv_kernel_ = nullptr;
EasyCL::Kernel *cl_set_kernel_ = nullptr;
EasyCL::Kernel *cl_addscalar_kernel_ = nullptr;
EasyCL::Kernel *cl_add_kernel_ = nullptr;
EasyCL::Kernel *cl_sub_kernel_ = nullptr;
EasyCL::Kernel *cl_mul_kernel_ = nullptr;
EasyCL::Kernel *cl_div_kernel_ = nullptr;
EasyCL::Kernel *cl_sqr_kernel_ = nullptr;
EasyCL::Kernel *cl_exp_kernel_ = nullptr;
EasyCL::Kernel *cl_log_kernel_ = nullptr;
EasyCL::Kernel *cl_abs_kernel_ = nullptr;
EasyCL::Kernel *cl_pow_kernel_ = nullptr;

EasyCL::Kernel *cl_datatransform_kernel_ = nullptr;
EasyCL::Kernel *cl_im2col_kernel_ = nullptr;
EasyCL::Kernel *cl_pooling_kernel_ = nullptr;
EasyCL::Kernel *cl_concat_kernel_ = nullptr;
EasyCL::Kernel *cl_permute_kernel_ = nullptr;
EasyCL::Kernel *cl_scale_kernel_ = nullptr;
EasyCL::Kernel *cl_bias_kernel_ = nullptr;
EasyCL::Kernel *cl_reorg_kernel_ = nullptr;
EasyCL::Kernel *cl_activate_kernel_ = nullptr;

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

  cl_channelmax_kernel_ = new EasyCL::Kernel(program_blas, "ChannelMax");
  cl_channelsub_kernel_ = new EasyCL::Kernel(program_blas, "ChannelSub");
  cl_channelsum_kernel_ = new EasyCL::Kernel(program_blas, "ChannelSum");
  cl_channeldiv_kernel_ = new EasyCL::Kernel(program_blas, "ChannelDiv");
  cl_set_kernel_ = new EasyCL::Kernel(program_blas, "Set");
  cl_addscalar_kernel_ = new EasyCL::Kernel(program_blas, "AddScalar");
  cl_add_kernel_ = new EasyCL::Kernel(program_blas, "Add");
  cl_sub_kernel_ = new EasyCL::Kernel(program_blas, "Sub");
  cl_mul_kernel_ = new EasyCL::Kernel(program_blas, "Mul");
  cl_div_kernel_ = new EasyCL::Kernel(program_blas, "Div");
  cl_sqr_kernel_ = new EasyCL::Kernel(program_blas, "Sqr");
  cl_exp_kernel_ = new EasyCL::Kernel(program_blas, "Exp");
  cl_log_kernel_ = new EasyCL::Kernel(program_blas, "Log");
  cl_abs_kernel_ = new EasyCL::Kernel(program_blas, "Abs");
  cl_pow_kernel_ = new EasyCL::Kernel(program_blas, "Pow");

  const std::string cl_image = "src/shadow/util/image.cl";
  auto program_image =
      EasyCL::Program(*context_, Util::read_text_from_file(cl_image));
  program_image.Build(*device_, compiler_options);

  cl_datatransform_kernel_ = new EasyCL::Kernel(program_image, "DataTransform");
  cl_im2col_kernel_ = new EasyCL::Kernel(program_image, "Im2Col");
  cl_pooling_kernel_ = new EasyCL::Kernel(program_image, "Pooling");
  cl_concat_kernel_ = new EasyCL::Kernel(program_image, "Concat");
  cl_permute_kernel_ = new EasyCL::Kernel(program_image, "Permute");
  cl_scale_kernel_ = new EasyCL::Kernel(program_image, "Scale");
  cl_bias_kernel_ = new EasyCL::Kernel(program_image, "Bias");
  cl_reorg_kernel_ = new EasyCL::Kernel(program_image, "Reorg");
  cl_activate_kernel_ = new EasyCL::Kernel(program_image, "Activate");

  clblasSetup();
}

void Release() {
  delete cl_channelmax_kernel_;
  delete cl_channelsub_kernel_;
  delete cl_channelsum_kernel_;
  delete cl_channeldiv_kernel_;
  delete cl_set_kernel_;
  delete cl_addscalar_kernel_;
  delete cl_add_kernel_;
  delete cl_sub_kernel_;
  delete cl_mul_kernel_;
  delete cl_div_kernel_;
  delete cl_sqr_kernel_;
  delete cl_exp_kernel_;
  delete cl_log_kernel_;
  delete cl_abs_kernel_;
  delete cl_pow_kernel_;

  delete cl_datatransform_kernel_;
  delete cl_im2col_kernel_;
  delete cl_pooling_kernel_;
  delete cl_concat_kernel_;
  delete cl_permute_kernel_;
  delete cl_scale_kernel_;
  delete cl_bias_kernel_;
  delete cl_reorg_kernel_;
  delete cl_activate_kernel_;

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
