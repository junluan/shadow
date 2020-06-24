#ifndef SHADOW_OPERATORS_KERNELS_POOLING_HPP_
#define SHADOW_OPERATORS_KERNELS_POOLING_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Pooling(const T* in_data, const VecInt& in_shape, int pool_type,
             int kernel_size_h, int kernel_size_w, int stride_h, int stride_w,
             int pad_h, int pad_w, const VecInt& out_shape, T* out_data,
             Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class PoolingKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int pool_type,
                   int kernel_size_h, int kernel_size_w, int stride_h,
                   int stride_w, int pad_h, int pad_w, bool full_pooling) = 0;
};

template <DeviceType D>
class PoolingKernelDefault : public PoolingKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int pool_type, int kernel_size_h, int kernel_size_w,
           int stride_h, int stride_w, int pad_h, int pad_w,
           bool full_pooling) override {
    Vision::Pooling<D, float>(input->data<float>(), input->shape(), pool_type,
                              kernel_size_h, kernel_size_w, stride_h, stride_w,
                              pad_h, pad_w, output->shape(),
                              output->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_POOLING_HPP_
