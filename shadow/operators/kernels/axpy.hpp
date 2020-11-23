#ifndef SHADOW_OPERATORS_KERNELS_AXPY_HPP_
#define SHADOW_OPERATORS_KERNELS_AXPY_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Axpy(const float* alpha_data, const float* x_data, const float* y_data,
          int outer_num, int inner_num, float* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class AxpyKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& alpha,
                   const std::shared_ptr<Blob>& x,
                   const std::shared_ptr<Blob>& y,
                   std::shared_ptr<Blob>& output, Workspace* ws) = 0;
};

template <DeviceType D>
class AxpyKernelDefault : public AxpyKernel {
 public:
  void Run(const std::shared_ptr<Blob>& alpha, const std::shared_ptr<Blob>& x,
           const std::shared_ptr<Blob>& y, std::shared_ptr<Blob>& output,
           Workspace* ws) override {
    Vision::Axpy<D, float>(alpha->data<float>(), x->data<float>(),
                           y->data<float>(), alpha->count(), x->count(2),
                           output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_AXPY_HPP_
