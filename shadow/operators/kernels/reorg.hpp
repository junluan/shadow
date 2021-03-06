#ifndef SHADOW_OPERATORS_KERNELS_REORG_HPP_
#define SHADOW_OPERATORS_KERNELS_REORG_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Reorg(const T* in_data, const VecInt& in_shape, int type, int stride,
           T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

enum { kDarknet = 0, kNatural = 1 };

class ReorgKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int type,
                   int stride) = 0;
};

template <DeviceType D>
class ReorgKernelDefault : public ReorgKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int type, int stride) override {
    Vision::Reorg<D, float>(input->data<float>(), input->shape(), type, stride,
                            output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_REORG_HPP_
