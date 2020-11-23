#ifndef SHADOW_OPERATORS_KERNELS_LRN_HPP_
#define SHADOW_OPERATORS_KERNELS_LRN_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void LRN(const T* in_data, const VecInt& in_shape, int size, float alpha,
         float beta, float k, T* scale_data, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class LRNKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int size,
                   float alpha, float beta, float k) = 0;
};

template <DeviceType D>
class LRNKernelDefault : public LRNKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int size, float alpha, float beta, float k) override {
    ws->GrowTempBuffer(input->raw_size());

    auto scale = ws->CreateTempBlob(input->shape(), DataType::kF32);

    Vision::LRN<D, float>(input->data<float>(), input->shape(), size, alpha,
                          beta, k, scale->mutable_data<float>(),
                          output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_LRN_HPP_
