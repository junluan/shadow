#ifndef SHADOW_OPERATORS_KERNELS_SOFTMAX_HPP_
#define SHADOW_OPERATORS_KERNELS_SOFTMAX_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Softmax(const T* in_data, int outer_num, int dim, int inner_num,
             T* val_data, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class SoftmaxKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis) = 0;
};

template <DeviceType D>
class SoftmaxKernelDefault : public SoftmaxKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis) override {
    int outer_num = input->count(0, axis), dim = input->shape(axis),
        inner_num = input->count(axis + 1);

    ws->GrowTempBuffer(outer_num * inner_num * sizeof(float));

    auto scalar = ws->CreateTempBlob({outer_num, inner_num}, DataType::kF32);

    Vision::Softmax<D, float>(input->data<float>(), outer_num, dim, inner_num,
                              scalar->mutable_data<float>(),
                              output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_SOFTMAX_HPP_
