#ifndef SHADOW_OPERATORS_KERNELS_PERMUTE_HPP_
#define SHADOW_OPERATORS_KERNELS_PERMUTE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Permute(const T* in_data, int count, int num_axes,
             const int* permute_order, const int* old_steps,
             const int* new_steps, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class PermuteKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   const VecInt& order_value) = 0;
};

template <DeviceType D>
class PermuteKernelDefault : public PermuteKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& order_value) override {
    int num_axes = input->num_axes();

    VecInt old_steps_value(num_axes), new_steps_value(num_axes);
    for (int d = 0; d < num_axes; ++d) {
      if (d == num_axes - 1) {
        old_steps_value[d] = 1;
        new_steps_value[d] = 1;
      } else {
        old_steps_value[d] = input->count(d + 1);
        new_steps_value[d] = output->count(d + 1);
      }
    }

    ws->GrowTempBuffer(3 * num_axes * sizeof(int));

    auto order = ws->CreateTempBlob({num_axes}, DataType::kI32);
    auto old_steps = ws->CreateTempBlob({num_axes}, DataType::kI32);
    auto new_steps = ws->CreateTempBlob({num_axes}, DataType::kI32);

    order->set_data<int>(order_value.data(), num_axes);
    old_steps->set_data<int>(old_steps_value.data(), num_axes);
    new_steps->set_data<int>(new_steps_value.data(), num_axes);

    Vision::Permute<D, float>(input->data<float>(), input->count(),
                              input->num_axes(), order->data<int>(),
                              old_steps->data<int>(), new_steps->data<int>(),
                              output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_PERMUTE_HPP_
