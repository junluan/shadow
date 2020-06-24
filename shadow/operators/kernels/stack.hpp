#ifndef SHADOW_OPERATORS_KERNELS_STACK_HPP_
#define SHADOW_OPERATORS_KERNELS_STACK_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Stack(const T* in_data, int count, int num_stacks, int stack_size,
           int out_stack_axis, int offset_stack_axis, T* out_data,
           Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class StackKernel : public Kernel {
 public:
  virtual void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis) = 0;
};

template <DeviceType D>
class StackKernelDefault : public StackKernel {
 public:
  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::shared_ptr<Blob>& output, Workspace* ws, int axis) override {
    int num_stacks = output->count(0, axis);
    int stack_size = output->count(axis + 1);
    int out_stack_axis = output->shape(axis);
    for (int n = 0; n < inputs.size(); ++n) {
      Vision::Stack<D, float>(inputs[n]->data<float>(), inputs[n]->count(),
                              num_stacks, stack_size, out_stack_axis, n,
                              output->mutable_data<float>(), ws->Ctx());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_STACK_HPP_
