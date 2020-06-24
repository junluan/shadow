#ifndef SHADOW_OPERATORS_KERNELS_CONCAT_HPP_
#define SHADOW_OPERATORS_KERNELS_CONCAT_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Concat(const T* in_data, int count, int num_concats, int concat_size,
            int out_concat_axis, int in_concat_axis, int offset_concat_axis,
            T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ConcatKernel : public Kernel {
 public:
  virtual void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis) = 0;
};

template <DeviceType D>
class ConcatKernelDefault : public ConcatKernel {
 public:
  void Run(const std::vector<std::shared_ptr<Blob>>& inputs,
           std::shared_ptr<Blob>& output, Workspace* ws, int axis) override {
    int offset_concat_axis = 0;
    int num_concats = output->count(0, axis);
    int concat_size = output->count(axis + 1);
    int out_concat_axis = output->shape(axis);
    for (const auto& input : inputs) {
      int in_concat_axis = input->shape(axis);
      Vision::Concat<D, float>(input->data<float>(), input->count(),
                               num_concats, concat_size, out_concat_axis,
                               in_concat_axis, offset_concat_axis,
                               output->mutable_data<float>(), ws->Ctx());
      offset_concat_axis += in_concat_axis;
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_CONCAT_HPP_
