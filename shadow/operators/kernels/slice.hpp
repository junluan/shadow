#ifndef SHADOW_OPERATORS_KERNELS_SLICE_HPP_
#define SHADOW_OPERATORS_KERNELS_SLICE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Slice(const T* in_data, int count, int num_slices, int slice_size,
           int in_slice_axis, int out_slice_axis, int offset_slice_axis,
           T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class SliceKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::vector<std::shared_ptr<Blob>>& outputs, Workspace* ws,
                   int axis) = 0;
};

template <DeviceType D>
class SliceKernelDefault : public SliceKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           std::vector<std::shared_ptr<Blob>>& outputs, Workspace* ws,
           int axis) override {
    int offset_slice_axis = 0;
    int num_slices = input->count(0, axis);
    int slice_size = input->count(axis + 1);
    int in_slice_axis = input->shape(axis);
    for (auto& output : outputs) {
      int out_slice_axis = output->shape(axis);
      Vision::Slice<D, float>(input->data<float>(), output->count(), num_slices,
                              slice_size, in_slice_axis, out_slice_axis,
                              offset_slice_axis, output->mutable_data<float>(),
                              ws->Ctx().get());
      offset_slice_axis += out_slice_axis;
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_SLICE_HPP_
