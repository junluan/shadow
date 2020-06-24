#ifndef SHADOW_OPERATORS_KERNELS_GATHER_HPP_
#define SHADOW_OPERATORS_KERNELS_GATHER_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Gather(const T* in_data, const int* indexes_data, int num_indexes,
            int gather_dim, int inner_num, int count, T* out_data,
            Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class GatherKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& indexes,
                   std::shared_ptr<Blob>& output, Workspace* ws, int axis) = 0;
};

template <DeviceType D>
class GatherKernelDefault : public GatherKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& indexes, std::shared_ptr<Blob>& output,
           Workspace* ws, int axis) override {
    Vision::Gather<D, float>(input->data<float>(), indexes->data<int>(),
                             indexes->count(), input->shape(axis),
                             input->count(axis + 1), output->count(),
                             output->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_GATHER_HPP_
