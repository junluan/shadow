#ifndef SHADOW_OPERATORS_KERNELS_SHUFFLE_CHANNEL_HPP_
#define SHADOW_OPERATORS_KERNELS_SHUFFLE_CHANNEL_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void ShuffleChannel(const T* in_data, int batch, int channel, int spatial_dim,
                    int group, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ShuffleChannelKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int group) = 0;
};

template <DeviceType D>
class ShuffleChannelKernelDefault : public ShuffleChannelKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int group) override {
    Vision::ShuffleChannel<D, float>(
        input->data<float>(), input->shape(0), input->shape(1), input->count(2),
        group, output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_SHUFFLE_CHANNEL_HPP_
