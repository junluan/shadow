#ifndef SHADOW_OPERATORS_KERNELS_ACTIVATE_HPP_
#define SHADOW_OPERATORS_KERNELS_ACTIVATE_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Activate(const T* in_data, T* out_data, int count, int type, float slope,
              Context* context);

template <DeviceType D, typename T>
void PRelu(const T* in_data, T* out_data, const VecInt& in_shape,
           bool channel_shared, const T* slope_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

enum {
  kPRelu = 0,
  kRelu = 1,
  kLeaky = 2,
  kSigmoid = 3,
  kSoftPlus = 4,
  kTanh = 5,
  kRelu6 = 6,
  kHardSwish = 7
};

class ActivateKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& slope,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   int activate_type, float slope_val) = 0;
};

template <DeviceType D>
class ActivateKernelDefault : public ActivateKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input,
           const std::shared_ptr<Blob>& slope, std::shared_ptr<Blob>& output,
           Workspace* ws, int activate_type, float slope_val) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    if (activate_type == kRelu || activate_type == kLeaky ||
        activate_type == kSigmoid || activate_type == kSoftPlus ||
        activate_type == kTanh || activate_type == kRelu6 ||
        activate_type == kHardSwish) {
      Vision::Activate<D, float>(in_data, out_data, output->count(),
                                 activate_type, slope_val, ws->Ctx().get());
    } else if (activate_type == kPRelu) {
      CHECK_GE(input->num_axes(), 2);
      bool channel_shared = slope->count() == 1;
      if (!channel_shared) {
        CHECK_EQ(slope->count(), input->shape(1));
      }
      Vision::PRelu<D, float>(in_data, out_data, output->shape(),
                              channel_shared, slope->data<float>(),
                              ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_ACTIVATE_HPP_
