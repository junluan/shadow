#ifndef SHADOW_OPERATORS_KERNELS_PAD_HPP_
#define SHADOW_OPERATORS_KERNELS_PAD_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Pad(const T* in_data, const VecInt& in_shape, const VecInt& paddings,
         const VecInt& out_shape, T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class PadKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   const VecInt& paddings, float value) = 0;
};

template <DeviceType D>
class PadKernelDefault : public PadKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, const VecInt& paddings, float value) override {
    const auto* in_data = input->data<float>();
    auto* out_data = output->mutable_data<float>();

    if (paddings[0] == 0 && paddings[1] == 0 && paddings[2] == 0 &&
        paddings[3] == 0) {
      Blas::BlasScopy<D, float>(input->count(), in_data, 0, out_data, 0,
                                ws->Ctx().get());
    } else {
      Blas::Set<D, float>(output->count(), value, out_data, 0, ws->Ctx().get());
      Vision::Pad<D, float>(in_data, input->shape(), paddings, output->shape(),
                            out_data, ws->Ctx().get());
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_PAD_HPP_
