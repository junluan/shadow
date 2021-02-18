#ifndef SHADOW_OPERATORS_KERNELS_RESIZE_HPP_
#define SHADOW_OPERATORS_KERNELS_RESIZE_HPP_

#include "core/blas.hpp"
#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void ResizeNearest2D(const T* in_data, const VecInt& in_shape,
                     const VecInt& out_shape, T* out_data, Context* context);

template <DeviceType D, typename T>
void ResizeBilinear2D(const T* in_data, const VecInt& in_shape,
                      bool align_corners, const VecInt& out_shape, T* out_data,
                      Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ResizeKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   std::shared_ptr<Blob>& output, Workspace* ws, int type,
                   bool align_corners) = 0;
};

template <DeviceType D>
class ResizeKernelDefault : public ResizeKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, std::shared_ptr<Blob>& output,
           Workspace* ws, int type, bool align_corners) override {
    int in_h = input->shape(2), in_w = input->shape(3);
    int out_h = output->shape(2), out_w = output->shape(3);

    if (out_h == in_h && out_w == in_w) {
      Blas::BlasScopy<D, float>(input->count(), input->data<float>(), 0,
                                output->mutable_data<float>(), 0,
                                ws->Ctx().get());
    } else {
      if (type == 0) {
        Vision::ResizeNearest2D<D, float>(
            input->data<float>(), input->shape(), output->shape(),
            output->mutable_data<float>(), ws->Ctx().get());
      } else {
        Vision::ResizeBilinear2D<D, float>(
            input->data<float>(), input->shape(), align_corners,
            output->shape(), output->mutable_data<float>(), ws->Ctx().get());
      }
    }
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_RESIZE_HPP_
