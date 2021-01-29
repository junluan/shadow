#ifndef SHADOW_OPERATORS_KERNELS_PSROI_ALIGN_HPP_
#define SHADOW_OPERATORS_KERNELS_PSROI_ALIGN_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void PSROIAlign(const T* in_data, const VecInt& in_shape, const T* roi_data,
                int num_rois, int out_c, int pooled_h, int pooled_w,
                float spatial_scale, int sampling_ratio, T* out_data,
                Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class PSROIAlignKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& roi,
                   std::shared_ptr<Blob>& output, Workspace* ws,
                   float spatial_scale, int sampling_ratio) = 0;
};

template <DeviceType D>
class PSROIAlignKernelDefault : public PSROIAlignKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, const std::shared_ptr<Blob>& roi,
           std::shared_ptr<Blob>& output, Workspace* ws, float spatial_scale,
           int sampling_ratio) override {
    Vision::PSROIAlign<D, float>(
        input->data<float>(), input->shape(), roi->data<float>(), roi->shape(0),
        output->shape(1), output->shape(2), output->shape(3), spatial_scale,
        sampling_ratio, output->mutable_data<float>(), ws->Ctx().get());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_PSROI_ALIGN_HPP_
