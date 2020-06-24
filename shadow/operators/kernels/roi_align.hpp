#ifndef SHADOW_OPERATORS_KERNELS_ROI_ALIGN_HPP_
#define SHADOW_OPERATORS_KERNELS_ROI_ALIGN_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void ROIAlign(const T* in_data, const VecInt& in_shape, const T* roi_data,
              int num_rois, int pooled_h, int pooled_w, float spatial_scale,
              T* out_data, Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ROIAlignKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& roi,
                   std::shared_ptr<Blob>& output, Workspace* ws, int pooled_h,
                   int pooled_w, float spatial_scale) = 0;
};

template <DeviceType D>
class ROIAlignKernelDefault : public ROIAlignKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, const std::shared_ptr<Blob>& roi,
           std::shared_ptr<Blob>& output, Workspace* ws, int pooled_h,
           int pooled_w, float spatial_scale) override {
    Vision::ROIAlign<D, float>(input->data<float>(), input->shape(),
                               roi->data<float>(), roi->shape(0), pooled_h,
                               pooled_w, spatial_scale,
                               output->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_ROI_ALIGN_HPP_
