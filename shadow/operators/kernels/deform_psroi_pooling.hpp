#ifndef SHADOW_OPERATORS_KERNELS_DEFORM_PSROI_POOLING_HPP_
#define SHADOW_OPERATORS_KERNELS_DEFORM_PSROI_POOLING_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void DeformPSROIPooling(const T* in_data, const VecInt& in_shape,
                        const T* roi_data, const T* trans_data,
                        const VecInt& trans_shape, int num_rois, int output_dim,
                        int group_size, int pooled_size, int part_size,
                        int sample_per_part, float spatial_scale,
                        float trans_std, bool no_trans, T* out_data,
                        Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class DeformPSROIPoolingKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& input,
                   const std::shared_ptr<Blob>& roi,
                   const std::shared_ptr<Blob>& trans,
                   std::shared_ptr<Blob>& output, Workspace* ws, int output_dim,
                   int group_size, int pooled_size, int part_size,
                   int sample_per_part, float spatial_scale, float trans_std,
                   bool no_trans) = 0;
};

template <DeviceType D>
class DeformPSROIPoolingKernelDefault : public DeformPSROIPoolingKernel {
 public:
  void Run(const std::shared_ptr<Blob>& input, const std::shared_ptr<Blob>& roi,
           const std::shared_ptr<Blob>& trans, std::shared_ptr<Blob>& output,
           Workspace* ws, int output_dim, int group_size, int pooled_size,
           int part_size, int sample_per_part, float spatial_scale,
           float trans_std, bool no_trans) override {
    const auto* trans_data = no_trans ? nullptr : trans->data<float>();
    const auto& trans_shape = no_trans ? VecInt{} : trans->shape();

    Vision::DeformPSROIPooling<D, float>(
        input->data<float>(), input->shape(), roi->data<float>(), trans_data,
        trans_shape, roi->shape(0), output_dim, group_size, pooled_size,
        part_size, sample_per_part, spatial_scale, trans_std, no_trans,
        output->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_DEFORM_PSROI_POOLING_HPP_
