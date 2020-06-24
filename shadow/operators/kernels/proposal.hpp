#ifndef SHADOW_OPERATORS_KERNELS_PROPOSAL_HPP_
#define SHADOW_OPERATORS_KERNELS_PROPOSAL_HPP_

#include "core/kernel.hpp"

namespace Shadow {

namespace Vision {

template <DeviceType D, typename T>
void Proposal(const T* anchor_data, const T* score_data, const T* delta_data,
              const T* info_data, const VecInt& in_shape, int num_anchors,
              int feat_stride, int min_size, T* proposal_data,
              Context* context);

}  // namespace Vision

}  // namespace Shadow

namespace Shadow {

class ProposalKernel : public Kernel {
 public:
  virtual void Run(const std::shared_ptr<Blob>& anchors,
                   const std::shared_ptr<Blob>& score,
                   const std::shared_ptr<Blob>& delta,
                   const std::shared_ptr<Blob>& info,
                   std::shared_ptr<Blob>& proposals, Workspace* ws,
                   int feat_stride, int min_size, int num_anchors) = 0;
};

template <DeviceType D>
class ProposalKernelDefault : public ProposalKernel {
 public:
  void Run(const std::shared_ptr<Blob>& anchors,
           const std::shared_ptr<Blob>& score,
           const std::shared_ptr<Blob>& delta,
           const std::shared_ptr<Blob>& info, std::shared_ptr<Blob>& proposals,
           Workspace* ws, int feat_stride, int min_size,
           int num_anchors) override {
    Vision::Proposal<D, float>(
        anchors->data<float>(), score->data<float>(), delta->data<float>(),
        info->data<float>(), score->shape(), num_anchors, feat_stride, min_size,
        proposals->mutable_data<float>(), ws->Ctx());
  }

  DeviceType device_type() const override { return D; }

  std::string kernel_type() const override { return "Default"; }
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_KERNELS_PROPOSAL_HPP_
