#ifndef SHADOW_OPERATORS_PROPOSAL_OP_HPP
#define SHADOW_OPERATORS_PROPOSAL_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ProposalOp : public Operator {
 public:
  explicit ProposalOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ProposalOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int feat_stride_, pre_nms_topN_, post_nms_topN_, min_size_, base_size_,
      num_anchors_;
  float nms_thresh_;
  VecFloat ratios_, scales_, selected_rois_;

  BlobF *anchors_ = nullptr, *proposals_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PROPOSAL_OP_HPP
