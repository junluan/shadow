#ifndef SHADOW_OPERATORS_PROPOSAL_OP_HPP
#define SHADOW_OPERATORS_PROPOSAL_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

static inline VecFloat generate_anchors(int base_size, const VecFloat &ratios,
                                        const VecFloat &scales) {
  auto num_ratio = static_cast<int>(ratios.size());
  auto num_scale = static_cast<int>(scales.size());
  VecFloat anchors(4 * num_ratio * num_scale);
  float cx = (base_size - 1) * 0.5f, cy = (base_size - 1) * 0.5f;
  for (int i = 0; i < num_ratio; ++i) {
    float ar = ratios[i];
    float r_w = Util::round(std::sqrt(base_size * base_size / ar));
    float r_h = Util::round(r_w * ar);
    for (int j = 0; j < num_scale; ++j) {
      float scale = scales[j];
      float rs_w = r_w * scale, rs_h = r_h * scale;
      float *anchor = anchors.data() + (i * num_scale + j) * 4;
      anchor[0] = cx - (rs_w - 1) * 0.5f;
      anchor[1] = cy - (rs_h - 1) * 0.5f;
      anchor[2] = cx + (rs_w - 1) * 0.5f;
      anchor[3] = cy + (rs_h - 1) * 0.5f;
    }
  }
  return anchors;
}

class ProposalOp : public Operator {
 public:
  explicit ProposalOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    feat_stride_ = get_single_argument<int>("feat_stride", 16);
    pre_nms_top_n_ = get_single_argument<int>("pre_nms_top_n", 6000);
    post_nms_top_n_ = get_single_argument<int>("post_nms_top_n", 300);
    min_size_ = get_single_argument<int>("min_size", 16);
    nms_thresh_ = get_single_argument<float>("nms_thresh", 0.7f);
    ratios_ = get_repeated_argument<float>("ratios", {0.5f, 1.f, 2.f});
    scales_ = get_repeated_argument<float>("scales", {8.f, 16.f, 32.f});
    anchors_ = generate_anchors(16, ratios_, scales_);
    num_anchors_ = static_cast<int>(ratios_.size() * scales_.size());
  }

  void Forward() override;

 private:
  int feat_stride_, pre_nms_top_n_, post_nms_top_n_, min_size_, num_anchors_;
  float nms_thresh_;
  VecFloat ratios_, scales_, anchors_, selected_rois_;
};

namespace Vision {

template <typename T>
void Proposal(const T *anchor_data, const T *score_data, const T *delta_data,
              const T *info_data, const VecInt &in_shape, int num_anchors,
              int feat_stride, int min_size, T *proposal_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PROPOSAL_OP_HPP
