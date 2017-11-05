#ifndef SHADOW_OPERATORS_ROI_POOLING_OP_HPP
#define SHADOW_OPERATORS_ROI_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ROIPoolingOp : public Operator {
 public:
  explicit ROIPoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    pooled_h_ = get_single_argument<int>("pooled_h", 0);
    pooled_w_ = get_single_argument<int>("pooled_w", 0);
    CHECK_GT(pooled_h_, 0) << "pooled_h must be > 0";
    CHECK_GT(pooled_w_, 0) << "pooled_w must be > 0";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
    CHECK_EQ(bottoms_size(), 2);
  }

  void Reshape() override;
  void Forward() override;

 private:
  int pooled_h_, pooled_w_;
  float spatial_scale_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ROI_POOLING_OP_HPP
