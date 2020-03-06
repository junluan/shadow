#ifndef SHADOW_OPERATORS_ROI_ALIGN_OP_HPP
#define SHADOW_OPERATORS_ROI_ALIGN_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ROIAlignOp : public Operator {
 public:
  ROIAlignOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    pooled_h_ = get_single_argument<int>("pooled_h", 0);
    pooled_w_ = get_single_argument<int>("pooled_w", 0);
    CHECK_GT(pooled_h_, 1) << "pooled_h must be > 1";
    CHECK_GT(pooled_w_, 1) << "pooled_w must be > 1";
    spatial_scale_ = get_single_argument<float>("spatial_scale", 1.f / 16);
  }

  void Forward() override;

 private:
  int pooled_h_, pooled_w_;
  float spatial_scale_;
};

namespace Vision {

template <typename T>
void ROIAlign(const T *in_data, const VecInt &in_shape, const T *roi_data,
              int num_rois, int pooled_h, int pooled_w, float spatial_scale,
              T *out_data, Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ROI_ALIGN_OP_HPP
