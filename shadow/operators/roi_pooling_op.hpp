#ifndef SHADOW_OPERATORS_ROI_POOLING_OP_HPP
#define SHADOW_OPERATORS_ROI_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ROIPoolingOp : public Operator {
 public:
  explicit ROIPoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ROIPoolingOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int pooled_h_, pooled_w_;
  float spatial_scale_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ROI_POOLING_OP_HPP
