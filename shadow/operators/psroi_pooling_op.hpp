#ifndef SHADOW_OPERATORS_PSROI_POOLING_OP_HPP
#define SHADOW_OPERATORS_PSROI_POOLING_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PSROIPoolingOp : public Operator {
 public:
  explicit PSROIPoolingOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~PSROIPoolingOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int output_dim_, group_size_, pooled_h_, pooled_w_;
  float spatial_scale_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PSROI_POOLING_OP_HPP
