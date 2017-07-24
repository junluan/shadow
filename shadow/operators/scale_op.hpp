#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  explicit ScaleOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ScaleOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  bool bias_term_;
  int axis_, num_axis_, scale_dim_, inner_dim_, bias_param_id_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
