#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  explicit ScaleOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ScaleOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  bool bias_term_;
  int axis_, num_axis_, scale_dim_, inner_dim_, bias_param_id_;

  BlobF *scale_ = nullptr, *bias_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
