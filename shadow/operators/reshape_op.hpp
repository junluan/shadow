#ifndef SHADOW_OPERATORS_RESHAPE_OP_HPP
#define SHADOW_OPERATORS_RESHAPE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  explicit ReshapeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ReshapeOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int axis_, num_axes_, inferred_axis_, constant_count_;
  VecInt shape_, copy_axes_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESHAPE_OP_HPP
