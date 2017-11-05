#ifndef SHADOW_OPERATORS_INPUT_OP_HPP
#define SHADOW_OPERATORS_INPUT_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class InputOp : public Operator {
 public:
  explicit InputOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}

  void Reshape() override;
  void Forward() override;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_INPUT_OP_HPP
