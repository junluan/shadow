#ifndef SHADOW_OPERATORS_RESHAPE_OP_HPP
#define SHADOW_OPERATORS_RESHAPE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  ReshapeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    num_axes_ = get_single_argument<int>("num_axes", -1);
    CHECK_GE(num_axes_, -1);
    shape_ = get_repeated_argument<int>("shape");
  }

  void Forward() override;

 private:
  int axis_, num_axes_;
  VecInt shape_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESHAPE_OP_HPP
