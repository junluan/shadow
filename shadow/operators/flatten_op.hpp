#ifndef SHADOW_OPERATORS_FLATTEN_OP_HPP
#define SHADOW_OPERATORS_FLATTEN_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class FlattenOp : public Operator {
 public:
  FlattenOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    CHECK_GE(axis_, 0);
    end_axis_ = get_single_argument<int>("end_axis", -1);
  }

  void Forward() override;

 private:
  int axis_, end_axis_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_FLATTEN_OP_HPP
