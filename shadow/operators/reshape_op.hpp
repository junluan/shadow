#ifndef SHADOW_OPERATORS_RESHAPE_OP_HPP
#define SHADOW_OPERATORS_RESHAPE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  explicit ReshapeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    num_axes_ = get_single_argument<int>("num_axes", -1);
    CHECK_GE(num_axes_, -1);
    shape_ = get_repeated_argument<int>("shape");

    inferred_axis_ = -1;
    copy_axes_.clear();
    constant_count_ = 1;
    for (int i = 0; i < shape_.size(); ++i) {
      int top_dim = shape_[i];
      if (top_dim == 0) {
        copy_axes_.push_back(i);
      } else if (top_dim == -1) {
        CHECK_EQ(inferred_axis_, -1);
        inferred_axis_ = i;
      } else {
        constant_count_ *= top_dim;
      }
    }
  }

  void Reshape() override;
  void Forward() override;

 private:
  int axis_, num_axes_, inferred_axis_, constant_count_;
  VecInt shape_, copy_axes_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESHAPE_OP_HPP
