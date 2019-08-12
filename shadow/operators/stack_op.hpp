#ifndef SHADOW_OPERATORS_STACK_OP_HPP
#define SHADOW_OPERATORS_STACK_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class StackOp : public Operator {
 public:
  StackOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
  }

  void Forward() override;

 private:
  int axis_ = 0;
};

namespace Vision {

template <typename T>
void Stack(const T *in_data, int count, int num_stacks, int stack_size,
           int top_stack_axis, int offset_stack_axis, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_STACK_OP_HPP
