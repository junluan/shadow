#ifndef SHADOW_OPERATORS_INPUT_OP_HPP
#define SHADOW_OPERATORS_INPUT_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class InputOp : public Operator {
 public:
  InputOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    for (int n = 0; n < tops_size(); ++n) {
      auto *top = mutable_tops<float>(n);
      const auto &top_name = top->name();
      if (has_argument(top_name)) {
        top->reshape(get_repeated_argument<int>(top_name, VecInt{}));
      }
    }
  }

  void Forward() override;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_INPUT_OP_HPP
