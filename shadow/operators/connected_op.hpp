#ifndef SHADOW_OPERATORS_CONNECTED_OP_HPP
#define SHADOW_OPERATORS_CONNECTED_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ConnectedOp : public Operator {
 public:
  ConnectedOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    CHECK(has_argument("num_output"));
    num_output_ = get_single_argument<int>("num_output", 0);
    bias_term_ = get_single_argument<bool>("bias_term", true);
    transpose_ = get_single_argument<bool>("transpose", true);
  }

  void Forward() override;

 private:
  int num_output_;
  bool bias_term_, transpose_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_CONNECTED_OP_HPP
