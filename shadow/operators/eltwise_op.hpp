#ifndef SHADOW_OPERATORS_ELTWISE_OP_HPP
#define SHADOW_OPERATORS_ELTWISE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class EltwiseOp : public Operator {
 public:
  explicit EltwiseOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", 1);
    coeff_ = get_repeated_argument<float>("coeff");
  }

  void Forward() override;

 private:
  enum { kProd = 0, kSum = 1, kMax = 2, kMin = 3 };

  int operation_;
  VecFloat coeff_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ELTWISE_OP_HPP
