#ifndef SHADOW_OPERATORS_BINARY_OP_HPP
#define SHADOW_OPERATORS_BINARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BinaryOp : public Operator {
 public:
  explicit BinaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 6);
    if (has_argument("scalar")) {
      scalar_data_ = get_single_argument<float>("scalar", 0);
      has_scalar_arg_ = true;
    }
  }

  void Forward() override;

 private:
  enum { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kPow = 4, kMax = 5, kMin = 6 };

  int operation_;
  float scalar_data_;
  bool has_scalar_arg_ = false;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BINARY_OP_HPP
