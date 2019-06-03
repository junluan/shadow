#ifndef SHADOW_OPERATORS_UNARY_OP_HPP
#define SHADOW_OPERATORS_UNARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class UnaryOp : public Operator {
 public:
  explicit UnaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 14);
  }

  void Forward() override;

 private:
  enum {
    kAbs = 0,
    kSquare = 1,
    kSqrt = 2,
    kLog = 3,
    kExp = 4,
    kSin = 5,
    kCos = 6,
    kTan = 7,
    kAsin = 8,
    kAcos = 9,
    kAtan = 10,
    kFloor = 11,
    kCeil = 12,
    kNeg = 13,
    kReciprocal = 14
  };

  int operation_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_UNARY_OP_HPP
