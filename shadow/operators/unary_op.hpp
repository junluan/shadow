#ifndef SHADOW_OPERATORS_UNARY_OP_HPP
#define SHADOW_OPERATORS_UNARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class UnaryOp : public Operator {
 public:
  explicit UnaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~UnaryOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

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
    kCeil = 12
  };

  int operation_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_UNARY_OP_HPP
