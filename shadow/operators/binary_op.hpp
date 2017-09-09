#ifndef SHADOW_OPERATORS_BINARY_OP_HPP
#define SHADOW_OPERATORS_BINARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BinaryOp : public Operator {
 public:
  explicit BinaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~BinaryOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  enum { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kPow = 4, kMax = 5, kMin = 6 };

  int operation_;

  BlobF *scalar_ = nullptr;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BINARY_OP_HPP
