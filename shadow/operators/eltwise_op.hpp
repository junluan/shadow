#ifndef SHADOW_OPERATORS_ELTWISE_OP_HPP
#define SHADOW_OPERATORS_ELTWISE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class EltwiseOp : public Operator {
 public:
  explicit EltwiseOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~EltwiseOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int operation_, coeff_size_;
  VecFloat coeff_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ELTWISE_OP_HPP
