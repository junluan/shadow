#ifndef SHADOW_OPERATORS_ELTWISE_OP_HPP
#define SHADOW_OPERATORS_ELTWISE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class EltwiseOp : public Operator {
 public:
  explicit EltwiseOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~EltwiseOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int operation_, coeff_size_;
  VecFloat coeff_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ELTWISE_OP_HPP
