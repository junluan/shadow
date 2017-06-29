#ifndef SHADOW_OPERATORS_RESHAPE_OP_HPP
#define SHADOW_OPERATORS_RESHAPE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReshapeOp : public Operator {
 public:
  explicit ReshapeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ReshapeOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int axis_, num_axes_, inferred_axis_, constant_count_;
  VecInt shape_, copy_axes_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESHAPE_OP_HPP
