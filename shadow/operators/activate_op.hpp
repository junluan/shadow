#ifndef SHADOW_OPERATORS_ACTIVATE_OP_HPP
#define SHADOW_OPERATORS_ACTIVATE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ActivateOp : public Operator {
 public:
  explicit ActivateOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ActivateOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int activate_type_;
  bool channel_shared_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ACTIVATE_OP_HPP
