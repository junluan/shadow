#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  explicit ReorgOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ReorgOp() { Release(); }

  void Setup();
  void Reshape();
  void Forward();
  void Release();

 private:
  int stride_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
