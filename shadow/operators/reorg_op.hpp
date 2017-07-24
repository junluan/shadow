#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  explicit ReorgOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ReorgOp() override { Release(); }

  void Setup() override;
  void Reshape() override;
  void Forward() override;
  void Release() override;

 private:
  int stride_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
