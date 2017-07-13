#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  explicit ReorgOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~ReorgOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  int stride_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
