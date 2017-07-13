#ifndef SHADOW_OPERATORS_LRN_OP_HPP
#define SHADOW_OPERATORS_LRN_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class LRNOp : public Operator {
 public:
  explicit LRNOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~LRNOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  int size_, norm_region_;
  float alpha_, beta_, k_;

  BlobF scale_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_LRN_OP_HPP
