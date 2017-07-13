#ifndef SHADOW_OPERATORS_PERMUTE_OP_HPP
#define SHADOW_OPERATORS_PERMUTE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PermuteOp : public Operator {
 public:
  explicit PermuteOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~PermuteOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  int num_axes_;
  VecInt permute_order_data_;

  BlobI permute_order_, old_steps_, new_steps_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PERMUTE_OP_HPP
