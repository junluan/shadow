#ifndef SHADOW_OPERATORS_BIAS_OP_HPP
#define SHADOW_OPERATORS_BIAS_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BiasOp : public Operator {
 public:
  explicit BiasOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~BiasOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  int axis_, num_axis_, bias_dim_, inner_dim_;

  BlobF *bias_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BIAS_OP_HPP
