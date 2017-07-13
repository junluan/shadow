#ifndef SHADOW_OPERATORS_PRIOR_BOX_OP_HPP
#define SHADOW_OPERATORS_PRIOR_BOX_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PriorBoxOp : public Operator {
 public:
  explicit PriorBoxOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {}
  ~PriorBoxOp() { Release(); }

  virtual void Setup() override;
  virtual void Reshape() override;
  virtual void Forward() override;
  virtual void Release() override;

 private:
  int num_priors_;
  float step_, offset_;
  bool flip_, clip_, is_initial_;
  VecFloat min_sizes_, max_sizes_, aspect_ratios_, variance_, top_data_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PRIOR_BOX_OP_HPP
