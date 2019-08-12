#ifndef SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP
#define SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class InstanceNormOp : public Operator {
 public:
  InstanceNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    eps_ = get_single_argument<float>("eps", 1e-5);
  }

  void Forward() override;

 private:
  float eps_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_INSTANCE_NORM_OP_HPP
