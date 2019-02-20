#ifndef SHADOW_OPERATORS_SQUEEZE_HPP
#define SHADOW_OPERATORS_SQUEEZE_HPP

#include "core/operator.hpp"

namespace Shadow {

class SqueezeOp : public Operator {
 public:
  explicit SqueezeOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axes_ = get_repeated_argument<int>("axes");
  }

  void Forward() override;

 private:
  VecInt axes_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SQUEEZE_HPP
