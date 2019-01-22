#ifndef SHADOW_OPERATORS_BATCH_NORM_OP_HPP
#define SHADOW_OPERATORS_BATCH_NORM_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BatchNormOp : public Operator {
 public:
  explicit BatchNormOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    use_global_stats_ = get_single_argument<bool>("use_global_stats", true);
    eps_ = get_single_argument<float>("eps", 1e-5);
  }

  void Forward() override;

 private:
  bool use_global_stats_;
  float eps_;
};

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BATCH_NORM_OP_HPP
