#ifndef SHADOW_OPERATORS_PERMUTE_OP_HPP
#define SHADOW_OPERATORS_PERMUTE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PermuteOp : public Operator {
 public:
  PermuteOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    permute_order_value_ = get_repeated_argument<int>("order");
  }

  void Forward() override;

 private:
  VecInt permute_order_value_;
};

namespace Vision {

template <typename T>
void Permute(const T *in_data, int count, int num_axes,
             const int *permute_order, const int *old_steps,
             const int *new_steps, T *out_data, Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PERMUTE_OP_HPP
