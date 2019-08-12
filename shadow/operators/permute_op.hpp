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

template <typename T, typename Dtype>
void Permute(const T *in_data, int count, int num_axes,
             const Dtype *permute_order, const Dtype *old_steps,
             const Dtype *new_steps, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PERMUTE_OP_HPP
