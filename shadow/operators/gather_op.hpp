#ifndef SHADOW_OPERATORS_GATHER_OP_HPP
#define SHADOW_OPERATORS_GATHER_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class GatherOp : public Operator {
 public:
  GatherOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 0);
    indexes_value_ = get_repeated_argument<int>("indexes");
  }

  void Forward() override;

 private:
  int axis_;
  VecInt indexes_value_;
};

namespace Vision {

template <typename T>
void Gather(const T *in_data, const int *indexes_data, int num_indexes,
            int gather_dim, int inner_num, int count, T *out_data,
            Context *context);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_GATHER_OP_HPP
