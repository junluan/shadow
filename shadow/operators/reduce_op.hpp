#ifndef SHADOW_OPERATORS_REDUCE_OP_HPP
#define SHADOW_OPERATORS_REDUCE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReduceOp : public Operator {
 public:
  explicit ReduceOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", 0);
    axes_ = get_repeated_argument<int>("axes");
    keep_dims_ = get_single_argument<bool>("keep_dims", true);
  }

  void Forward() override;

  enum { kProd = 0, kSum = 1, kMax = 2, kMin = 3, kAvg = 4 };

 private:
  int operation_;
  bool keep_dims_;
  VecInt axes_, bottom_shape_, top_shape_, list_value_, offset_value_;
};

namespace Vision {

template <typename T>
void Reduce(const T *in_data, const int *list_data, const int *offset_data,
            int num_list, int operation, int count, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REDUCE_OP_HPP
