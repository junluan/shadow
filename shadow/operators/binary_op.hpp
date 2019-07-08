#ifndef SHADOW_OPERATORS_BINARY_OP_HPP
#define SHADOW_OPERATORS_BINARY_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class BinaryOp : public Operator {
 public:
  explicit BinaryOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    operation_ = get_single_argument<int>("operation", -1);
    CHECK_GE(operation_, 0);
    CHECK_LE(operation_, 6);
    if (has_argument("scalar")) {
      scalar_data_ = get_single_argument<float>("scalar", 0);
      has_scalar_arg_ = true;
    }
  }

  void Forward() override;

 private:
  enum { kAdd = 0, kSub = 1, kMul = 2, kDiv = 3, kPow = 4, kMax = 5, kMin = 6 };

  int operation_;
  float scalar_data_;
  bool has_scalar_arg_ = false, need_broadcast_ = false;
  VecInt bottom_shape_, scalar_shape_, top_shape_;
};

namespace Vision {

template <typename T>
void BroadcastBinary(const T *in_data, const int *in_shape,
                     const T *scalar_data, const int *scalar_shape,
                     int operation, int num_axes, int count,
                     const int *out_shape, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_BINARY_OP_HPP
