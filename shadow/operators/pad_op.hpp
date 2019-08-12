#ifndef SHADOW_OPERATORS_PAD_OP_HPP
#define SHADOW_OPERATORS_PAD_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class PadOp : public Operator {
 public:
  PadOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    // [top, bottom, left, right]
    paddings_ = get_repeated_argument<int>("paddings");
    CHECK_EQ(paddings_.size(), 4);
    value_ = get_single_argument<float>("value", 0);
  }

  void Forward() override;

 private:
  float value_;
  VecInt paddings_;
};

namespace Vision {

template <typename T>
void Pad(const T *in_data, const VecInt &in_shape, const VecInt &paddings,
         const VecInt &out_shape, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_PAD_OP_HPP
