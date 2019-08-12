#ifndef SHADOW_OPERATORS_SCALE_OP_HPP
#define SHADOW_OPERATORS_SCALE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ScaleOp : public Operator {
 public:
  ScaleOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    axis_ = get_single_argument<int>("axis", 1);
    has_scale_ = get_single_argument<bool>("has_scale", true);
    has_bias_ = get_single_argument<bool>("has_bias", true);
    scale_value_ = get_repeated_argument<float>("scale_value");
    bias_value_ = get_repeated_argument<float>("bias_value");
  }

  void Forward() override;

 private:
  int axis_, scale_dim_, inner_dim_;
  bool has_scale_, has_bias_;
  VecFloat scale_value_, bias_value_;
};

namespace Vision {

template <typename T>
void Scale(const T *in_data, int count, const T *scale_data, const T *bias_data,
           int scale_dim, int inner_dim, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_SCALE_OP_HPP
