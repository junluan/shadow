#ifndef SHADOW_OPERATORS_RESIZE_OP_HPP
#define SHADOW_OPERATORS_RESIZE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ResizeOp : public Operator {
 public:
  ResizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    out_h_ = get_single_argument<int>("out_h", 0);
    out_w_ = get_single_argument<int>("out_w", 0);
    if (has_argument("scale")) {
      scale_h_ = scale_w_ = get_single_argument<float>("scale", 1);
    } else {
      scale_h_ = get_single_argument<float>("scale_h", 1);
      scale_w_ = get_single_argument<float>("scale_w", 1);
    }
    type_ = get_single_argument<int>("type", 1);
  }

  void Forward() override;

 private:
  int out_h_, out_w_, type_;
  float scale_h_ = 1.f, scale_w_ = 1.f;
};

namespace Vision {

template <typename T>
void Resize(const T* in_data, const VecInt& in_shape, int type,
            const VecInt& out_shape, T* out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESIZE_OP_HPP
