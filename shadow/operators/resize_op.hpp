#ifndef SHADOW_OPERATORS_RESIZE_OP_HPP
#define SHADOW_OPERATORS_RESIZE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ResizeOp : public Operator {
 public:
  ResizeOp(const shadow::OpParam& op_param, Workspace* ws)
      : Operator(op_param, ws) {
    if (has_argument("size")) {
      const auto& size = get_repeated_argument<int>("size");
      CHECK_LE(size.size(), 2);
      if (size.empty()) {
        out_h_ = out_w_ = get_single_argument<int>("size", 0);
      } else if (size.size() == 1) {
        out_h_ = out_w_ = size[0];
      } else {
        out_h_ = size[0], out_w_ = size[1];
      }
    } else {
      out_h_ = get_single_argument<int>("out_h", 0);
      out_w_ = get_single_argument<int>("out_w", 0);
    }
    const auto& scale = get_repeated_argument<float>("scale");
    CHECK_LE(scale.size(), 2);
    if (scale.empty()) {
      scale_h_ = scale_w_ = get_single_argument<float>("scale", 1);
    } else if (scale.size() == 1) {
      scale_h_ = scale_w_ = scale[0];
    } else {
      scale_h_ = scale[0], scale_w_ = scale[1];
    }
    type_ = get_single_argument<int>("type", 1);
  }

  void Forward() override;

 private:
  int out_h_, out_w_, type_;
  float scale_h_, scale_w_;
};

namespace Vision {

template <typename T>
void Resize(const T* in_data, const VecInt& in_shape, int type,
            const VecInt& out_shape, T* out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_RESIZE_OP_HPP
