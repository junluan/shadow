#ifndef SHADOW_OPERATORS_ACTIVATE_OP_HPP
#define SHADOW_OPERATORS_ACTIVATE_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ActivateOp : public Operator {
 public:
  explicit ActivateOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    activate_type_ = get_single_argument<int>("type", 1);
    slope_ = get_single_argument<float>("slope", 0.1);
    channel_shared_ = get_single_argument<bool>("channel_shared", false);
    CHECK_GE(activate_type_, 0);
    CHECK_LE(activate_type_, 5);
  }

  void Forward() override;

 private:
  enum {
    kPRelu = 0,
    kRelu = 1,
    kLeaky = 2,
    kSigmoid = 3,
    kSoftPlus = 4,
    kTanh = 5
  };

  int activate_type_;
  float slope_;
  bool channel_shared_;
};

namespace Vision {

template <typename T>
void Activate(T *data, int count, int type, float slope = 0.1);

template <typename T>
void PRelu(T *data, const VecInt &in_shape, bool channel_shared,
           const T *slope_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_ACTIVATE_OP_HPP
