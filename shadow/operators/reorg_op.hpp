#ifndef SHADOW_OPERATORS_REORG_OP_HPP
#define SHADOW_OPERATORS_REORG_OP_HPP

#include "core/operator.hpp"

namespace Shadow {

class ReorgOp : public Operator {
 public:
  explicit ReorgOp(const shadow::OpParam &op_param, Workspace *ws)
      : Operator(op_param, ws) {
    stride_ = get_single_argument<int>("stride", 2);
    CHECK_EQ(bottoms<float>(0)->shape(2) % stride_, 0);
    CHECK_EQ(bottoms<float>(0)->shape(3) % stride_, 0);
  }

  void Reshape() override;
  void Forward() override;

 private:
  int stride_;
};

namespace Vision {

template <typename T>
void Reorg(const T *in_data, const VecInt &in_shape, int stride, T *out_data);

}  // namespace Vision

}  // namespace Shadow

#endif  // SHADOW_OPERATORS_REORG_OP_HPP
