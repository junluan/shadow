#include "flatten_op.hpp"

namespace Shadow {

void FlattenOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int num_axes = bottom->num_axes();
  if (end_axis_ == -1) {
    end_axis_ = num_axes - 1;
  }
  CHECK_LT(end_axis_, num_axes);
  CHECK_LE(axis_, end_axis_);

  VecInt top_shape;
  for (int d = 0; d < axis_; ++d) {
    top_shape.push_back(bottom->shape(d));
  }
  top_shape.push_back(bottom->count(axis_, end_axis_ + 1));
  for (int d = end_axis_ + 1; d < bottom->num_axes(); ++d) {
    top_shape.push_back(bottom->shape(d));
  }

  top->clear();
  top->set_shape(top_shape);
  CHECK_EQ(top->count(), bottom->count());

  top->share_data(*bottom);

  DLOG(INFO) << debug_log();
}

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
