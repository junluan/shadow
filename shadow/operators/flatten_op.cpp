#include "flatten_op.hpp"

namespace Shadow {

void FlattenOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

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

  top->share_data(bottom->data<float>(), top_shape);
  CHECK_EQ(top->count(), bottom->count());
}

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
