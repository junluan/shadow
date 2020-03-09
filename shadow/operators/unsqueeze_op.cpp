#include "unsqueeze_op.hpp"

namespace Shadow {

void UnsqueezeOp::Forward() {
  const auto bottom = bottoms(0);
  auto top = tops(0);

  CHECK_NE(bottom, top);

  int num_axes = bottom->num_axes();

  VecInt axes;
  for (auto axis : axes_) {
    if (axis < 0) {
      axis += num_axes + 1;
    }
    axes.push_back(axis);
  }

  VecInt top_shape(num_axes + axes.size(), 0);
  for (auto axis : axes) {
    top_shape[axis] = 1;
  }
  int d = 0;
  for (auto &dim : top_shape) {
    if (dim == 0) {
      dim = bottom->shape(d++);
    }
  }
  CHECK_EQ(d, num_axes);

  top->share_data(bottom->data<float>(), top_shape);
  CHECK_EQ(top->count(), bottom->count());
}

REGISTER_OPERATOR(Unsqueeze, UnsqueezeOp);

}  // namespace Shadow
