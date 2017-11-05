#include "flatten_op.hpp"

namespace Shadow {

void FlattenOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  VecInt top_shape;
  for (int i = 0; i < axis_; ++i) {
    top_shape.push_back(bottom->shape(i));
  }
  top_shape.push_back(bottom->count(axis_, end_axis_ + 1));
  for (int i = end_axis_ + 1; i < bottom->num_axes(); ++i) {
    top_shape.push_back(bottom->shape(i));
  }
  top->set_shape(top_shape);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void FlattenOp::Forward() {
  mutable_tops<float>(0)->share_data(*bottoms<float>(0));
}

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
