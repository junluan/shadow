#include "flatten_op.hpp"

namespace Shadow {

void FlattenOp::Setup() {
  axis_ = get_single_argument<int>("axis", 1);
  end_axis_ = get_single_argument<int>("end_axis", -1);
  int num_axes = bottoms<float>(0)->num_axes();
  if (end_axis_ == -1) {
    end_axis_ = num_axes - 1;
  }
  CHECK_GE(axis_, 0);
  CHECK_LT(end_axis_, num_axes);
  CHECK_LE(axis_, end_axis_);
}

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

void FlattenOp::Release() {
  // DLOG(INFO) << "Free FlattenOp!";
}

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
