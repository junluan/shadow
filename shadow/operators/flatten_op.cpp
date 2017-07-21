#include "flatten_op.hpp"

namespace Shadow {

void FlattenOp::Setup() {
  CHECK_NE(tops<float>(0), bottoms<float>(0));

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

  for (int i = 0; i < axis_; ++i) {
    top->add_shape(bottom->shape(i));
  }
  top->add_shape(bottom->count(axis_, end_axis_ + 1));
  for (int i = end_axis_ + 1; i < bottom->num_axes(); ++i) {
    top->add_shape(bottom->shape(i));
  }

  DLOG(INFO) << op_name_ << ": "
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << Util::format_vector(top->shape(), ",", "(", ")");
}

void FlattenOp::Forward() {
  mutable_tops<float>(0)->share_data(*bottoms<float>(0));
}

void FlattenOp::Release() {
  // DLOG(INFO) << "Free FlattenOp!";
}

REGISTER_OPERATOR(Flatten, FlattenOp);

}  // namespace Shadow
