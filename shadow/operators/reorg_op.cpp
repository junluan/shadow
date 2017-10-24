#include "reorg_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void ReorgOp::Setup() {
  stride_ = get_single_argument<int>("stride", 2);
  CHECK_EQ(bottoms<float>(0)->shape(2) % stride_, 0);
  CHECK_EQ(bottoms<float>(0)->shape(3) % stride_, 0);
}

void ReorgOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  CHECK_NE(bottom, top);

  int in_c = bottom->shape(1), in_h = bottom->shape(2), in_w = bottom->shape(3);

  VecInt top_shape = bottom->shape();
  top_shape[1] = in_c * stride_ * stride_;
  top_shape[2] = in_h / stride_;
  top_shape[3] = in_w / stride_;
  top->reshape(top_shape);

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ReorgOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Reorg(bottom->data(), bottom->shape(), stride_, top->mutable_data());
}

void ReorgOp::Release() {
  // DLOG(INFO) << "Free ReorgOp!";
}

REGISTER_OPERATOR(Reorg, ReorgOp);

}  // namespace Shadow
