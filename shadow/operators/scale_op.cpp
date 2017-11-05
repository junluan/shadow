#include "scale_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void ScaleOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = scale_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + scale_->num_axes());
  for (int i = 0; i < scale_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), scale_->shape(i));
  }
  scale_dim_ = scale_->count();
  inner_dim_ = bottom->count(start_axis + scale_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void ScaleOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Scale(bottom->data(), bottom->count(), scale_->data(), bias_->data(),
                scale_dim_, inner_dim_, top->mutable_data());
}

REGISTER_OPERATOR(Scale, ScaleOp);

}  // namespace Shadow
