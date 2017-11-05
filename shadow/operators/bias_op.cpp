#include "bias_op.hpp"
#include "core/vision.hpp"

namespace Shadow {

void BiasOp::Reshape() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  if (bottom != top) {
    top->reshape(bottom->shape());
  }

  int start_axis = bias_->num_axes() == 0 ? 0 : axis_;
  CHECK_GE(bottom->num_axes(), start_axis + bias_->num_axes());
  for (int i = 0; i < bias_->num_axes(); ++i) {
    CHECK_EQ(bottom->shape(start_axis + i), bias_->shape(i));
  }
  bias_dim_ = bias_->count();
  inner_dim_ = bottom->count(start_axis + bias_->num_axes());

  DLOG(INFO) << op_name_ << "(" << op_type_ << "): " << bottom->name()
             << Util::format_vector(bottom->shape(), ",", "(", ")") << " -> "
             << top->name() << Util::format_vector(top->shape(), ",", "(", ")");
}

void BiasOp::Forward() {
  const auto *bottom = bottoms<float>(0);
  auto *top = mutable_tops<float>(0);

  Vision::Bias(bottom->data(), bottom->count(), bias_->data(), bias_dim_,
               inner_dim_, top->mutable_data());
}

REGISTER_OPERATOR(Bias, BiasOp);

}  // namespace Shadow
